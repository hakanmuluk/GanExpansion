"""Single-skill trajectory manifold teachers and coordinate conversions.

Analytic teachers are explicitly diagnostic, not a learned composition result.
Each skill receives only its own region. The transport frame uses fixed progress,
with two free transverse coordinates; 11/10 modifier coordinates replace one.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F

NODES=65
ANCHORS=7
FREE=10
BOUND=math.sqrt(3)*.02


def matrices(device,dtype=torch.float32):
    t=torch.linspace(0,1,NODES,device=device,dtype=dtype)
    a=torch.linspace(0,1,ANCHORS,device=device,dtype=dtype)
    weights=(1-(t[:,None]-a[None]).abs()*(ANCHORS-1)).clamp_min(0)
    return t,weights


def amplitude(base_z):
    magnitude=.035+.04*torch.erf(base_z[:,0].abs()/math.sqrt(2))
    return torch.where(base_z[:,0]>=0,magnitude,-magnitude)


def raw_path(z,geometry):
    t,weights=matrices(z.device,z.dtype)
    offsets=.08*torch.tanh(z[:,:FREE].reshape(-1,5,2))
    pad=z.new_zeros((len(z),1,2));anchors=torch.cat([pad,offsets,pad],1)
    transverse=torch.einsum('pa,bad->bpd',weights,anchors)
    transverse=transverse+torch.stack([geometry[:,1,None]*torch.sin(math.pi*t)[None],torch.zeros_like(transverse[:,:,0])],-1)
    return transverse,t,weights


def oracle(skill,z,geometry,margin=.005):
    raw,t,weights=raw_path(z,geometry)
    if skill=='waypoint':
        center=geometry[:,7:9]
        v=z[:,:FREE].reshape(-1,5,2)[:,2]
        target=center+.85*geometry[:,9,None]*v/torch.sqrt(1+v.square().sum(-1,keepdim=True))
        return raw+weights[:,3][None,:,None]*(target-raw[:,NODES//2])[:,None]
    if skill!='keepout':raise ValueError(skill)
    along=t[None]*geometry[:,0,None]
    center=geometry[:,3:5];radius=geometry[:,5]+BOUND+margin
    d0=(along[:,:-1]-geometry[:,2,None]);d1=along[:,1:]-geometry[:,2,None]
    interval_distance=torch.where(d0*d1<=0,torch.zeros_like(d0),torch.minimum(d0.abs(),d1.abs()))
    needed=torch.sqrt((radius[:,None].square()-interval_distance.square()).clamp_min(1e-12))
    intersect=interval_distance<radius[:,None]
    needed=torch.where(intersect,needed,torch.zeros_like(needed))
    zero=torch.zeros_like(needed[:,:1])
    node_needed=torch.maximum(torch.cat([zero,needed],1),torch.cat([needed,zero],1))
    node_active=torch.maximum(torch.cat([zero,intersect.float()],1),torch.cat([intersect.float(),zero],1))>0
    angle=z[:,FREE]
    direction=torch.stack([angle.cos(),angle.sin()],-1)
    signed=((raw-center[:,None])*direction[:,None]).sum(-1)
    amount=torch.where(node_active,(node_needed-signed).clamp_min(0),torch.zeros_like(signed))
    result=raw+amount[:,:,None]*direction[:,None]
    # Endpoints represent the unchanged original pick/place columns.
    result=torch.cat([raw[:,:1],result[:,1:-1],raw[:,-1:]],1)
    return result


def paths_xyz(transverse,geometry):
    t,_=matrices(transverse.device,transverse.dtype)
    return torch.cat([t[None,:,None]*geometry[:,0,None,None],transverse],-1)


def segment_gap(transverse,geometry,skill):
    xyz=paths_xyz(transverse,geometry)
    center=geometry[:,2:5] if skill=='keepout' else geometry[:,6:9]
    radius=geometry[:,5]+BOUND if skill=='keepout' else geometry[:,9]
    a,b=xyz[:,:-1],xyz[:,1:];v=b-a
    frac=((center[:,None]-a)*v).sum(-1)/v.square().sum(-1).clamp_min(1e-12)
    distance=(a+frac.clamp(0,1)[:,:,None]*v-center[:,None]).norm(dim=-1)
    return distance.min(-1).values-radius


def geometry_cases(cases,device='cpu'):
    records=[]
    for case in cases:
        start=np.asarray(case['initial_cube']);goal=np.asarray(case['goal']);delta=goal[:2]-start[:2]
        length=np.linalg.norm(delta);axis=delta/length;perp=np.array([-axis[1],axis[0]])
        z=np.asarray(case['base_z']);amp=(.035+.04*math.erf(abs(float(z[0]))/math.sqrt(2)))*(1 if z[0]>=0 else -1)
        def relative(spec):
            rel=np.asarray(spec[:2])-start[:2]
            return [rel@axis,rel@perp,float(spec[2])-.2,float(spec[3])]
        records.append([length,amp,*relative(case['keepout']),*relative(case['waypoint'])])
    return torch.tensor(records,dtype=torch.float32,device=device)


def world_path(transverse,case):
    t=np.linspace(0,1,len(transverse));start=np.asarray(case['initial_cube']);goal=np.asarray(case['goal'])
    delta=goal[:2]-start[:2];axis=delta/np.linalg.norm(delta);perp=np.array([-axis[1],axis[0]])
    xy=start[:2]+t[:,None]*delta+np.asarray(transverse)[:,:1]*perp
    return np.c_[xy,.2+np.asarray(transverse)[:,1]]
