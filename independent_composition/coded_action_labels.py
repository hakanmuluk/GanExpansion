"""TRAINING ONLY: counterfactual own-skill controls at recorded carry states.

One observed state is paired with varied OWN regions and behavior codes. This
prevents learning the code/region from correlated motion states. No joint scene
or other skill region is supplied. These teachers never run in policy inference.
"""
import math
import torch
from cube_agreement.manifolds import oracle,amplitude,BOUND
from cube.cube_gan.style import decode_style


@torch.no_grad()
def cube_counterfactuals(batch,mean,std,skill,margin=.005,waypoint_core=1.):
    from .inference_guard import inference_active,forbidden
    if inference_active():forbidden()
    assert margin>=0 and 0<waypoint_core<=1
    s=batch['state'];c=batch['context'];bz=batch['base_z']
    good=(c[:,13]>.5)&(c[:,14]>.5)&(c[:,10]>.05)&(c[:,10]<( .72 if skill=='keepout' else .46))
    ids=torch.nonzero(good).flatten()
    if not len(ids):return None
    s=s[ids];c=c[ids].clone();bz=bz[ids];n=len(s)
    origin=s.new_tensor([.425,0.,0.]);start=s[:,31:34]/10+origin;goal=s[:,28:31]/10+origin
    delta=goal[:,:2]-start[:,:2];length=delta.norm(dim=-1);axis=delta/length[:,None]
    perp=torch.stack([-axis[:,1],axis[:,0]],-1);amp=amplitude(bz)
    oldspec=c[:,:3]*s.new_tensor([.2,.3,.2])+s.new_tensor([.425,0.,.2])
    cube=oldspec+.15*c[:,4:7];ee=oldspec+.15*c[:,7:10]
    progress=.30+.40*torch.rand(n,device=s.device) if skill=='keepout' else s.new_full((n,),.5)
    lateral=amp+(.02*torch.rand(n,device=s.device)-.01) if skill=='keepout' else amp+amp.sign()*(.045+.030*torch.rand(n,device=s.device))
    height=.2+.01*(2*torch.rand(n,device=s.device)-1)
    radius=(.018+.012*torch.rand(n,device=s.device)) if skill=='keepout' else (.020+.008*torch.rand(n,device=s.device))
    spec=torch.cat([start[:,:2]+progress[:,None]*delta+lateral[:,None]*perp,height[:,None]],-1)
    raw=.7*torch.randn(n,len(mean),device=s.device)
    if skill=='keepout':raw[:,-1]=(2*torch.rand(n,device=s.device)-1)*math.pi
    geometry=s.new_zeros(n,10);geometry[:,:2]=torch.stack([length,amp],-1)
    offset=2 if skill=='keepout' else 6
    geometry[:,offset:offset+4]=torch.stack([progress*length,lateral,height-.2,radius],-1)
    assert not bool(torch.count_nonzero(geometry[:,6:10] if skill=='keepout' else geometry[:,2:6]))
    teacher_geometry=geometry.clone()
    if skill=='waypoint':teacher_geometry[:,9]*=waypoint_core
    transverse=oracle(skill,raw,teacher_geometry,margin=margin)
    t=torch.linspace(0,1,65,device=s.device)
    xy=start[:,None,:2]+t[None,:,None]*delta[:,None]+transverse[:,:,:1]*perp[:,None]
    path=torch.cat([xy,.2+transverse[:,:,1:2]],-1)
    supported=((path>=s.new_tensor([.255,-.30,.10]))&(path<=s.new_tensor([.595,.30,.32]))).all((1,2))
    index=(torch.floor(c[:,10]*64).long()+2).clamp(0,64);target=path[torch.arange(n,device=s.device),index]
    if skill=='keepout':
        d=target-cube;fraction=((spec-cube)*d).sum(-1)/d.square().sum(-1).clamp_min(1e-10)
        gap=(cube+fraction.clamp(0,1)[:,None]*d-spec).norm(dim=-1)-radius-BOUND
        supported&=gap>=.005
    _,_,speed=decode_style(bz);limit=torch.minimum(speed,s.new_full((n,),.65))
    action=s.new_zeros(n,5);action[:,:3]=((target-cube)*40*speed[:,None]).clamp(-limit[:,None],limit[:,None]);action[:,4]=1.
    c[:,:3]=(spec-s.new_tensor([.425,0.,.2]))/s.new_tensor([.2,.3,.2]);c[:,3]=radius/.1
    c[:,4:7]=(cube-spec)/.15;c[:,7:10]=(ee-spec)/.15;c[:,15]=0.
    if not bool(supported.any()):return None
    return {'state':s[supported],'context':c[supported],'base_z':bz[supported],
            'code':((raw-mean)/std)[supported],'action':action[supported]}
