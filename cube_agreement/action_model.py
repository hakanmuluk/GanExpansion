"""Path-conditioned StyleGAN action adapter, retaining the original frozen base.

The new context carries a global route and a local future window. It generates
all 16 robot controls with the original style-modulation/LoRA architecture; no
identified proportional controller is used by this actor at inference.
"""
from dataclasses import asdict,replace
from pathlib import Path
import hashlib,math
import numpy as np
import torch
from .manifolds import NODES

CONTEXT=90
GLOBAL_INDICES=[0,11,21,32,43,53,64]


def path_context(reference,cube,path,index):
    # Supports an arbitrary candidate batch; reference/cube may have one row.
    n=len(path);reference=reference.expand(n,-1);cube=cube.expand(n,-1)
    if not torch.is_tensor(index):index=torch.full((n,),int(index),device=path.device,dtype=torch.long)
    elif len(index)==1:index=index.expand(n)
    ids=(index[:,None]+torch.arange(16,device=path.device)[None]).clamp_max(NODES-1)
    local=path.gather(1,ids[:,:,None].expand(-1,-1,3))-cube[:,None]
    global_path=path[:,GLOBAL_INDICES]-cube[:,None]
    return torch.cat([reference,local.flatten(1)*10,global_path.flatten(1)*10,index[:,None]/(NODES-1)],-1)


def world_path_tensor(transverse,start,goal):
    n=len(transverse);start=start.expand(n,-1);goal=goal.expand(n,-1)
    delta=goal[:,:2]-start[:,:2];axis=delta/delta.norm(dim=-1,keepdim=True);perp=torch.stack([-axis[:,1],axis[:,0]],-1)
    t=torch.linspace(0,1,NODES,device=transverse.device,dtype=transverse.dtype)
    xy=start[:,None,:2]+t[None,:,None]*delta[:,None]+transverse[:,:,:1]*perp[:,None]
    return torch.cat([xy,.2+transverse[:,:,1:]],-1)


def reference_context(trace):
    start,goal,bz=trace['initial_cube'],trace['goal'],trace['base_z'];cube=trace['poses'][:-1,3:];ee=trace['poses'][:-1,:3]
    delta=goal[:2]-start[:2];axis=delta/np.linalg.norm(delta);perp=np.array([-axis[1],axis[0]])
    amp=(.035+.04*math.erf(abs(float(bz[0]))/math.sqrt(2)))*(1 if bz[0]>=0 else -1)
    spec=np.r_[(start[:2]+goal[:2])/2+(amp+np.sign(amp)*.060)*perp,.20,.022]
    context=trace['contexts'].copy();context[:,:3]=(spec[:3]-np.array([.425,0,.2]))/np.array([.2,.3,.2]);context[:,3]=spec[3]/.1
    context[:,4:7]=(cube-spec[:3])/.15;context[:,7:10]=(ee-spec[:3])/.15;context[:,15]=trace['path_done']
    return context.astype(np.float32)


def initialize(source='cube_addon_artifacts/final/waypoint_s42/adapter.pt',base='cube/artifacts/final/base_policy.pt',device='cuda',ungated=False):
    from cube_addons.models import GatedGenerator,AdapterConfig
    checkpoint=torch.load(source,map_location='cpu',weights_only=False);old=AdapterConfig(**checkpoint['adapter_config']);config=replace(old,context_dim=CONTEXT,transport_gate=False if ungated else old.transport_gate)
    actor=GatedGenerator(base,config,device);weights={k:v.clone() for k,v in checkpoint['addon'].items()}
    name='context.0.weight';previous=weights[name];current=actor.addon_state()[name].detach().cpu().clone().zero_();prefix=actor.base_config.obs_dim+old.context_dim
    # Actual conditioning order is normalized state, context, base_z, mapped
    # modifier. Insert the 70 new path features immediately after old context.
    current[:,:prefix]=previous[:,:prefix];current[:,prefix+CONTEXT-old.context_dim:]=previous[:,prefix:];weights[name]=current
    missing,unexpected=actor.load_state_dict(weights,strict=False)
    assert not unexpected and all(k.startswith(('base.','original_encoder.')) for k in missing)
    return actor


def load(path,base='cube/artifacts/final/base_policy.pt',device='cuda'):
    from cube_addons.models import GatedGenerator,AdapterConfig
    ck=torch.load(path,map_location=device,weights_only=False);assert hashlib.sha256(Path(base).read_bytes()).hexdigest()==ck['base_sha256']
    actor=GatedGenerator(base,AdapterConfig(**ck['adapter_config']),device);missing,unexpected=actor.load_state_dict(ck['addon'],strict=False)
    assert not unexpected and all(k.startswith(('base.','original_encoder.')) for k in missing)
    return actor.eval().requires_grad_(False)
