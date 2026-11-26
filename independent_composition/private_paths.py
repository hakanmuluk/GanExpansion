"""Private path-conditioned controllers and a generic trajectory decoder.

No constraint predicate, environment, geometric projector or shared learned
executor is used by these forward functions. A fixed interpolation prior is a
representation choice, not a learned shared module.
"""
from dataclasses import dataclass, replace
import math
from pathlib import Path
import torch
from torch import nn


POINTS=65
FREE=10


def path_context(own_context,position,path,index,done):
    n=len(path);own_context=own_context.expand(n,-1);position=position.expand(n,-1)
    index=torch.as_tensor(index,device=path.device,dtype=torch.long).reshape(-1).expand(n)
    ids=(index[:,None]+torch.arange(16,device=path.device)[None]).clamp_max(len(path[0])-1)
    local=path.gather(1,ids[:,:,None].expand(-1,-1,path.shape[-1]))-position[:,None]
    global_ids=torch.linspace(0,path.shape[1]-1,7,device=path.device).round().long()
    global_path=path[:,global_ids]-position[:,None]
    done=torch.as_tensor(done,device=path.device,dtype=path.dtype).reshape(-1,1).expand(n,1)
    return torch.cat([own_context,10*local.flatten(1),10*global_path.flatten(1),
                      index[:,None]/(path.shape[1]-1),done],-1)


def initialize_controller(skill,device='cuda',seed=42):
    from cube_addons.models import GatedGenerator,AdapterConfig
    source=Path(f'cube_addon_artifacts/final/{skill}_s{seed}/adapter.pt')
    ck=torch.load(source,map_location='cpu',weights_only=False)
    old=AdapterConfig(**ck['adapter_config']);config=replace(old,context_dim=91,transport_gate=False)
    actor=GatedGenerator('cube/artifacts/final/base_policy.pt',config,device)
    weights={k:v.clone() for k,v in ck['addon'].items()}
    previous=weights['context.0.weight'];current=actor.addon_state()['context.0.weight'].detach().cpu().clone().zero_()
    prefix=actor.base_config.obs_dim+old.context_dim
    current[:,:prefix]=previous[:,:prefix]
    current[:,prefix+config.context_dim-old.context_dim:]=previous[:,prefix:]
    weights['context.0.weight']=current
    missing,unexpected=actor.load_state_dict(weights,strict=False)
    assert not unexpected and all(k.startswith(('base.','original_encoder.')) for k in missing)
    return actor,source


def load_controller(path,device='cuda'):
    from cube_addons.models import GatedGenerator,AdapterConfig
    from .legacy import digest_file
    ck=torch.load(path,map_location=device,weights_only=False)
    assert ck['base_sha256']==digest_file('cube/artifacts/final/base_policy.pt')
    actor=GatedGenerator('cube/artifacts/final/base_policy.pt',AdapterConfig(**ck['adapter_config']),device)
    missing,unexpected=actor.load_state_dict(ck['addon'],strict=False)
    assert not unexpected and all(k.startswith(('base.','original_encoder.')) for k in missing)
    return actor.eval().requires_grad_(False),ck


@dataclass
class RouteConfig:
    code_dim:int
    context_dim:int=6
    coordinates:int=3
    points:int=POINTS
    hidden:int=256


class RouteDecoder(nn.Module):
    """Same residual trajectory architecture for either skill."""
    def __init__(self,config):
        super().__init__();self.config=c=config
        inputs=c.code_dim+c.context_dim+c.points*c.coordinates
        self.net=nn.Sequential(nn.Linear(inputs,c.hidden),nn.SiLU(),nn.Linear(c.hidden,c.hidden),
                               nn.SiLU(),nn.Linear(c.hidden,c.hidden),nn.SiLU(),nn.Linear(c.hidden,c.points*c.coordinates))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('context_mean',torch.zeros(c.context_dim))
        self.register_buffer('context_std',torch.ones(c.context_dim))
        self.register_buffer('coordinate_scale',torch.tensor([.4,.1,.1])[:c.coordinates])

    @torch.no_grad()
    def fit_normalization(self,context):
        self.context_mean.copy_(context.mean(0));self.context_std.copy_(context.std(0).clamp_min(.005))

    def forward(self,code,own_context,proposal):
        c=self.config
        normalized=((own_context-self.context_mean)/self.context_std).clamp(-10,10)
        x=torch.cat([code,normalized,(proposal/self.coordinate_scale).flatten(1)],-1)
        residual=self.net(x).reshape(-1,c.points,c.coordinates)*self.coordinate_scale
        envelope=torch.sin(torch.linspace(0,math.pi,c.points,device=code.device,dtype=code.dtype))[None,:,None]
        # Explicit endpoint identity is independent of either skill's meaning.
        interior=proposal[:,1:-1]+envelope[:,1:-1]*residual[:,1:-1]
        return torch.cat([proposal[:,:1],interior,proposal[:,-1:]],1)


def spline_proposal(code,own_context):
    """A fixed free curve prior; its weights are not learned or shared."""
    t=torch.linspace(0,1,POINTS,device=code.device,dtype=code.dtype)
    anchors=torch.linspace(0,1,7,device=code.device,dtype=code.dtype)
    weights=(1-(t[:,None]-anchors[None]).abs()*6).clamp_min(0)
    offsets=.08*torch.tanh(code[:,:FREE].reshape(-1,5,2))
    zero=code.new_zeros(len(code),1,2);offsets=torch.cat([zero,offsets,zero],1)
    transverse=torch.einsum('pa,bad->bpd',weights,offsets)
    y=transverse[:,:,0]+own_context[:,1,None]*torch.sin(math.pi*t)[None]
    return torch.stack([t[None]*own_context[:,0,None],y,.2+transverse[:,:,1]],-1)


def own_geometry(full_geometry,skill):
    # The neural model itself accepts only its own six context values.
    return torch.cat([full_geometry[:,:2],full_geometry[:,2:6] if skill=='keepout' else full_geometry[:,6:10]],-1)


def world_path(relative,start,goal):
    delta=goal[:,:2]-start[:,:2];axis=delta/delta.norm(dim=-1,keepdim=True)
    perpendicular=torch.stack([-axis[:,1],axis[:,0]],-1)
    xy=start[:,None,:2]+relative[:,:,:1]*axis[:,None]+relative[:,:,1:2]*perpendicular[:,None]
    return torch.cat([xy,relative[:,:,2:]],-1)
