"""Private TwoRoom path-conditioned adapters and an unlearned curve prior.

Only observed state/context, predicted paths, and frozen neural modules are
needed by these functions. Geometry teachers and constraints live elsewhere.
"""
from dataclasses import asdict
import math
import torch
from .private_paths import POINTS,RouteConfig,RouteDecoder


def canonical(state):
    from tworoom_gan.style import canonical_state
    return canonical_state(state)


def condition(state,own_context,base_z):
    state,_=canonical(state)
    return torch.cat([state,own_context,base_z,torch.sign(base_z[:,:1])],-1)


def control_points(code,condition_vector):
    """Five independent XY control points, with ordered progress coordinates.

The fixed curve is a representation prior. It does not enforce walls, region
avoidance, visitation, or goal ordering. Both skill networks use the same math.
"""
    n=len(code);z=code.reshape(n,5,2)
    spacing=torch.softmax(torch.cat([z[:,:,0],code.new_zeros(n,1)],-1),-1)
    fraction=spacing.cumsum(-1)[:,:5]
    start=condition_vector[:,:2];goal=condition_vector[:,2:4]
    upper=201/112-1
    x=start[:,:1]+fraction*(upper-start[:,:1])
    y=(28+168*.5*(1+torch.erf(z[:,:,1]/math.sqrt(2))))/112-1
    return torch.cat([start[:,None],torch.stack([x,y],-1),goal[:,None]],1)


def proposal(code,condition_vector):
    controls=control_points(code,condition_vector)
    t=torch.linspace(0,1,POINTS,device=code.device,dtype=code.dtype)
    knots=torch.linspace(0,1,7,device=code.device,dtype=code.dtype)
    weights=(1-(t[:,None]-knots[None]).abs()*6).clamp_min(0)
    return torch.einsum('pa,bad->bpd',weights,controls)


def make_route(hidden=512):
    model=RouteDecoder(RouteConfig(10,context_dim=25,coordinates=2,hidden=hidden))
    model.coordinate_scale.fill_(1.)
    return model


def path_context(own_context,position,path,index,done):
    n=len(path);index=torch.as_tensor(index,device=path.device,dtype=torch.long).reshape(-1).expand(n)
    ids=(index[:,None]+torch.arange(16,device=path.device)[None]).clamp_max(POINTS-1)
    local=path.gather(1,ids[:,:,None].expand(-1,-1,2))-position[:,None]
    global_ids=torch.linspace(0,POINTS-1,7,device=path.device).round().long()
    global_path=path[:,global_ids]-position[:,None]
    done=torch.as_tensor(done,device=path.device,dtype=path.dtype).reshape(-1,1).expand(n,1)
    return torch.cat([own_context,local.flatten(1)/56,global_path.flatten(1)/56,
                      index[:,None]/(POINTS-1),done],-1)


def initialize_controller(device='cuda'):
    from tworoom_addons.models import GatedGenerator,AdapterConfig
    config=AdapterConfig(rank=8,context_hidden=128,modulation_rank=16,
                         context_dim=60,output_modulation=True)
    return GatedGenerator('artifacts/final/base_policy.pt',config,device)


def load_controller(path,device='cuda'):
    from tworoom_addons.models import GatedGenerator,AdapterConfig
    from .legacy import digest_file
    saved=torch.load(path,map_location=device,weights_only=False)
    assert saved['base_sha256']==digest_file('artifacts/final/base_policy.pt')
    model=GatedGenerator('artifacts/final/base_policy.pt',AdapterConfig(**saved['adapter_config']),device)
    missing,unexpected=model.load_state_dict(saved['addon'],strict=False)
    assert not unexpected and all(x.startswith(('base.','original_encoder.')) for x in missing)
    return model.eval().requires_grad_(False),saved
