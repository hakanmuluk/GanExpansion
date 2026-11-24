"""Action-chunk and multiscale latent agreement of frozen neural skill models."""
import math
import torch
from .manifolds import FREE
from .action_model import path_context,world_path_tensor


def refine(bank,geometry,case,state,reference,cube,index,previous=None,starts=12,steps=40,seed=42,path_weight=0.,continuity_weight=0.):
    device=geometry.device;rng=torch.Generator(device=device).manual_seed(seed)
    common=.5*torch.randn((starts,FREE),device=device,generator=rng)
    angles=torch.linspace(-math.pi,math.pi,starts+1,device=device)[:-1,None]
    initial=torch.cat([common,angles,common.clone()],-1)
    if previous is not None:initial[:]=previous.expand(starts,-1);initial[1:,:FREE]+=.08*torch.randn((starts-1,FREE),device=device,generator=rng);initial[1:,FREE+1:]+=.08*torch.randn((starts-1,FREE),device=device,generator=rng)
    z=torch.nn.Parameter(initial);opt=torch.optim.Adam([z],lr=.07)
    g=geometry.expand(starts,-1);s=state.expand(starts,-1);bz=torch.as_tensor(case['base_z'],device=device)[None].expand(starts,-1)
    start=torch.as_tensor(case['initial_cube'],device=device,dtype=torch.float32)[None];goal=torch.as_tensor(case['goal'],device=device,dtype=torch.float32)[None]
    best=None;best_value=float('inf');best_metrics=None
    for iteration in range(steps+1):
        pk,pw=bank.models['keepout'](z[:,:FREE+1],g),bank.models['waypoint'](z[:,FREE+1:],g)
        paths=[world_path_tensor(p,start,goal) for p in [pk,pw]]
        contexts=[path_context(reference,cube,p,index) for p in paths]
        actions=bank.actor(torch.cat([s,s]),torch.cat([bz,bz]),torch.full((2*starts,1),.8,device=device),torch.cat(contexts))
        ak,aw=actions[:starts],actions[starts:]
        action_loss=(ak-aw).square().mean((1,2));route_loss=((pk-pw)/.1).square().mean((1,2))
        prior=z[:,:FREE].square().mean(-1)+z[:,FREE+1:].square().mean(-1)
        values=action_loss+path_weight*route_loss+1e-5*prior
        if previous is not None:values=values+continuity_weight*(z-previous).square().mean(-1)
        with torch.no_grad():
            value,choice=values.min(0)
            if float(value)<best_value:
                best_value=float(value);best=z[choice:choice+1].clone();best_metrics={'action_disagreement':float(action_loss[choice]),'path_disagreement':float(route_loss[choice]),'agreement_loss':best_value}
        if iteration==steps:break
        opt.zero_grad();values.sum().backward();opt.step()
        with torch.no_grad():z[:,:FREE].clamp_(-3,3);z[:,FREE+1:].clamp_(-3,3)
    with torch.no_grad():pk,pw=bank.models['keepout'](best[:,:FREE+1],geometry),bank.models['waypoint'](best[:,FREE+1:],geometry)
    return best,pk,pw,best_metrics
