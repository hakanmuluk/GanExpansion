"""Batch independent action-agreement queries without mixing their objectives."""
import math
import numpy as np
import torch
from .manifolds import FREE
from .action_model import path_context,world_path_tensor


def refine_batch(bank,geometry,cases,state,reference,cube,index,seeds,previous=None,starts=12,steps=20,continuity_weight=.001):
    device=geometry.device;b=len(cases);initial=[]
    for i,seed in enumerate(seeds):
        rng=torch.Generator(device=device).manual_seed(seed);common=.5*torch.randn((starts,FREE),device=device,generator=rng)
        angles=torch.linspace(-math.pi,math.pi,starts+1,device=device)[:-1,None]
        z=torch.cat([common,angles,common.clone()],-1)
        if previous is not None:
            z[:]=previous[i].expand(starts,-1)
            z[1:,:FREE]+=.08*torch.randn((starts-1,FREE),device=device,generator=rng)
            z[1:,FREE+1:]+=.08*torch.randn((starts-1,FREE),device=device,generator=rng)
        initial.append(z)
    z=torch.nn.Parameter(torch.cat(initial));opt=torch.optim.Adam([z],lr=.07)
    repeat=lambda t:t.repeat_interleave(starts,0)
    g=repeat(geometry);s=repeat(state);ref=repeat(reference);xyz=repeat(cube);ids=repeat(index)
    bz=repeat(torch.as_tensor(np.stack([c['base_z'] for c in cases]),device=device))
    start=repeat(torch.as_tensor(np.stack([c['initial_cube'] for c in cases]),device=device,dtype=torch.float32));goal=repeat(torch.as_tensor(np.stack([c['goal'] for c in cases]),device=device,dtype=torch.float32))
    best=None;best_value=None;best_action=None;best_path=None;offset=torch.arange(b,device=device)*starts
    for iteration in range(steps+1):
        pk,pw=bank.models['keepout'](z[:,:FREE+1],g),bank.models['waypoint'](z[:,FREE+1:],g)
        contexts=[path_context(ref,xyz,world_path_tensor(p,start,goal),ids) for p in [pk,pw]]
        actions=bank.actor(torch.cat([s,s]),torch.cat([bz,bz]),torch.full((2*b*starts,1),.8,device=device),torch.cat(contexts))
        ak,aw=actions[:b*starts],actions[b*starts:];action_loss=(ak-aw).square().mean((1,2))
        values=action_loss+1e-5*(z[:,:FREE].square().mean(-1)+z[:,FREE+1:].square().mean(-1))
        if previous is not None:values=values+continuity_weight*(z-repeat(previous)).square().mean(-1)
        with torch.no_grad():
            value,choice=values.reshape(b,starts).min(1);selected=offset+choice;codes=z[selected].clone()
            route=((pk[selected]-pw[selected])/.1).square().mean((1,2));action=action_loss[selected]
            if best is None:best=codes;best_value=value.clone();best_action=action.clone();best_path=route.clone()
            else:
                update=value<best_value;best=torch.where(update[:,None],codes,best);best_value=torch.minimum(value,best_value);best_action=torch.where(update,action,best_action);best_path=torch.where(update,route,best_path)
        if iteration==steps:break
        opt.zero_grad();values.sum().backward();opt.step()
        with torch.no_grad():z[:,:FREE].clamp_(-3,3);z[:,FREE+1:].clamp_(-3,3)
    with torch.no_grad():pk,pw=bank.models['keepout'](best[:,:FREE+1],geometry),bank.models['waypoint'](best[:,FREE+1:],geometry)
    metrics=[{'action_disagreement':float(best_action[i]),'path_disagreement':float(best_path[i]),'agreement_loss':float(best_value[i])} for i in range(b)]
    return best,pk,pw,metrics
