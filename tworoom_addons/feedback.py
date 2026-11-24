"""Feedback targets from FINITE stored paths, with no extra expert queries.

On-policy states are matched to the demonstrated curve for that exact latent
and context. Only target construction uses the curve; inference uses the actor.
"""
import torch
from tworoom_gan.env import transition
from .rollout import rollout
from .geometry import segment_distance


@torch.no_grad()
def finite_path_feedback(model,b,skill,count=12):
    count=min(count,len(b["lengths"]))
    r=rollout(model,b["start"][:count],b["goal"][:count],b["base_z"][:count],b["new_z"][:count],b["spec"][:count],skill)
    state=r["states"][:,::8];ctx=r["contexts"][:,::8]
    p=r["positions"][:,:-1:8]
    reference=(b["episode_states"][:count,:,:2]+1)*112
    dist=(p[:,:,None]-reference[:,None]).square().sum(-1)
    valid=torch.arange(240,device=p.device)[None]<b["lengths"][:count,None]
    compatible=valid[:,None].expand(-1,p.shape[1],-1)
    if skill=="waypoint":compatible=compatible&((b["episode_contexts"][:count,None,:,8]>.5)==(ctx[:,:,None,8]>.5))
    closest=dist.masked_fill(~compatible,1e9).argmin(-1)
    times=closest[...,None]+torch.arange(1,17,device=p.device)
    times=torch.minimum(times,b["lengths"][:count,None,None].clamp_max(239))
    rows=torch.arange(count,device=p.device)[:,None,None]
    # Equal sampling rate and recorded speed; correction is a path tracker.
    speed=b["episode_actions"][:count].norm(dim=-1).amax(1)[:,None,None]
    current=p;actions=[]
    tracker_visited=ctx[...,8]>.5
    first_visit=(b["episode_contexts"][:count,:,8]>.5).long().argmax(1)
    for j in range(16):
        index=times[...,j]
        if skill=="waypoint":
            # A stored POST-visit segment is inappropriate until THIS rollout
            # has actually touched the waypoint. Do not assume expert progress.
            index=torch.where(tracker_visited,index,torch.minimum(index,first_visit[:,None]))
        point=reference[torch.arange(count,device=p.device)[:,None],index]
        delta=(point-current)/5
        a=delta*(speed/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)).clamp_max(1)
        nxt,_=transition(current,a)
        if skill=="waypoint":
            tracker_visited|=segment_distance(current,nxt,b["spec"][:count,None,:2])<=b["spec"][:count,None,2]
        current=nxt;actions.append(a)
    target=torch.stack(actions,-2)
    alive=torch.arange(state.shape[1],device=p.device)[None]*8<r["lengths"][:,None]
    z=b["base_z"][:count,None].expand(-1,state.shape[1],-1)
    code=b["new_z"][:count,None].expand(-1,state.shape[1],-1)
    return state[alive],ctx[alive],z[alive],code[alive],target[alive]
