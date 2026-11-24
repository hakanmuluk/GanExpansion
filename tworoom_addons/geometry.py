"""Observed skill state and swept geometry; native dynamics are unchanged."""
import math
import torch
from tworoom_gan.env import TASK,transition,observation
from tworoom_gan.style import decode_style

MAX_STEPS=240
CONTEXT_DIM=12
SKILLS=("keepout","waypoint")


def frame(points,goal):
    sign=torch.where(goal[...,0]>=112,1.,-1.)
    return torch.stack([112+sign*(points[...,0]-112),points[...,1]],-1)


def unframe(points,goal):
    sign=torch.where(goal[...,0]>=112,1.,-1.)
    while sign.ndim<points.ndim-1:sign=sign[...,None]
    return torch.stack([112+sign*(points[...,0]-112),points[...,1]],-1)


def segment_distance(a,b,center):
    delta=b-a
    u=((center-a)*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-8)
    nearest=a+u.clamp(0,1)[...,None]*delta
    return (nearest-center).norm(dim=-1)


def skill_features(pos,goal,start,spec,visited,time,skill,previous_action=None):
    p,c,s=frame(pos,goal),frame(spec[...,:2],goal),frame(start,goal)
    delta=p-c;distance=delta.norm(dim=-1)
    radius=spec[...,2]+(TASK.radius if skill=="keepout" else 0)
    if not torch.is_tensor(time):time=distance.new_full(distance.shape,float(time))
    unit=delta/distance.clamp_min(1e-6)[...,None]
    features=torch.cat([c/112-1,spec[...,2:3]/40,s/112-1,delta/56,
                      ((distance-radius)/56)[...,None],visited.to(pos.dtype)[...,None],
                      (time/MAX_STEPS)[...,None],unit],-1)
    if previous_action is not None:
        direction=torch.where(goal[...,0]>=112,1.,-1.)
        past=previous_action*torch.stack([direction,torch.ones_like(direction)],-1)
        features=torch.cat([features,past],-1)
    return features


def update_constraint(before,after,spec,visited,violated,skill):
    distance=segment_distance(before,after,spec[...,:2])
    if skill=="keepout":violated=violated|(distance<spec[...,2]+TASK.radius-1e-5)
    else:visited=visited|(distance<=spec[...,2])
    return visited,violated


def extra_gaussian_styles(code,skill):
    u=torch.erf(code[...,0].abs()/math.sqrt(2))
    side=torch.where(code[...,0]>=0,1.,-1.)
    amount=(7+5*u) if skill=="keepout" else (18+8*u)
    return side,amount


def demo_paths(start,goal,base_z,spec,code,skill):
    """Training-only geometric expert. No planned nodes go to the policy."""
    s,g,c=frame(start,goal),frame(goal,goal),frame(spec[...,:2],goal)
    mode,speed,offset=decode_style(base_z)
    y=start.new_tensor(TASK.door_y)[mode]+offset
    pre=torch.stack([torch.full_like(y,99.),y],-1)
    exit=torch.stack([torch.full_like(y,142.),y],-1)
    side,amount=extra_gaussian_styles(code,skill)
    if skill=="waypoint":
        via=c+torch.stack([side*amount,torch.zeros_like(amount)],-1)
        nodes=torch.stack([pre,exit,via,c,g],1)
    else:
        radius=spec[...,2]+TASK.radius+amount
        rel=s-c;distance=rel.norm(dim=-1)
        egress=c+rel/distance.clamp_min(1e-6)[:,None]*torch.maximum(distance,radius+.75)[:,None]
        a,b=egress-c,pre-c
        alpha,beta=torch.atan2(a[:,1],a[:,0]),torch.atan2(b[:,1],b[:,0])
        # Positive code selects the positive side of the start-to-door axis.
        direction=-side
        begin=alpha+direction*torch.acos((radius/a.norm(dim=-1)).clamp(-.999999,.999999))
        end=beta-direction*torch.acos((radius/b.norm(dim=-1)).clamp(-.999999,.999999))
        arc=torch.remainder(direction*(end-begin),2*math.pi)
        theta=begin[:,None]+direction[:,None]*arc[:,None]*torch.linspace(0,1,13,device=s.device)[None]
        ring=c[:,None]+radius[:,None,None]*torch.stack([theta.cos(),theta.sin()],-1)
        nodes=torch.cat([egress[:,None],ring,pre[:,None],exit[:,None],g[:,None]],1)
    return unframe(nodes,goal),speed


def constraint_loss(positions,goal,spec,skill,lengths):
    valid=torch.arange(positions.shape[1]-1,device=positions.device)[None]<lengths[:,None]
    if skill=="keepout":
        dist=segment_distance(positions[:,:-1],positions[:,1:],spec[:,None,:2])
        penalty=torch.relu(spec[:,None,2]+TASK.radius+3-dist).square()/100
        return (penalty*valid).sum()/valid.sum().clamp_min(1)
    distance=segment_distance(positions[:,:-1],positions[:,1:],spec[:,None,:2])
    # The real swept minimum cannot be reduced merely by waiting near a disk.
    minimum=distance.masked_fill(~valid,float("inf")).amin(1)
    return torch.relu(minimum-spec[:,2]+1).square().mean()/100
