"""Strict swept geometry, separate from the native permissive collision rule."""
import torch
from tworoom_gan.env import TASK,transition,observation
from tworoom_addons.geometry import skill_features,segment_distance
from .runtime import constant

WALLS=((107.,117.,14.,35.),(107.,117.,77.,147.),(107.,117.,189.,210.))


def entry_fraction(a,b,center,radius):
    d=b-a;r=a-center;aa=d.square().sum(-1);bb=2*(r*d).sum(-1)
    cc=r.square().sum(-1)-radius**2;discriminant=bb.square()-4*aa*cc
    root=(-bb-discriminant.clamp_min(0).sqrt())/(2*aa.clamp_min(1e-12))
    end=(-bb+discriminant.clamp_min(0).sqrt())/(2*aa.clamp_min(1e-12))
    hit=(aa>1e-12)&(discriminant>=0)&(end>=0)&(root<=1)
    return torch.where(cc<=0,torch.zeros_like(root),torch.where(hit,root.clamp(0,1),torch.full_like(root,float('inf'))))


def segment_box_distance(a,b,box):
    lo=constant(a,[box[0],box[2]]);hi=constant(a,[box[1],box[3]])
    endpoint=torch.minimum((a-a.clamp(lo,hi)).norm(dim=-1),(b-b.clamp(lo,hi)).norm(dim=-1))
    corners=constant(a,[[box[0],box[2]],[box[0],box[3]],[box[1],box[2]],[box[1],box[3]]])
    corner=segment_distance(a[...,None,:],b[...,None,:],corners).amin(-1)
    delta=b-a;parallel=delta.abs()<1e-12
    safe=torch.where(parallel,torch.ones_like(delta),delta)
    t0=(lo-a)/safe;t1=(hi-a)/safe
    lower=torch.minimum(t0,t1);upper=torch.maximum(t0,t1)
    lower=torch.where(parallel,torch.full_like(lower,-float('inf')),lower)
    upper=torch.where(parallel,torch.full_like(upper,float('inf')),upper)
    inside_parallel=(~parallel|((a>=lo)&(a<=hi))).all(-1)
    intersects=inside_parallel&(lower.amax(-1).clamp_min(0)<=upper.amin(-1).clamp_max(1))
    return torch.where(intersects,torch.zeros_like(endpoint),torch.minimum(endpoint,corner))


def wall_clearance(a,b):
    d=torch.stack([segment_box_distance(a,b,box) for box in WALLS],-1).amin(-1)-TASK.radius
    border=torch.minimum(torch.minimum(a,b).amin(-1)-21,203-torch.maximum(a,b).amax(-1))
    return torch.minimum(d,border)


def contexts(pos,case,visited,time,previous=None,dimensions=(12,12)):
    return tuple(skill_features(pos,case['goal'],case['start'],case[name],visited if skill=='waypoint' else torch.zeros_like(visited),
                    time,skill,previous if dim==14 else None)
                 for name,skill,dim in zip(['keepout','waypoint'],['keepout','waypoint'],dimensions))


def update(pos,nxt,case,visited,violated,wall_hit,early_goal):
    wp=entry_fraction(pos,nxt,case['waypoint'][...,:2],case['waypoint'][...,2])
    goal=entry_fraction(pos,nxt,case['goal'],TASK.success_radius)
    early_goal=early_goal|(~visited&(goal<wp-1e-6))
    visited=visited|torch.isfinite(wp)
    violated=violated|(segment_distance(pos,nxt,case['keepout'][...,:2])<case['keepout'][...,2]+TASK.radius-1e-5)
    wall_hit=wall_hit|(wall_clearance(pos,nxt)<-1e-4)
    reached=(nxt-case['goal']).norm(dim=-1)<TASK.success_radius
    return visited,violated,wall_hit,early_goal,reached
