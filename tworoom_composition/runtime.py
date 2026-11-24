"""Exact native tensor formulas with device constants cached before graph capture."""
import torch
from tworoom_gan.env import TASK

_CONSTANTS={}


def constant(like,values):
    key=(str(like.device),like.dtype,repr(values))
    if key not in _CONSTANTS:_CONSTANTS[key]=like.new_tensor(values)
    return _CONSTANTS[key]


def observation(pos,goal):
    doors=constant(pos,[112.,56.,112.,168.,0.,0.])
    return torch.cat([pos,goal,doors.expand(*pos.shape[:-1],6)],-1)/112.-1.


def transition(pos,action,task=TASK):
    proposed=pos+action.clamp(-1,1)*task.speed
    bounded=proposed.clamp(task.border+task.radius,task.image_size-task.border-task.radius)
    in_door=((bounded[...,1,None]-constant(pos,task.door_y)).abs()<=task.door_half_width+1.75).any(-1)
    left=pos[...,0]<task.wall_x
    left_edge=task.wall_x-task.wall_thickness//2-task.radius;right_edge=task.wall_x+task.wall_thickness//2+task.radius
    blocked=(~in_door)&torch.where(left,bounded[...,0]>left_edge,bounded[...,0]<right_edge)
    clamp_x=torch.where(left,constant(pos,left_edge-.5),constant(pos,right_edge+.5))
    return torch.stack([torch.where(blocked,clamp_x,bounded[...,0]),bounded[...,1]],-1),blocked
