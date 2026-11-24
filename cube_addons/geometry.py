"""Observed task state and conservative swept transported-object constraints."""
import math
import numpy as np

MAX_STEPS=240
CONTEXT_DIM=20
CUBE_BOUND_RADIUS=math.sqrt(3)*.02
CARRY_Z=.20
SKILLS=("keepout","waypoint")


def segment_distance(a,b,c):
    a,b,c=map(np.asarray,(a,b,c));v=b-a
    u=np.sum((c-a)*v,axis=-1)/np.maximum(np.sum(v*v,axis=-1),1e-12)
    return np.linalg.norm(a+np.clip(u,0,1)[...,None]*v-c,axis=-1)


def axis_frame(initial,goal,cube):
    delta=np.asarray(goal)[:2]-np.asarray(initial)[:2];length=max(np.linalg.norm(delta),1e-6)
    axis=delta/length;perp=np.array([-axis[1],axis[0]])
    rel=np.asarray(cube)[:2]-np.asarray(initial)[:2]
    return float(rel@axis/length),float(rel@perp),axis,perp,length


def holding(info):
    ee=np.asarray(info["proprio/effector_pos"]);cube=np.asarray(info["privileged/block_0_pos"])
    return bool(info["proprio/gripper_opening"][0]>.5 and np.linalg.norm(ee-cube)<.065 and
                (info["proprio/gripper_contact"][0]>.2 or cube[2]>.045))


class Observer:
    def __init__(self,initial,goal,spec,skill,holding_fn=None):
        self.initial=np.asarray(initial);self.goal=np.asarray(goal);self.spec=np.asarray(spec);self.skill=skill
        self.is_holding=holding_fn or holding
        self.has_lifted=False;self.visited=False;self.violated=False;self.table_violation=False
        self.minimum_clearance=float("inf");self.held_transport_samples=0
        self.previous_holding=False;self.transport_path_length=0.;self.supported_path_length=0.

    def update(self,before,after,info,path=None):
        carried=self.is_holding(info);previously_held=self.previous_holding;self.previous_holding=carried
        self.has_lifted|=bool(carried and after[2]>.10)
        p,_,_,_,_=axis_frame(self.initial,self.goal,after)
        if path is None:path=np.stack([before,after])
        delta=self.goal[:2]-self.initial[:2]
        progress=((path[:,:2]-self.initial[:2])@delta)/max(float(delta@delta),1e-12)
        if self.has_lifted:
            self.table_violation|=bool(np.any((progress>.2)&(progress<.8)&(path[:,2]<.07)))
            middle=(progress[:-1]+progress[1:])/2
            length=float(np.linalg.norm(np.diff(path,axis=0),axis=-1)[(middle>.2)&(middle<.8)].sum())
            self.transport_path_length+=length
            if previously_held and carried:self.supported_path_length+=length
        if self.has_lifted and .2<p<.8:
            self.held_transport_samples+=int(carried and after[2]>.08)
        distance=float(segment_distance(path[:-1],path[1:],self.spec[:3]).min())
        if self.skill=="keepout":
            gap=float(distance-self.spec[3]-CUBE_BOUND_RADIUS)
            self.minimum_clearance=min(self.minimum_clearance,gap)
            self.violated|=gap < -1e-7
        else:self.visited|=bool(previously_held and carried and path[:,2].min()>.08 and distance<=self.spec[3])

    def supported_fraction(self):
        return self.supported_path_length/max(self.transport_path_length,1e-9)

    def satisfied(self):
        return bool(self.has_lifted and self.held_transport_samples>=3 and self.supported_fraction()>=.8 and not self.table_violation and
                    (not self.violated if self.skill=="keepout" else self.visited))

    def features(self,info,time):
        ee=np.asarray(info["proprio/effector_pos"]);cube=np.asarray(info["privileged/block_0_pos"])
        p,lateral,axis,_,length=axis_frame(self.initial,self.goal,cube)
        x=np.r_[(self.spec[:3]-np.array([.425,0,.2]))/np.array([.2,.3,.2]),self.spec[3]/.1,
                (cube-self.spec[:3])/.15,(ee-self.spec[:3])/.15,p,lateral/.15,(cube[2]-CARRY_Z)/.1,
                float(self.is_holding(info)),float(self.has_lifted),float(self.visited),time/MAX_STEPS,axis,length/.4].astype(np.float32)
        assert x.shape==(CONTEXT_DIM,),x.shape
        return x


def decode_modifier(z,skill):
    value=float(np.asarray(z).reshape(-1)[0]);u=math.erf(abs(value)/math.sqrt(2))
    return (-1. if value<0 else 1.),(.020+.015*u if skill=="keepout" else .040+.035*u)
