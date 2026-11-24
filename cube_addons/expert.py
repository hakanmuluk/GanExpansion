"""Training-only real-physics expert; no planned paths are policy inputs."""
import numpy as np
from cube.cube_gan.expert import Style
from .geometry import CARRY_Z,CUBE_BOUND_RADIUS,axis_frame,holding,decode_modifier


class SkillExpert:
    def __init__(self,initial,goal,base_style,spec,new_z,skill):
        self.initial=np.asarray(initial);self.goal=np.asarray(goal);self.style=base_style
        self.spec=np.asarray(spec);self.skill=skill;self.side,self.amount=decode_modifier(new_z,skill)
        self.ready=False;self.releasing=False

    def action(self,info,observer,visited_override=None):
        ee=np.asarray(info["proprio/effector_pos"]);cube=np.asarray(info["privileged/block_0_pos"])
        closing=float(info["proprio/gripper_opening"][0]);held=observer.is_holding(info)
        dist=np.linalg.norm(ee-cube);xy=np.linalg.norm(ee[:2]-cube[:2]);goal_xy=np.linalg.norm(cube[:2]-self.goal[:2])
        yaw=float(info["proprio/effector_yaw"][0]);cube_yaw=float(info["privileged/block_0_yaw"][0])
        yaw_error=((cube_yaw-yaw+np.pi/4)%(np.pi/2))-np.pi/4
        visited=observer.visited if visited_override is None else visited_override
        grip=-1.;allowed=self.skill=="keepout" or visited
        # Release is a committed phase. Re-evaluating a 14mm threshold after
        # opening caused repeated re-grasps when the cube settled by a few mm.
        self.releasing |= bool(allowed and goal_xy<.022 and cube[2]<.04)
        if self.releasing:
            target=np.r_[self.goal[:2],.20];phase=6 if closing>.25 else 7;yaw_error=0.
        elif held:
            grip=1.;yaw_error=0.;offset=ee-cube
            if allowed and goal_xy<.025:
                target=ee+(self.goal-cube);phase=5
            elif not self.ready and cube[2]<.17:
                target=ee+np.array([0.,0.,CARRY_Z-cube[2]]);phase=3
            else:
                self.ready=True;p,_,axis,perp,length=axis_frame(self.initial,self.goal,cube)
                ahead=min(1.,max(0.,p)+.14)
                if self.skill=="waypoint" and not visited:ahead=min(ahead,.5)
                side=-1. if self.style.mode==0 else 1.
                xy_target=self.initial[:2]+(self.goal[:2]-self.initial[:2])*ahead
                xy_target+=side*self.style.amplitude*np.sin(np.pi*ahead)*perp
                if self.skill=="keepout":
                    height=self.spec[3]+CUBE_BOUND_RADIUS+self.amount
                    z=CARRY_Z+self.side*height*np.sin(np.pi*ahead)
                else:
                    # The same waypoint is reached from above or below.
                    extra=float((self.spec[:2]-(self.initial[:2]+self.goal[:2])/2)@perp)-side*self.style.amplitude
                    xy_target+=extra*np.sin(np.pi*ahead)**2*perp
                    z=CARRY_Z+self.side*self.amount*np.sin(2*np.pi*ahead) if not visited else CARRY_Z
                target=np.r_[xy_target,z]+offset;phase=4
        elif xy>.014:
            self.ready=False;target=cube+np.array([0.,0.,.15]);phase=0
        elif dist>.012:
            self.ready=False;target=cube+np.array([0.,0.,.002]);phase=1
        else:
            self.ready=False;target=cube+np.array([0.,0.,.002]);grip=1.;phase=2
        a=np.zeros(5,np.float32);a[:3]=np.clip((target-ee)*15*self.style.speed,-self.style.speed,self.style.speed)
        a[3]=np.clip(yaw_error*3,-1,1);a[4]=grip
        return a,phase
