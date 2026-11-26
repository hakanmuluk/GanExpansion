"""Training-data-only Cube experts with broader independent skill families.

Original grasp/lift/lower/release control is reused. Only training carry paths
are diversified. These paths are never network inputs or inference planners.
Each teacher sees its own constraint only; no joint-task filtering is used.
"""
import math
import numpy as np
from cube_addons.expert import SkillExpert
from cube_addons.geometry import CARRY_Z,CUBE_BOUND_RADIUS,axis_frame


def angle_from_code(code):
    value=float(np.asarray(code).reshape(-1)[0])
    return math.pi*math.erf(value/math.sqrt(2))


def lookahead(points,position,distance=.032):
    points=np.asarray(points);a=points[:-1];delta=points[1:]-a
    lengths=np.linalg.norm(delta,axis=1);squared=np.maximum(lengths**2,1e-12)
    fraction=np.clip(np.sum((position-a)*delta,axis=1)/squared,0,1)
    projected=a+fraction[:,None]*delta
    segment=int(np.argmin(np.sum((projected-position)**2,axis=1)))
    accumulated=np.r_[0.,np.cumsum(lengths)]
    target=min(accumulated[-1],accumulated[segment]+fraction[segment]*lengths[segment]+distance)
    index=min(int(np.searchsorted(accumulated,target,side='right')-1),len(lengths)-1)
    portion=(target-accumulated[index])/max(lengths[index],1e-9)
    return a[index]+portion*delta[index]


class DiverseSkillExpert(SkillExpert):
    def __init__(self,initial,goal,base_style,spec,new_z,skill):
        super().__init__(initial,goal,base_style,spec,new_z,skill)
        self.angle=angle_from_code(new_z)
        _,_,axis,perp,_=axis_frame(initial,goal,initial)
        along=np.r_[axis,0.]
        # A common physical convention across independently collected skills:
        # near-zero codes point outward along the selected base-route side;
        # positive/negative codes still denote above/below alternatives.
        side=np.r_[perp,0.] * (-1. if base_style.mode==0 else 1.)
        direction=math.cos(self.angle)*side+np.array([0.,0.,math.sin(self.angle)])
        start=np.r_[self.initial[:2],CARRY_Z];end=np.r_[self.goal[:2],CARRY_Z]
        center=np.asarray(spec[:3])
        if skill=='keepout':
            # Include low-clearance horizontal detours as well as the original
            # larger vertical ones. This uses own-skill geometry only.
            radius=spec[3]+CUBE_BOUND_RADIUS+.008+.012*abs(math.sin(self.angle))
            extent=1.25*radius;radial=radius*direction
            self.points=np.stack([start,center-extent*along+radial,
                                  center+radial,center+extent*along+radial,end])
        else:
            # All targets remain inside the same waypoint ball. Different
            # codes expose its tolerance instead of always aiming at center.
            touch=center+.75*spec[3]*direction
            before=touch-.060*along+.030*direction
            after=touch+.060*along-.010*direction
            self.points=np.stack([start,before,touch,after,end])

    def action(self,info,observer,visited_override=None):
        action,phase=super().action(info,observer,visited_override)
        if phase!=4:return action,phase
        visited=observer.visited if visited_override is None else visited_override
        points=self.points
        if self.skill=='waypoint':points=points[2:] if visited else points[:3]
        cube=np.asarray(info['privileged/block_0_pos'])
        target=lookahead(points,cube,distance=.022 if self.skill=='keepout' else .032)
        action[:3]=np.clip((target-cube)*15*self.style.speed,-self.style.speed,self.style.speed)
        return action,phase


def independent_spec(case,style,skill,seed):
    rng=np.random.default_rng(seed+(171 if skill=='keepout' else 293))
    initial,goal=case['initial_cube'],case['goal']
    _,_,axis,perp,_=axis_frame(initial,goal,initial)
    p=rng.uniform(.42,.58);sign=-1. if style.mode==0 else 1.
    xy=initial[:2]+(goal[:2]-initial[:2])*p+sign*style.amplitude*np.sin(np.pi*p)*perp
    if skill=='keepout':
        xy+=rng.uniform(-.012,.012)*perp;height=rng.uniform(.185,.215);radius=rng.uniform(.014,.030)
    else:
        xy+=float(rng.choice([-1.,1.]))*rng.uniform(.035,.075)*perp
        height=rng.uniform(.18,.22);radius=rng.uniform(.024,.038)
    return np.r_[xy,height,radius]
