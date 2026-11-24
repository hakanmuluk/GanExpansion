"""Native path following reference, used for diagnosis and single-skill data only."""
import numpy as np
from cube_addons.expert import SkillExpert
from cube.cube_gan.expert import Style
from cube.cube_gan.style import decode_style
import torch


class PathTeacher:
    def __init__(self,case,path,gain=40,tolerance=.010):
        mode,amp,speed=decode_style(torch.as_tensor(case['base_z']))
        self.style=Style(int(mode),float(amp),float(speed))
        self.teacher=SkillExpert(case['initial_cube'],case['goal'],self.style,case['waypoint'],[1.],'waypoint')
        self.path=np.asarray(path);self.index=0;self.gain=gain;self.tolerance=tolerance;self.done=False

    def plan(self,state,ctx,bz,runtime):
        info=runtime['info'];cube=np.asarray(info['privileged/block_0_pos'])
        if self.index==len(self.path)-1 and np.linalg.norm(cube-self.path[-1])<self.tolerance:self.done=True
        action,phase=self.teacher.action(info,runtime['observer'].w,visited_override=self.done)
        if phase==4:
            while self.index<len(self.path)-1 and np.linalg.norm(cube-self.path[self.index])<self.tolerance:self.index+=1
            limit=min(.65,self.style.speed)
            action[:3]=np.clip((self.path[self.index]-cube)*self.gain*self.style.speed,-limit,limit)
            action[3]=0.;action[4]=1.
        return np.repeat(action[None],16,axis=0),np.zeros(2,np.float32),{'seconds':0.,'iterations':0,'backward_steps':0,'calls':{},'disagreement':None,'teacher_phase':phase,'path_index':self.index}
