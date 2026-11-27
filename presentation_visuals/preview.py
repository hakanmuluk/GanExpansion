import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
from .common import ROOT,read,load_trial
from .design import CubeView,compose_frame
from independent_composition.observation_replay import prepare
from cube_addons.env import make_env,close_env

def main():
 c=next(c for c in read(ROOT/'PRIVATE_SELECTION.json')['clips'] if c['id']=='cube_composition_conflict_01');t=c['trials'][0];data,row=load_trial(t)
 out=ROOT/'preview';out.mkdir(exist_ok=True)
 for a in [90,135,175]:
  e=make_env(224)
  try:
   prepare(e,data);v=CubeView(e,data,['keepout','waypoint'],a)
   trail=[data['initial_cube']]
   for i,action in enumerate(data['actions']):
    _,_,_,_,info=e.step(action);trail.append(info['privileged/block_0_pos'].copy())
    if i==48:break
   frame=compose_frame(v.frame(trail),c,t,data,trail,{'goal':'approaching','keepout':'clear','waypoint':'pending'},['keepout','waypoint'])
   frame.save(out/f'cube_angle_{a}.png');v.close()
  finally:close_env(e)
 print({'previews':3,'render_only':True})
if __name__=='__main__':main()
