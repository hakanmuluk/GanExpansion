"""Fresh frozen-base native demonstrations; no training or latent search."""
import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,STUDY,read,write,load_trial,entry,job,sha

def tworoom(bank,case,code,out):
 from tworoom_gan.env import make_reference,reset_reference,observation
 env=make_reference();pos=case['start'].copy();positions=[pos.copy()];actions=[]
 try:
  reset_reference(env,pos,case['goal']);bz=torch.as_tensor(code,dtype=torch.float32,device='cuda')[None]
  for t in range(160):
   if t%8==0:
    with torch.no_grad():chunk=bank.base(observation(torch.as_tensor(pos,device='cuda')[None],torch.as_tensor(case['goal'],device='cuda')[None]),bz)[0].cpu().numpy()
   a=chunk[t%8].copy();ob,_,done,_,_=env.step(a);pos=np.asarray(ob[:2],np.float32).copy();positions.append(pos);actions.append(a)
   if np.linalg.norm(pos-case['goal'])<16:break
  row={'success':bool(np.linalg.norm(pos-case['goal'])<16),'goal_success':bool(np.linalg.norm(pos-case['goal'])<16),'steps':len(actions)}
  case={**case,'base_z':np.asarray(code,np.float32)}
  np.savez_compressed(out/'trajectories.npz',positions=np.asarray(positions)[None],actions=np.asarray(actions)[None],lengths=np.array([len(actions)]),**{k:np.asarray(v)[None] for k,v in case.items()})
 finally:env.close()
 return row

def cube(bank,case,code,out):
 from cube_addons.env import make_env,close_env,features,completion,step_traced
 from cube_addons.contacts import contact_info
 from independent_composition.observation_replay import prepare
 env=make_env();states=[];actions=[];poses=[];micro=[];previous=None;stable=0
 try:
  ob,info=prepare(env,case);contact_info(env,info);bz=torch.as_tensor(code,dtype=torch.float32,device='cuda')[None]
  for t in range(200):
   state=features(env,ob,info,case['initial_cube'],previous);states.append(state);poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']])
   if t%8==0:
    with torch.no_grad():chunk=bank.base(torch.as_tensor(state,device='cuda')[None],bz)[0].cpu().numpy()
   a=chunk[t%8].copy();a[4]=1. if a[4]>=0 else -1.;previous=info['proprio/effector_pos'].copy()
   result,path=step_traced(env,a);ob,_,_,_,info=result;contact_info(env,info);native,released=completion(env,info);stable=stable+1 if released else 0;actions.append(a);micro.append(path.astype(np.float32))
   if stable>=10:break
  poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']]);row={'success':bool(stable>=10),'goal_success':bool(stable>=10),'released_success':bool(stable>=10),'steps':len(actions)}
  np.savez_compressed(out/'rollout_0000.npz',states=np.asarray(states,np.float32),poses=np.asarray(poses,np.float32),actions=np.asarray(actions,np.float32),micro_cube=np.asarray(micro,np.float32),**{**case,'base_z':np.asarray(code,np.float32)})
 finally:close_env(env)
 return row

def main():
 from independent_composition.regularization_models import load_bank
 torch.set_num_threads(2);s=read(STUDY);selection=read(ROOT/'PRIVATE_SELECTION.json');assert not any(c['family']=='base' for c in selection['clips'])
 clips=[];attempts={}
 for env in ['tworoom','cube']:
  bank=load_bank(env,**s['environments'][env]['models'],device='cuda');accepted=0;tested=0
  j=job(s,env,'supported_easy',20,241)
  codes=[[-1.,-.8],[1.,.2],[-1.,.8]]
  for i in range(12):
   code=codes[accepted];data,_=load_trial(entry(j,i,'base'))
   keys=['start','goal','keepout','waypoint','base_z'] if env=='tworoom' else ['initial_state','seed','kind','initial_cube','goal','keepout','waypoint','base_z']
   data={k:data[k] for k in keys};out=ROOT/'base_native_v2'/env/f'attempt_{i:02d}';out.mkdir(parents=True);tested+=1
   row=(tworoom if env=='tworoom' else cube)(bank,data,code,out);write(out/'records.json',[row])
   if not row['success']:continue
   accepted+=1;clips.append({'id':f'{env}_base_{accepted:02d}','environment':env,'family':'base','trials':[{'environment':env,'category':'base','run':str(out),'index':0,'role':'base','method':'frozen_base','steps':0,'seed':42}]})
   if accepted==3:break
  assert accepted==3,(env,accepted);bank.assert_frozen();attempts[env]={'tested_native_base_episodes':tested,'selected_successful_clips':accepted}
  assert all(sha(d['path'])==d['sha256'] for d in selection['models'][env].values())
 selection['clips']=clips+selection['clips'];assert len(selection['clips'])==35;write(ROOT/'PRIVATE_SELECTION.json',selection)
 write(ROOT/'PUBLIC_PROGRESS.json',{'stage':'all_tasks_selected','task_clips':35,'overview_clips_pending':1,'final_target':36,'base_curation':attempts,'models_frozen':True,'raw_episode_data_exported':False})
 print(read(ROOT/'PUBLIC_PROGRESS.json'))
if __name__=='__main__':main()
