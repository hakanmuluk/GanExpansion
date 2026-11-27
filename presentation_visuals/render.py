"""Replay verified learned actions natively; stream annotated MP4s to disk."""
import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import argparse,time,math
import numpy as np
from PIL import Image,ImageDraw
from .common import ROOT,STUDY,read,write,load_trial,sha
from .design import CubeView,TwoRoomView,VideoWriter,compose_frame,font,wrapped,W,H,FPS,BG,TEXT,MUTED,RED,GREEN,GOLD,CARD

class Trial:
 def __init__(self,trial,active):
  self.trial=trial;self.data,self.row=load_trial(trial);self.environment=trial['environment'];self.active=active;self.t=0;self.maxerr=0.;self.render_state_error=0.;self.visited=False;self.keep=False;self.wall=False;self.early=False;self.stable=0
  if self.environment=='tworoom':
   from tworoom_gan.env import make_reference,reset_reference
   self.env=make_reference();reset_reference(self.env,self.data['start'],self.data['goal']);self.trail=[self.data['start'].copy()];self.view=TwoRoomView(self.env,self.data,active)
  else:
   from cube_addons.env import make_env
   from cube_addons.contacts import contact_info,force_holding
   from independent_composition.observation_replay import prepare
   from cube_composition.geometry import JointObserver
   self.env=make_env(224);_,self.info=prepare(self.env,self.data);contact_info(self.env,self.info)
   self.observer=JointObserver(self.data);held=force_holding(self.info);self.observer.previous_held=held;self.observer.k.previous_holding=self.observer.w.previous_holding=held
   self.trail=[self.info['privileged/block_0_pos'].copy()];self.view=CubeView(self.env,self.data,active)
   self.maxerr=float(np.max(np.abs(np.r_[self.info['proprio/effector_pos'],self.info['privileged/block_0_pos']]-self.data['poses'][0])))
 def status(self,end=False):
  if self.environment=='tworoom':
   reached=bool(np.linalg.norm(self.trail[-1]-self.data['goal'])<16)
   k='violated' if self.keep else 'clear';w='missed' if not self.visited and (end or self.early) else 'too early' if self.early else 'visited' if self.visited else 'pending';goal='reached' if reached else 'not reached' if end else 'approaching'
  else:
   k='violated' if self.observer.k.violated else 'clear';w='visited' if self.observer.w.visited else 'missed' if end else 'pending';goal='settled' if self.stable>=10 else 'not reached' if end else 'approaching'
  role=self.trial['role'];success=self.row['success'] if role in ['base','composition'] else self.row[role+'_full_success']
  return {'goal':goal,'keepout':k,'waypoint':w,'task_success':bool(success)}
 def step(self):
  a=self.data['actions'][self.t]
  if self.environment=='tworoom':
   from tworoom_composition.audit import entry,box_distance
   from tworoom_composition.planner import segment_distance
   old=self.trail[-1];ob,_,_,_,_=self.env.step(a);p=np.asarray(ob[:2]).copy();self.trail.append(p)
   wp=entry(old,p,self.data['waypoint'][:2],self.data['waypoint'][2]);goal=entry(old,p,self.data['goal'],16)
   self.early|=bool(not self.visited and goal<wp-1e-6);self.visited|=bool(np.isfinite(wp));self.keep|=bool(segment_distance(old,p,self.data['keepout'][:2])<self.data['keepout'][2]+7-1e-5)
   self.wall|=any(box_distance(old,p,b)<7-1e-4 for b in [(107,117,14,35),(107,117,77,147),(107,117,189,210)])
   self.wall|=min(*old,*p)<21-1e-4 or max(*old,*p)>203+1e-4
   self.maxerr=max(self.maxerr,float(np.max(np.abs(p-self.data['positions'][self.t+1]))))
  else:
   from cube_addons.env import step_traced,completion
   from cube_addons.contacts import contact_info
   before=self.info['privileged/block_0_pos'].copy();result,path=step_traced(self.env,a);_,_,_,_,self.info=result;contact_info(self.env,self.info);p=self.info['privileged/block_0_pos'].copy();self.trail.append(p)
   self.observer.update(before,p,self.info,path,self.env);native,released=completion(self.env,self.info);self.stable=self.stable+1 if released else 0
   self.maxerr=max(self.maxerr,float(np.max(np.abs(np.r_[self.info['proprio/effector_pos'],p]-self.data['poses'][self.t+1]))))
  self.t+=1
 def frame(self):return self.view.frame(self.trail)
 def verify(self):
  if self.environment=='tworoom':
   from tworoom_composition.audit import monitor
   m=monitor(np.asarray(self.trail),self.data);assert self.maxerr<5e-5
   assert m['visited']==self.visited and m['keepout_violation']==self.keep and m['strict_wall_violation']==self.wall and m['early_goal_entry']==self.early
   actual={'goal_success':m['goal_success'],'keepout_satisfied':not self.keep,'waypoint_satisfied':bool(self.visited and not self.early),'success':m['success']}
  else:
   from cube_addons.env import completion
   native,_=completion(self.env,self.info);m=self.observer.result(self.stable>=10,native);assert self.maxerr<2e-6,self.maxerr
   actual={'goal_success':bool(self.stable>=10),'keepout_satisfied':m['keepout_satisfied'],'waypoint_satisfied':m['waypoint_satisfied'],'success':m['success']}
  if self.trial['role']=='base':assert actual['goal_success']==self.row['goal_success'] and self.row['success']
  else:
   for k,v in actual.items():assert v==self.row[k],(k,v,self.row[k])
  return {'environment':self.environment,'role':self.trial['role'],'maximum_native_replay_error':self.maxerr,'render_only_state_checks_passed':True,'all_displayed_labels_match_original_monitor':True,'goal_verified':actual['goal_success'],'own_task_verified':self.status(True)['task_success'],'joint_verified':actual['success'] if self.trial['role']!='base' else None}
 def close(self):
  self.view.close()
  if self.environment=='cube':
   from cube_addons.env import close_env
   close_env(self.env)
  else:self.env.close()

def summary_frame(clip,panels):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);env='Cube' if clip['environment']=='cube' else 'TwoRoom'
 d.text((50,32),env+' | same scene, combined skills',font=font(48,True),fill=TEXT)
 d.text((52,100),'Same reset, goal and regions. Different learned-policy inputs.',font=font(27),fill=MUTED)
 for i,(native,trial,status) in enumerate(panels):
  x=45+i*630;label=['Keepout adapter','Waypoint adapter','Latent composition'][i];d.text((x+12,174),label,font=font(31,True),fill=TEXT)
  im.paste(native.resize((580,580),Image.Resampling.LANCZOS),(x,232));d=ImageDraw.Draw(im)
  for j,(key,title) in enumerate([('goal','Goal'),('keepout','Keepout'),('waypoint','Waypoint')]):
   v=status[key];ok=v in ['reached','settled','clear','visited'];d.text((x+15,842+j*46),title+': '+v.upper(),font=font(26,True),fill=GREEN if ok else RED)
 d.text((50,1025),'Selected matched illustration | Native success monitors verify the labels.',font=font(22),fill=MUTED)
 return im

def render_clip(clip,out):
 target=out/'videos'/clip['environment']/clip['family']/(clip['id']+'.mp4');writer=VideoWriter(target);images=out/'images';images.mkdir(parents=True,exist_ok=True);results=[];panels=[];started=time.time();last=None
 story=clip['family']=='matched_comparison'
 for q,t in enumerate(clip['trials'],1):
  active=['keepout','waypoint'] if story or t['role']=='composition' else [t['role']] if t['role']!='base' else []
  trial=Trial(t,active)
  try:
   status=trial.status();status['task_success']=False
   if story:status['task_success']=bool(trial.row['success'])
   native=trial.frame();initial=compose_frame(native,clip,t,trial.data,trial.trail,status,active,'setup',q if story else None)
   if q==1:initial.save(images/(clip['id']+'_setup.png'));writer.append(initial,60)
   else:writer.append(initial,24)
   steps=len(trial.data['actions']);poster_step=max(1,int(steps*.48))
   for i in range(steps):
    trial.step();status=trial.status()
    if story:status['task_success']=bool(trial.row['success'])
    frame=compose_frame(trial.frame(),clip,t,trial.data,trial.trail,status,active,'running',q if story else None)
    hz=10 if clip['environment']=='tworoom' else 20
    repeat=((i+1)*FPS)//hz-(i*FPS)//hz;writer.append(frame,repeat)
    if i+1==poster_step and (not story or t['role']=='composition'):frame.save(images/(clip['id']+'_keyframe.png'))
   result=trial.verify();results.append(result);status=trial.status(True)
   if story:status['task_success']=bool(trial.row['success'])
   native=trial.frame();last=compose_frame(native,clip,t,trial.data,trial.trail,status,active,'outcome',q if story else None);writer.append(last,30 if story else 45);panels.append((native.copy(),t,status))
  finally:trial.close()
 if story:
  assert results[0]['own_task_verified'] and not results[0]['joint_verified'] and results[1]['own_task_verified'] and not results[1]['joint_verified'] and results[2]['joint_verified']
  for key in (['start','goal','keepout','waypoint','base_z'] if clip['environment']=='tworoom' else ['initial_state','initial_cube','goal','keepout','waypoint','base_z']):
   arrays=[load_trial(t)[0][key] for t in clip['trials']];assert all(np.array_equal(arrays[0],x) for x in arrays[1:]),key
  last=summary_frame(clip,panels);writer.append(last,60);last.save(images/(clip['id']+'_comparison.png'))
 else:assert results[0]['own_task_verified']
 last.save(images/(clip['id']+'_outcome.png'));writer.close()
 return {'id':clip['id'],'environment':clip['environment'],'family':clip['family'],'category':clip.get('category',clip['trials'][0].get('category')),'video':str(target.relative_to(out)),
         'poster':'images/'+clip['id']+'_keyframe.png','frames':writer.frames,'fps':FPS,'duration_seconds':writer.frames/FPS,'bytes':target.stat().st_size,'sha256':sha(target),'render_seconds':time.time()-started,
         'native_replay_verified':True,'render_only_markers':True,'selected_illustration':True,'private_verification':results}

def main():
 p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--out',default=str(ROOT/'media_final'));a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True);s=read(ROOT/'PRIVATE_SELECTION.json')
 clips=[c for c in s['clips'] if not a.ids or c['id'] in a.ids]
 for c in clips:
  receipt=ROOT/'render_receipts_final'/(c['id']+'.json')
  if receipt.exists():
   prior=read(receipt);assert sha(out/prior['video'])==prior['sha256'];continue
  result=render_clip(c,out);write(receipt,result)
  receipts=list((ROOT/'render_receipts_final').glob('*.json'));write(ROOT/'PUBLIC_PROGRESS.json',{'stage':'rendering','completed_task_videos':len(receipts),'target_task_videos':35,'overview_pending':True,'last_completed_clip':c['id'],'raw_episode_data_exported':False});print({'completed':len(receipts),'clip':c['id']},flush=True)
 for env,models in s['models'].items():assert all(sha(d['path'])==d['sha256'] for d in models.values())
 print({'finished_requested_render_batch':len(clips),'models_unchanged':True},flush=True)
if __name__=='__main__':main()
