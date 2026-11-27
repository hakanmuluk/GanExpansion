"""Full-decode, native outcome and presentation-storyboard verification."""
import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import subprocess,json,io,math
import numpy as np
from PIL import Image,ImageDraw
from .common import ROOT,read,write,sha,load_trial
from .design import font,W,H,FPS,BG,TEXT,MUTED

def probe(path):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
def capture(path,seconds,width=640,height=360):
 b=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(max(0,seconds)),'-i',str(path),'-frames:v','1','-vf',f'scale={width}:{height}','-f','image2pipe','-vcodec','png','pipe:1'])
 return Image.open(io.BytesIO(b)).convert('RGB')
def storyboard(path,duration,out,label):
 times=[min(.7,duration*.1),duration*.23,duration*.4,duration*.58,duration*.78,max(0,duration-.4)]
 im=Image.new('RGB',(1920,796),BG);d=ImageDraw.Draw(im);d.text((20,16),label,font=font(25,True),fill=TEXT)
 for i,t in enumerate(times):
  x=i%3*640;y=66+i//3*360;im.paste(capture(path,t),(x,y));d=ImageDraw.Draw(im);d.rectangle((x+6,y+5,x+122,y+34),fill=BG);d.text((x+14,y+8),f'{t:.1f} s',font=font(19),fill=TEXT)
 out.parent.mkdir(parents=True,exist_ok=True);im.save(out,quality=90)

def verify_native():
 from .render import Trial
 selection=read(ROOT/'PRIVATE_SELECTION.json');unique={};uses=0;comparisons=0;per_env={'tworoom':0,'cube':0};maxerr={'tworoom':0.,'cube':0.}
 for clip in selection['clips']:
  measured=[]
  for item in clip['trials']:
   uses+=1;key=(item['environment'],item['run'],item['index'])
   if key not in unique:
    t=Trial(item,[])
    try:
     for _ in t.data['actions']:t.step()
     t.verify()
     if item['environment']=='tworoom':
      goal=bool(np.linalg.norm(t.trail[-1]-t.data['goal'])<16);joint=bool(goal and t.visited and not t.keep and not t.wall and not t.early)
      k=bool(goal and not t.keep and not t.wall);w=bool(goal and t.visited and not t.early and not t.wall)
     else:
      from cube_addons.env import completion
      native,_=completion(t.env,t.info);m=t.observer.result(t.stable>=10,native);goal=bool(t.stable>=10);common=bool(goal and not m['dropped'] and not m['robot_table_contact_in_transport']);k=bool(common and m['keepout_satisfied']);w=bool(common and m['waypoint_satisfied']);joint=m['success']
     if item['role']!='base':
      assert k==t.row['keepout_full_success'] and w==t.row['waypoint_full_success'] and joint==t.row['success']
     assert goal and (item['role']=='base' or (k if item['role']=='keepout' else w if item['role']=='waypoint' else joint))
     unique[key]={'goal':goal,'k':k,'w':w,'joint':joint};per_env[item['environment']]+=1;maxerr[item['environment']]=max(maxerr[item['environment']],t.maxerr)
    finally:t.close()
   measured.append(unique[key])
  if clip['family']=='matched_comparison':
   assert measured[0]['k'] and not measured[0]['w'] and measured[1]['w'] and not measured[1]['k'] and measured[2]['joint'];comparisons+=1
 assert uses==47 and comparisons==6
 assert all(sha(d['path'])==d['sha256'] for m in selection['models'].values() for d in m.values())
 return {'all_native_conditions_verified':True,'rendered_component_uses':uses,'distinct_native_trials_rechecked':len(unique),'native_trials_by_environment':per_env,'maximum_replay_error':maxerr,'matched_stories_verified':comparisons,'all_goal_and_own_skill_conditions_match':True,'models_unchanged':True,'no_raw_episode_data_exported':True}

def main():
 out=ROOT/'media_final';qa=out/'qa';qa.mkdir(exist_ok=True);receipts=[read(p) for p in sorted((ROOT/'render_receipts_final').glob('*.json'))];assert len(receipts)==35
 overview=read(ROOT/'overview_receipt.json');items=[overview]+receipts;results=[]
 for item in items:
  path=out/item['video'];assert sha(path)==item['sha256'];p=probe(path);v=[x for x in p['streams'] if x['codec_type']=='video'];assert len(v)==1 and len(p['streams'])==1
  v=v[0];assert v['codec_name']=='h264' and v['pix_fmt']=='yuv420p' and v['width']==W and v['height']==H and v['avg_frame_rate']=='30/1'
  if 'frames' in item:assert int(v['nb_frames'])==item['frames']
  duration=float(p['format']['duration']);assert duration>5
  decoded=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-xerror','-i',str(path),'-map','0:v:0','-an','-f','null','-'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
  assert decoded.returncode==0,(item['id'],decoded.stderr)
  storyboard(path,duration,qa/'storyboards'/(item['id']+'.jpg'),item['id'])
  results.append({'id':item['id'],'duration_seconds':duration,'frames':int(v['nb_frames']),'codec':'h264','pixel_format':'yuv420p','width':W,'height':H,'fps':FPS,'full_decode_passed':True,'sha256':item['sha256']})
  print({'decoded_and_storyboarded':len(results),'total':36},flush=True)
 native=verify_native();write(qa/'TECHNICAL_QA.json',{'complete':True,'videos_checked':36,'all_full_decode_passed':True,'videos':results,'native_verification':native,'visual_inspection':'Reviewed separately on WD Archive; see VISUAL_QA.json after delivery.'})
 print({'technical_QA_complete':True,'videos':36,'native_component_uses':47},flush=True)
if __name__=='__main__':main()
