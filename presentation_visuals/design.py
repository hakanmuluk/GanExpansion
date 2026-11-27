"""Render-only markers and a readable 1080p presentation layout."""
import math,subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
W,H,FPS=1920,1080,30
BG='#0B1322';CARD='#152238';TEXT='#F4F7FC';MUTED='#AFC0D7';RED='#F46D75';GREEN='#43D6A0';GOLD='#FFD267';BLUE='#69B7FF'
@lru_cache(maxsize=32)
def font(size,bold=False):
 from matplotlib import font_manager
 p=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold' if bold else 'normal'))
 return ImageFont.truetype(p,size)
def wrapped(draw,text,xy,width,size=30,fill=TEXT,spacing=9,bold=False):
 f=font(size,bold);words=text.split();line='';y=xy[1]
 for word in words:
  if line and draw.textlength(line+' '+word,font=f)>width:
   draw.text((xy[0],y),line,font=f,fill=fill);y+=size+spacing;line=word
  else:line=(line+' '+word).strip()
 if line:draw.text((xy[0],y),line,font=f,fill=fill);y+=size+spacing
 return y

def add_geom(scene,kind,pos,size,color):
 import mujoco
 assert scene.ngeom<len(scene.geoms)
 g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,kind,np.asarray(size,float),np.asarray(pos,float),np.eye(3).ravel(),np.asarray(color,np.float32));g.category=int(mujoco.mjtCatBit.mjCAT_DECOR);g.objid=-1;g.segid=-1;scene.ngeom+=1;return g

def line3(scene,a,b,color,width=.001):
 import mujoco
 g=add_geom(scene,mujoco.mjtGeom.mjGEOM_CAPSULE,(np.asarray(a)+b)/2,[width,0,0],color)
 mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,width,np.asarray(a,float),np.asarray(b,float))

def ring3(scene,center,radius,color,planes=(0,1,2),dashed=False,width=.001):
 for plane in planes:
  q=np.linspace(0,2*np.pi,41);v=np.zeros((len(q),3));axes=[j for j in range(3) if j!=plane];v[:,axes[0]]=radius*np.cos(q);v[:,axes[1]]=radius*np.sin(q);v+=center
  for i in range(len(q)-1):
   if not dashed or i%2==0:line3(scene,v[i],v[i+1],color,width)

class CubeView:
 def __init__(self,env,case,active,azimuth=175,elevation=-27):
  import mujoco
  self.env,self.case,self.active=env,case,active
  env._model.vis.global_.offwidth=max(920,env._model.vis.global_.offwidth);env._model.vis.global_.offheight=max(920,env._model.vis.global_.offheight)
  self.renderer=mujoco.Renderer(env._model,height=900,width=900)
  self.camera=mujoco.MjvCamera();self.camera.type=mujoco.mjtCamera.mjCAMERA_FREE
  self.camera.lookat[:]=[.42,0,.12];self.camera.distance=.83;self.camera.azimuth=azimuth;self.camera.elevation=elevation
 def frame(self,trail):
  import mujoco
  from cube.cube_gan.env import snapshot
  before=snapshot(self.env)
  self.renderer.update_scene(self.env._data,camera=self.camera);scene=self.renderer.scene
  # Visual colors live only in the temporary rendering scene.
  cube_id=self.env._model.geom('object_0').id;target_id=self.env._model.geom('target_object_0').id
  for geom in scene.geoms[:scene.ngeom]:
   if geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and geom.objid==cube_id:geom.rgba[:3]=[.12,.60,.90]
   elif geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and geom.objid==target_id:geom.rgba[:3]=[1.,.72,.12]
  if 'keepout' in self.active:
   k=self.case['keepout'];add_geom(scene,mujoco.mjtGeom.mjGEOM_SPHERE,k[:3],[k[3],0,0],[.95,.12,.19,.16]);ring3(scene,k[:3],k[3],[.95,.2,.25,.9],width=.0012)
   ring3(scene,k[:3],k[3]+math.sqrt(3)*.02,[1.,.28,.32,.45],dashed=True,width=.0007)
  if 'waypoint' in self.active:
   q=self.case['waypoint'];add_geom(scene,mujoco.mjtGeom.mjGEOM_SPHERE,q[:3],[q[3],0,0],[.14,.85,.48,.12]);ring3(scene,q[:3],q[3],[.10,.75,.4,.9],width=.0011)
  goal=self.case['goal'];ring3(scene,goal,.04,[1.,.67,.07,.95],planes=(2,),width=.0016)
  path=np.asarray(trail)
  if len(path)>1:
   for i in range(0,len(path)-1,2):line3(scene,path[i],path[min(i+2,len(path)-1)],[.1,.45,.95,.70],.00085)
  frame=Image.fromarray(self.renderer.render().copy())
  assert np.array_equal(before,snapshot(self.env)),'Render annotations changed integration state'
  return frame
 def close(self):self.renderer.close()

class TwoRoomView:
 def __init__(self,env,case,active):self.env,self.case,self.active=env,case,active
 def frame(self,trail):
  # Native 224px pixels; annotations do not enter the observation or dynamics.
  pixels=self.env.render().copy();f=pixels.astype(float)
  # Recolor native indicator glows only, before drawing new regions.
  # Goal is gold and the measured agent is blue; wall/background pixels stay native.
  for mask,color,strength in [((f[:,:,1]>f[:,:,0]+10)&(f[:,:,1]>f[:,:,2]+10),[255,195,35],(f[:,:,1]-np.minimum(f[:,:,0],f[:,:,2]))/255),((f[:,:,0]>f[:,:,1]+10)&(f[:,:,0]>f[:,:,2]+10),[30,105,225],(f[:,:,0]-np.minimum(f[:,:,1],f[:,:,2]))/255)]:
   a=strength[mask,None];pixels[mask]=np.clip(255*(1-a)+np.asarray(color)*a,0,255).astype(np.uint8)
  image=Image.fromarray(pixels).convert('RGBA').resize((900,900),Image.Resampling.LANCZOS)
  overlay=Image.new('RGBA',image.size);d=ImageDraw.Draw(overlay);scale=900/224
  def circle(v,color,inflate=0,dash=False):
   x,y,r=v;r+=inflate;box=[scale*(x-r),scale*(y-r),scale*(x+r),scale*(y+r)]
   if dash:
    for a in range(0,360,24):d.arc(box,a,a+12,fill=color,width=2)
   else:d.ellipse(box,fill=color[:3]+(40,),outline=color,width=4)
  if 'keepout' in self.active:
   circle(self.case['keepout'],(225,45,65,230));circle(self.case['keepout'],(225,45,65,110),7,True)
  if 'waypoint' in self.active:circle(self.case['waypoint'],(25,175,110,240))
  goal=np.r_[self.case['goal'],16];circle(goal,(218,155,15,200))
  center=self.case['goal']*scale;star=[(center[0]+math.sin(j*math.pi/5)*(34 if j%2==0 else 14),center[1]-math.cos(j*math.pi/5)*(34 if j%2==0 else 14)) for j in range(10)];d.polygon(star,fill=(240,175,25,245),outline=(170,112,15,255))
  xy=np.asarray(trail)*scale
  if len(xy)>1:d.line([tuple(p) for p in xy],fill=(25,100,205,190),width=4,joint='curve')
  image=Image.alpha_composite(image,overlay)
  # Re-emphasize the native agent location without replacing its measured center.
  if len(xy):
   d=ImageDraw.Draw(image);x,y=xy[-1];r=7*scale;d.ellipse((x-r,y-r,x+r,y+r),fill=(25,105,225,255),outline='white',width=3)
  return image.convert('RGB')
 def close(self):pass

def diagram(case,environment,trail,active):
 image=Image.new('RGB',(800,310),CARD);d=ImageDraw.Draw(image)
 if environment=='cube':
  initial=case['initial_cube'];goal=case['goal'];axis=goal[:2]-initial[:2];axis/=np.linalg.norm(axis)
  def coords(p):p=np.asarray(p);return np.c_[((p[...,:2]-initial[:2])@axis).reshape(-1)*100,p[...,2].reshape(-1)*100]
  path=coords(np.asarray(trail));points=[coords(case[k][:3][None])[0] for k in active];xlim=(-8,max(32,np.linalg.norm(goal[:2]-initial[:2])*100+5));ylim=(-1,36)
  def xy(p):return (48+(p[0]-xlim[0])/(xlim[1]-xlim[0])*700,270-(p[1]-ylim[0])/(ylim[1]-ylim[0])*225)
  d.text((22,12),'Cube path: side projection',font=font(20),fill=MUTED)
  for key in active:
   spec=case[key];center=coords(spec[:3][None])[0];r=(spec[3]+(math.sqrt(3)*.02 if key=='keepout' else 0))*100;x,y=xy(center);rx=r/(xlim[1]-xlim[0])*700;ry=r/(ylim[1]-ylim[0])*225;color=RED if key=='keepout' else GREEN
   d.ellipse((x-rx,y-ry,x+rx,y+ry),outline=color,width=2)
  for a,b in zip(path[:-1],path[1:]):d.line([xy(a),xy(b)],fill=BLUE,width=3)
  x,y=xy(coords(goal[None])[0]);d.ellipse((x-7,y-7,x+7,y+7),fill=GOLD)
  d.text((48,282),'Progress toward goal',font=font(18),fill=MUTED)
 else:
  d.text((22,12),'Actual agent path',font=font(20),fill=MUTED);scale=1.1;off=np.array([270.,38.]);path=np.asarray(trail)*scale+off
  for box in [(107,117,14,35),(107,117,77,147),(107,117,189,210)]:d.rectangle((off[0]+box[0]*scale,off[1]+box[2]*scale,off[0]+box[1]*scale,off[1]+box[3]*scale),fill='#8392A5')
  for key in active:
   spec=case[key];c=spec[:2]*scale+off;r=(spec[2]+(7 if key=='keepout' else 0))*scale;d.ellipse((*list(c-r),*list(c+r)),outline=RED if key=='keepout' else GREEN,width=2)
  if len(path)>1:d.line([tuple(p) for p in path],fill=BLUE,width=3)
 return image

def compose_frame(native,clip,trial,case,trail,status,active,phase='running',story_position=None):
 image=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(image)
 env='Cube' if clip['environment']=='cube' else 'TwoRoom';family=clip['family'];role=trial['role']
 labels={'base':'Base model','keepout':'Keepout adapter','waypoint':'Waypoint adapter','composition':'Skill composition'}
 title=env+' | '+('Same scene, three policies' if family=='matched_comparison' else labels[role])
 d.text((42,24),title,font=font(48,True),fill=TEXT)
 category=clip.get('category',trial.get('category',''))
 sub={'supported_conflict':'Conflicting constraints','coupled_ood':'Coupled / out-of-distribution','supported_easy':'Supported task layout'}.get(category,'Goal-conditioned control')
 if family=='matched_comparison':sub+=f'  |  {story_position} / 3: '+labels[role]
 elif role=='composition':sub+='  |  one Gaussian start, '+str(trial['steps'])+' Adam updates'
 else:sub+='  |  frozen learned policy'
 d.text((45,88),sub,font=font(28),fill=MUTED)
 image.paste(native.resize((900,900),Image.Resampling.LANCZOS),(40,136));d=ImageDraw.Draw(image)
 panelx=990
 setup=phase=='setup';end=phase=='outcome'
 heading='THE TASK' if setup else ('FINAL OUTCOME' if end else labels[role].upper())
 d.text((panelx,145),heading,font=font(27,True),fill=GOLD if setup else TEXT)
 if clip['environment']=='cube':task='Place the blue cube at the gold goal, release it, and let it settle.'
 else:task='Navigate through a doorway and reach the gold goal.'
 y=wrapped(d,task,(panelx,191),850,31)
 if active:
  desc='Avoid red; visit green before the goal.' if len(active)==2 else ('Keep the full body clear of red.' if active[0]=='keepout' else 'Visit green, then reach the goal.')
  if clip['environment']=='cube' and 'waypoint' in active:desc+=' Visit while holding the cube.'
  y=wrapped(d,desc,(panelx,y+9),850,26,MUTED)
 # Stable card positions, readable from presentation distance.
 cards=[('goal','GOAL',GOLD)] + [('keepout','KEEP-OUT',RED)]*('keepout' in active)+[('waypoint','WAYPOINT',GREEN)]*('waypoint' in active)
 yy=360
 for key,label,color in cards:
  value=status[key];passed=value in ['reached','settled','clear','visited'];bad=value in ['violated','missed','too early','not reached']
  badge={'pending':'pending','approaching':'approaching','reached':'REACHED','settled':'SETTLED','clear':'CLEAR','visited':'VISITED','violated':'VIOLATED','missed':'MISSED','too early':'TOO EARLY','not reached':'NOT REACHED'}.get(value,value)
  d.rounded_rectangle((panelx,yy,panelx+840,yy+76),radius=12,fill=CARD,outline=color,width=2)
  d.text((panelx+22,yy+18),label,font=font(28,True),fill=color)
  d.text((panelx+415,yy+18),badge,font=font(28,True),fill=RED if bad else GREEN if passed else MUTED);yy+=91
 if setup:message='Markers show the task; they do not change physics.'
 elif end:
  good=status.get('task_success',False)
  message=('TASK COMPLETE' if good else 'OTHER CONSTRAINT VIOLATED')
  if family=='matched_comparison' and role!='composition':message=('Keepout satisfied; waypoint missed.' if role=='keepout' else 'Waypoint satisfied; keepout violated.')
 else:message='Executing learned actions, then observing again.'
 wrapped(d,message,(panelx,650),850,28,(GREEN if status.get('task_success') else RED) if end else MUTED,bold=end)
 inset=diagram(case,clip['environment'],trail,active);image.paste(inset,(panelx,725))
 footer='Native simulation | selected illustration | 16 actions predicted, 8 executed'
 if category=='coupled_ood':footer=('Selected OOD comparison' if family=='matched_comparison' else 'Selected OOD success')+'; general OOD performance remains limited.'
 d=ImageDraw.Draw(image);d.text((45,1049),footer,font=font(19),fill=MUTED)
 return image

class VideoWriter:
 def __init__(self,path):
  self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);assert not self.path.exists();self.working=self.path.with_name(self.path.stem+'.encoding.mp4');assert not self.working.exists()
  self.log=self.path.with_suffix('.encoder.log').open('wb')
  self.p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(self.working)],stdin=subprocess.PIPE,stderr=self.log)
  self.frames=0
 def append(self,image,count=1):
  b=np.asarray(image.convert('RGB'),dtype=np.uint8).tobytes()
  for _ in range(count):self.p.stdin.write(b);self.frames+=1
 def close(self):
  self.p.stdin.close();rc=self.p.wait();self.log.close();assert rc==0,f'Video encoding failed: {self.path}';self.working.rename(self.path)
