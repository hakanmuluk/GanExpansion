"""A short preview assembled exclusively from the verified task videos."""
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw
from .common import ROOT,read,write,sha
from .design import VideoWriter,font,wrapped,W,H,BG,TEXT,MUTED,GOLD,GREEN,RED

def card(title,subtitle,note):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 wrapped(d,title,(100,220),1720,72,TEXT,bold=True,spacing=20)
 wrapped(d,subtitle,(105,455),1700,40,MUTED,spacing=20)
 for x,color,label in [(120,RED,'Keepout'),(710,GREEN,'Waypoint'),(1320,GOLD,'Goal')]:
  d.ellipse((x,710,x+65,775),fill=color);d.text((x+95,710),label,font=font(42,True),fill=TEXT)
 wrapped(d,note,(105,930),1700,27,MUTED)
 return im

def main():
 assert (ROOT/'render_supervisor_final/DONE.json').exists()
 out=ROOT/'media_final';dest=out/'videos/00_overview.mp4';assert not dest.exists();work=ROOT/'overview_work';work.mkdir()
 intro=card('Generative policies + independent skills','TwoRoom navigation | Native Cube manipulation','Preview excerpts from selected native model rollouts. Full videos are organized by task.')
 outro=card('From one skill to compatible skills','Base models, independent adapters, and latent composition','Curated illustrations, not success-rate estimates. Coupled/OOD remains difficult.')
 for name,im in [('intro',intro),('outro',outro)]:
  w=VideoWriter(work/(name+'.mp4'));w.append(im,75);w.close()
 wanted=['tworoom_base_01','cube_base_01','cube_keepout_01','cube_waypoint_01','tworoom_composition_conflict_01','cube_composition_conflict_01','tworoom_composition_ood_01','cube_composition_ood_01','cube_matched_story_01']
 receipts={p.stem:read(p) for p in (ROOT/'render_receipts_final').glob('*.json')};segments=[work/'intro.mp4']
 for i,key in enumerate(wanted):
  r=receipts[key];start=max(0,r['duration_seconds']-4) if 'matched_story' in key else 2.
  target=work/f'segment_{i:02d}.mp4'
  subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(start),'-i',str(out/r['video']),'-t','4','-an','-vf','fps=30,setsar=1','-c:v','libx264','-preset','veryfast','-crf','19','-threads','2','-pix_fmt','yuv420p',str(target)],check=True)
  segments.append(target)
 segments.append(work/'outro.mp4');listing=work/'concat.txt';listing.write_text(''.join("file '"+str(p)+"'\n" for p in segments))
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-an','-c:v','libx264','-preset','veryfast','-crf','19','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
 intro.save(out/'images/00_overview_keyframe.png')
 write(ROOT/'overview_receipt.json',{'id':'00_overview','environment':'both','family':'overview','video':'videos/00_overview.mp4','poster':'images/00_overview_keyframe.png','sha256':sha(dest),'bytes':dest.stat().st_size,'source_clip_ids':wanted,'only_verified_encoded_video_excerpts':True})
 print({'overview_complete':True,'source_clips':len(wanted)},flush=True)
if __name__=='__main__':main()
