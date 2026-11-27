"""Two independent native rendering workers; never restart a failed worker."""
import os,sys,time,subprocess
from pathlib import Path
from .common import ROOT,read,write,sha

def main():
 root=ROOT/'render_supervisor_final';assert not root.exists();root.mkdir()
 s=read(ROOT/'PRIVATE_SELECTION.json');children=[];sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
 for env in ['tworoom','cube']:
  ids=[c['id'] for c in s['clips'] if c['environment']==env];log=(root/(env+'.log')).open('wb')
  cmd=[sys.executable,'-u','-m','presentation_visuals.render','--ids',*ids]
  p=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[1],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT);children.append((env,p,log,len(ids)))
 write(root/'processes.json',{'supervisor':os.getpid(),'children':[{'environment':e,'pid':p.pid,'clips':n} for e,p,l,n in children],'source_sha256':sources})
 while True:
  status=[{'environment':e,'pid':p.pid,'returncode':p.poll(),'expected_clips':n} for e,p,l,n in children]
  complete=len(list((ROOT/'render_receipts_final').glob('*.json')))
  write(root/'status.json',{'completed_clips':complete,'expected_clips':35,'workers':status,'raw_episode_data_exported':False})
  if all(p.poll() is not None for _,p,_,_ in children):break
  time.sleep(5)
 for _,p,log,_ in children:log.close()
 assert all(p.returncode==0 for _,p,_,_ in children),status
 assert complete==35
 assert sources=={n:sha(Path(__file__).parent/n) for n in sources},'Source changed during final rendering'
 write(root/'DONE.json',{'complete':True,'task_videos':35,'workers':status,'source_unchanged':True})
 print({'final_task_render_complete':35},flush=True)
if __name__=='__main__':main()
