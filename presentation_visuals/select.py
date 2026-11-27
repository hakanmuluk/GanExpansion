"""Outcome-selected illustrations only; detailed case selection stays on GPU."""
from pathlib import Path
import json
import numpy as np
from .common import ROOT,STUDY,REFERENCE,KINDS,read,write,job,entry,load_trial,sha

def vector(e):
 d,r=load_trial(e)
 keys=['start','goal','keepout','waypoint'] if e['environment']=='tworoom' else ['initial_cube','goal','keepout','waypoint']
 return np.concatenate([np.asarray(d[k]).ravel() for k in keys]).astype(float)
def diverse(candidates,n,excluded=()):
 # Prefer distinct contexts, then maximize geometric variety among valid outcomes.
 unique={}
 for e in candidates:
  key=(e['category'],e['index'])
  if key not in excluded:unique.setdefault(key,e)
 values=list(unique.values());assert len(values)>=n,(len(values),n)
 x=np.stack([vector(e) for e in values]);scale=np.std(x,axis=0);scale=np.maximum(scale,1. if values[0]['environment']=='tworoom' else .05);x=x/scale
 selected=[0]
 while len(selected)<n:
  distance=((x[:,None]-x[selected][None])**2).sum(-1).min(1);distance[selected]=-1;selected.append(int(distance.argmax()))
 return [values[i] for i in selected]

def main():
 s=read(STUDY);ref=read(REFERENCE);ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'PRIVATE_SELECTION.json').exists()
 clips=[];sources=[];coverage={}
 for env in ['tworoom','cube']:
  # Fixed budget per environment, with all initialization seeds eligible as illustrations.
  n=20 if env=='tworoom' else 40
  successes={}
  singles={}
  for kind in KINDS:
   candidates=[]
   for seed in s['seeds']:
    j=job(s,env,kind,n,seed)
    for i,r in enumerate(read(Path(j['out'])/'records.json')):
     if r['success']:candidates.append(entry(j,i,'composition'))
   successes[kind]=candidates
   singles[kind]={}
   for skill in ['keepout','waypoint']:
    j=job(ref,env,kind,skill=skill);rows=read(Path(j['out'])/'records.json');singles[kind][skill]=(j,rows)
  # Independent skills: three distinct supported scenes per skill, original fixed private code.
  for skill in ['keepout','waypoint']:
   candidates=[]
   for kind in ['supported_easy','supported_conflict']:
    j,rows=singles[kind][skill]
    candidates += [entry(j,i,skill) for i,r in enumerate(rows) if r[skill+'_full_success']]
   for q,e in enumerate(diverse(candidates,3),1):
    clips.append({'id':f'{env}_{skill}_{q:02d}','environment':env,'family':skill,'trials':[e]})
  # Joint solutions. TwoRoom has only two distinct verified OOD successes.
  for kind,family,count in [('supported_conflict','composition_conflict',3),('coupled_ood','composition_ood',2 if env=='tworoom' else 3)]:
   for q,e in enumerate(diverse(successes[kind],count),1):clips.append({'id':f'{env}_{family}_{q:02d}','environment':env,'family':family,'trials':[e]})
  # Two supported-conflict stories plus one OOD story.
  for kind,count,offset in [('supported_conflict',2,0),('coupled_ood',1,2)]:
   candidates=[];jk,kr=singles[kind]['keepout'];jw,wr=singles[kind]['waypoint']
   for e in successes[kind]:
    i=e['index'];k,w=kr[i],wr[i]
    if k['keepout_full_success'] and not k['waypoint_satisfied'] and w['waypoint_full_success'] and not w['keepout_satisfied']:candidates.append(e)
   chosen=diverse(candidates,count)
   for q,e in enumerate(chosen,offset+1):
    i=e['index'];clips.append({'id':f'{env}_matched_story_{q:02d}','environment':env,'family':'matched_comparison','category':kind,
                             'trials':[entry(jk,i,'keepout'),entry(jw,i,'waypoint'),e]})
  coverage[env]={family:sum(c['environment']==env and c['family']==family for c in clips) for family in ['keepout','waypoint','composition_conflict','composition_ood','matched_comparison']}
 assert len(clips)==29 # 6 fresh base clips follow, plus one overview = 36.
 models={env:{k:{'path':p,'sha256':sha(p)} for k,p in e['models'].items()} for env,e in s['environments'].items()}
 selection={'study_sha256':sha(STUDY),'reference_sha256':sha(REFERENCE),'models':models,'clips':clips,'expected_task_videos':35,'expected_overview_videos':1,'raw_selection_stays_on_GPU':True}
 write(ROOT/'PRIVATE_SELECTION.json',selection)
 write(ROOT/'PUBLIC_PROGRESS.json',{'stage':'selection_complete','selected_existing_task_clips':len(clips),'fresh_base_clips_pending':6,'final_target':36,'coverage':coverage,'curation':'Selected successful illustrations and matched single-skill counterexamples; not new benchmark estimates.','raw_episode_data_exported':False})
 print(json.dumps(read(ROOT/'PUBLIC_PROGRESS.json')))
if __name__=='__main__':main()
