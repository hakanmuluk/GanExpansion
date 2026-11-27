"""Shared paths and immutable-input readers for presentation curation."""
from pathlib import Path
import hashlib,json,os
import numpy as np
ROOT=Path(os.environ.get('GANEXPANSION_PRESENTATION_ROOT', 'outputs/presentation')).resolve()
STUDY=Path(os.environ.get('GANEXPANSION_SINGLE_STUDY', 'research_assets/single_gaussian/study.json')).resolve()
REFERENCE=Path(os.environ.get('GANEXPANSION_REFERENCE_STUDY', 'research_assets/composition_reference/study.json')).resolve()
KINDS=['supported_easy','supported_conflict','coupled_ood']
def read(p):return json.loads(Path(p).read_text())
def write(p,s):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp.'+str(os.getpid()));temp.write_text(json.dumps(s,indent=2)+'\n');temp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def job(study,env,kind,steps=None,seed=None,skill=None):
 jobs=[j for j in study['jobs'] if j['environment']==env and j['category']==kind]
 if skill is not None:jobs=[j for j in jobs if 'diagnostic_single_'+skill in j['id']]
 else:jobs=[j for j in jobs if j['steps']==steps and j['seed']==seed]
 assert len(jobs)==1
 return jobs[0]
def load_trial(entry):
 env=entry['environment'];folder=Path(entry['run']);i=entry['index'];row=read(folder/'records.json')[i]
 if env=='tworoom':
  with np.load(folder/'trajectories.npz') as z:
   length=int(z['lengths'][i]);data={k:z[k][i].copy() for k in z.files if k not in ['positions','actions','lengths']}
   data['positions']=z['positions'][i,:length+1].copy();data['actions']=z['actions'][i,:length].copy()
 else:
  with np.load(folder/f'rollout_{i:04d}.npz') as z:data={k:z[k].copy() for k in z.files}
 return data,row

def entry(j,i,role):
 return {'environment':j['environment'],'category':j['category'],'run':j['out'],'index':i,'role':role,
         'method':'single_gaussian' if role=='composition' else 'single_adapter','steps':j.get('steps',0),'seed':j.get('seed',241)}
