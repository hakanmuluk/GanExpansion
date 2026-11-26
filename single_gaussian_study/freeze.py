"""Freeze the new experiment while referring to the untouched completed study."""
import argparse,json,hashlib,sys
from pathlib import Path
from copy import deepcopy
from composition_ablation_study.evaluate import verify as verify_original

CORE=['__init__.py','planning.py','freeze.py','evaluate.py','preflight.py','audit.py','run.py']
STEPS=[0,5,10,20,40];SEEDS=[241,1241,2241]
KINDS=['supported_easy','supported_conflict','coupled_ood']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def verify(s):
    assert sha(s['reference_study'])==s['reference_study_sha256']
    verify_original(read(s['reference_study']))
    for p,h in s['source_hashes'].items():assert sha(p)==h,p
    for p,h in s['extra_original_sources'].items():assert sha(p)==h,p
    for j in s['jobs']:assert sha(j['config'])==j['config_sha256']

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--out',required=True);p.add_argument('--serial-pilot');a=p.parse_args()
    old=read(a.reference);verify_original(old);out=Path(a.out).resolve();assert not out.exists();out.mkdir(parents=True)
    s={'reference_study':str(Path(a.reference).resolve()),'reference_study_sha256':sha(a.reference),
       'seeds':SEEDS,'steps':STEPS,'contexts_per_category':64,'execute':8,'prediction_horizon':16,
       'agreement_horizon':8,'environments':deepcopy(old['environments']),'jobs':[],
       'source_hashes':{str(Path(__file__).parent/n):sha(Path(__file__).parent/n) for n in CORE},
       'extra_original_sources':{},'environment':old['environment'],
       'scope':'One Gaussian pair per episode, no candidate grid or extra samples; one fixed-sign Adam trajectory, warm previous codes; original models/objective/executors/monitors frozen.',
       'initialization':'Raw standard normal; project magnitude to original bounds and permitted signs (TwoRoom negative); original amplitude; original logit fraction clamp[.001,.999] for every N including0.',
       'warm_start':'Previous selected code is the sole next-replan start; Adam state resets as in original. Fixed private branch; no new Gaussian draw.',
       'uncertainty':'Paired task contexts and independently sampled initialization seeds; never choose best seed.'}
    s['evaluation_workers']=4
    if a.serial_pilot:s['serial_pilot']=str(Path(a.serial_pilot).resolve())
    for env,e in s['environments'].items():
        for n,h in read(e['old_freeze'])['original_sources'].items():
            path=Path(n) if Path(n).is_absolute() else Path(old['original_root'])/n
            assert sha(path)==h;s['extra_original_sources'][str(path)]=h
        e['previous_results']={}
        for n in STEPS:
            job=next(j for j in old['jobs'] if j['environment']==env and j['setting']==f'h8_n{n}' and j['category']==KINDS[0])
            # Each already reported category is retained with a file hash.
            for k in KINDS:
                j=next(j for j in old['jobs'] if j['environment']==env and j['setting']==f'h8_n{n}' and j['category']==k)
                path=Path(j['out'])/'summary.json';e['previous_results'][f'n{n}/{k}']={'path':str(path),'sha256':sha(path)}
            c=deepcopy(e['reference_config']);c.update(steps=n,starts=1)
            folder=out/'configs'/env;folder.mkdir(parents=True,exist_ok=True);config=folder/f'n{n}.json';write(config,c)
            for seed in SEEDS:
                for kind in KINDS:
                    name=f'{env}_n{n}_s{seed}_{kind}'
                    s['jobs'].append({'id':name,'environment':env,'steps':n,'seed':seed,'category':kind,
                        'config':str(config),'config_sha256':sha(config),'out':str(out/'evaluations'/name)})
    # Run N0 before refinements so paired random initializations can be audited.
    s['jobs'].sort(key=lambda j:(STEPS.index(j['steps']),SEEDS.index(j['seed']),KINDS.index(j['category']),j['environment']))
    assert len(s['jobs'])==90;write(out/'study.json',s);verify(s)
    print(json.dumps({'study':str(out/'study.json'),'jobs':90,'evaluations':5760,'contexts':384}))

if __name__=='__main__':main()
