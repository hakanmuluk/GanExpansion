"""Validation-only extension selection and immutable held-out source protocol."""
import argparse,datetime,hashlib,json,shutil,subprocess,xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',default='cube_composition_artifacts');p.add_argument('--launch',action='store_true');args=p.parse_args();r=Path(args.artifacts)
    for study in ['validation','ablation_validation','encoder_causal_validation','grip_blend_validation']:
        assert (r/study/'DONE.json').exists(),study+' is incomplete'
    parity=json.loads((r/'executor_parity.json').read_text());assert parity['frozen_unchanged'] and parity['pairs']==24
    suites=list(ET.parse(r/'tests_v6.xml').getroot().iter('testsuite'));assert sum(int(s.attrib['tests']) for s in suites)>=10 and all(int(s.attrib['failures'])==int(s.attrib['errors'])==0 for s in suites)
    dest=r/'protocols/extensions';assert not dest.exists(),'Refuse to overwrite a frozen extension protocol'
    validation=r/'validation';tasks=['val_supported_easy','val_supported_conflict'];selection={}
    def candidates(names):
        result=[]
        for name in names:
            values=[json.loads((validation/task/name/'summary.json').read_text())['success'] for task in tasks]
            config=json.loads((validation/tasks[0]/name/'config.json').read_text())['search']
            result.append({'name':name,'success_by_stratum':dict(zip(tasks,values)),'minimum':min(values),'mean':float(np.mean(values)),'search':config})
        return result
    for family in ['projected_waypoint','common']:
        rows=candidates([f'{family}_m{margin}_{sign}' for margin in [5,8] for sign in ['neg','pos']])
        best=max(rows,key=lambda row:(row['minimum'],row['mean'],-row['search']['common_margin'],row['search']['fixed_codes'][1]>0))
        selection[family]={'candidates':rows,'selected':best}
    rows=candidates(['common_adam_'+temporal for temporal in ['fixed','warm','tail','cold']]);priority={'fixed':3,'warm':2,'tail':1,'cold':0}
    best=max(rows,key=lambda row:(row['minimum'],row['mean'],priority[row['search']['temporal']]))
    selection['common_latent']={'candidates':rows,'selected':best}
    jobs=[]
    def add(name,search,count=None):
        job={'name':name,'search':deepcopy(search)}
        if count is not None:job['count']=count
        jobs.append(job)
    add('native_select_full4',dict(method='random',family='best',objective='task',candidates=4,initialization='structured',temporal='fixed',regularizer='empirical',lookahead=240,workers=4))
    add('native_latent_256',dict(method='cem',steps=2,candidates=4,family='waypoint',objective='task',lookahead=32,initialization='encoder',temporal='warm',regularizer='empirical',workers=4))
    add('direct_action_256',dict(method='action',steps=1,candidates=7,family='waypoint',objective='task',lookahead=32,temporal='warm',regularizer='none',workers=4))
    add('direct_action_CEM272',dict(method='action',steps=2,candidates=8,family='waypoint',objective='task',lookahead=16,temporal='warm',regularizer='none',workers=4))
    add('common_native_256',dict(method='cem',steps=2,candidates=4,family='common',objective='task',lookahead=32,initialization='encoder',temporal='warm',regularizer='empirical',workers=4))
    add('native_trajectory_256',dict(method='random',candidates=4,family='waypoint',objective='trajectory',lookahead=32,initialization='encoder',temporal='warm',regularizer='empirical',workers=4))
    add('projected_waypoint_selected',selection['projected_waypoint']['selected']['search'])
    add('common_pair_selected',selection['common']['selected']['search'])
    blend=deepcopy(selection['common']['selected']['search']);blend['family']='grip_safe_blend';add('grip_safe_blend',blend)
    add('common_latent_selected',selection['common_latent']['selected']['search'])
    for family in ['waypoint','common']:
        for init in ['encoder','encoder_prefix']:
            add(f'{family}_{init}_ablation',dict(method='adam',family=family,steps=8,starts=4,initialization=init,regularizer='empirical',temporal='warm'),16)
    dest.mkdir(parents=True);source=dest/'source';shutil.copytree(Path(__file__).parent,source,ignore=shutil.ignore_patterns('__pycache__'))
    (dest/'plan.json').write_text(json.dumps(jobs,indent=2)+'\n')
    decisions={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_scope':'Validation only. No learned-policy test results inspected for selection.',
        'selection_rule':'Maximize worst supported-stratum joint success, then mean. Fixed ties prefer smaller margin, then positive mode. Temporal ties prefer fixed, warm, tail, cold in that order.',
        'selected':selection,'count_per_stratum':32,'encoder_ablation_count_per_stratum':16,'model_search_seed_pairs':[[42,41],[43,73],[44,101]],
        'native_budget':'Latent CEM2x4H32 and direct7H32+referenceH32 each have256maximum native controls/replan. Early goal stopping can reduce actual counts. Strong directCEM2x8H16+reference uses272.',
        'test_task_sha256':{kind:hashlib.sha256((r/f'tasks_v2/test_{kind}.npz').read_bytes()).hexdigest() for kind in ['supported_easy','supported_conflict','coupled_ood']},
        'validation_summary_sha256':{str(path.relative_to(r)):hashlib.sha256(path.read_bytes()).hexdigest() for study in ['validation','ablation_validation','encoder_causal_validation','grip_blend_validation'] for path in (r/study).glob('*/*/summary.json')},
        'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob('*.py')}}
    (dest/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n');processes=[]
    if args.launch:
        for model_seed,search_seed in decisions['model_search_seed_pairs']:
            argv=['python','-m','cube_composition_artifacts.protocols.extensions.source.queue','--plan',str(dest/'plan.json'),'--tasks',*[str(r/f'tasks_v2/test_{kind}.npz') for kind in decisions['test_task_sha256']],
                '--out',str(r/f'main_extensions/seed{model_seed}'),'--model-seed',str(model_seed),'--seed',str(search_seed),'--count','32']
            with (r/f'main_extensions_seed{model_seed}.log').open('w') as log:process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append({'pid':process.pid,'argv':argv});print(model_seed,process.pid,flush=True)
    (dest/'processes.json').write_text(json.dumps(processes,indent=2)+'\n');print(json.dumps({k:v['selected']['name'] for k,v in selection.items()},indent=2))


if __name__=='__main__':main()
