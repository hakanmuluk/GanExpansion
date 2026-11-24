"""Exact disabled-policy checks across original native and randomized tasks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from cube.cube_gan.evaluate import load_models
from .evaluate import load_adapter
from .models import state_digest
from .env import make_env
from .rollout import rollout

_G=_REFERENCE=_ENV=None


def initialize(checkpoint,base):
    global _G,_REFERENCE,_ENV
    torch.set_num_threads(1);_G,_,_=load_adapter(checkpoint,base,'cpu');_REFERENCE,_,_=load_models(base)
    assert state_digest(_G.base)==state_digest(_REFERENCE)
    _ENV=make_env()


def compare(item):
    index,case,code=item
    spec=np.array([3.,3.,3.,.025])
    reference,a=rollout(_ENV,_REFERENCE,case,spec,'keepout',[code],max_steps=200)
    disabled,b=rollout(_ENV,_G,case,spec,'keepout',[code],gate=0,max_steps=200)
    errors=[]
    for key in ['states','actions','poses']:
        if not np.array_equal(a[key],b[key]):errors.append(key)
    return {'episode':index,'context_seed':int(case['seed']),'task_id':int(case['task_id']),
            'base_z':case['base_z'].tolist(),'new_z':float(code),'steps':reference['steps'],
            'original_released_success':reference['released_success'],'disabled_released_success':disabled['released_success'],
            'exact_trajectory_equality':not errors,'mismatched_arrays':errors}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--base',default='cube/artifacts/final/base_policy.pt');p.add_argument('--contexts',default='cube/artifacts/data/test_contexts.npz')
    p.add_argument('--per-task',type=int,default=2);p.add_argument('--codes',type=int,default=4);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(9491)
    with np.load(args.contexts) as data:bank={k:data[k] for k in data.files}
    items=[]
    for task in range(6):
        ids=np.flatnonzero(bank['task_id']==task)[:args.per_task]
        if len(ids)!=args.per_task:raise ValueError('Missing original task contexts')
        for i in ids:
            for _ in range(args.codes):
                case={k:v[i] for k,v in bank.items()};case['base_z']=rng.standard_normal(2).astype(np.float32)
                items.append((len(items),case,float(rng.standard_normal())))
    with mp.get_context('spawn').Pool(args.workers,initializer=initialize,initargs=(args.checkpoint,args.base)) as pool:
        records=list(pool.imap(compare,items))
    summary={'episodes':len(records),'tasks':6,'original_horizon':16,'original_execute':8,'original_timeout':200,
             'exact_trajectory_equality':all(r['exact_trajectory_equality'] for r in records),
             'all_goal_outcomes_identical':all(r['original_released_success']==r['disabled_released_success'] for r in records),
             'original_released_success':float(np.mean([r['original_released_success'] for r in records])),
             'base_sha256':hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
             'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),'records':records}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
    if not summary['exact_trajectory_equality']:raise AssertionError('Disabled policy changed original behavior')


if __name__=='__main__':main()
