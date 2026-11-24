"""Optional real-physics expert queries on learner states from training contexts.

This is DAgger-style extra supervision, not finite-demonstration-only training.
All expert calls and actual learner trajectories are saved and counted.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import argparse
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from cube.cube_gan.style import decode_style
from cube.cube_gan.expert import Style
from .env import make_env,restore,features,step_traced
from .geometry import Observer,holding,MAX_STEPS
from .expert import SkillExpert
from .contacts import contact_info,force_holding
from .evaluate import load_adapter,load_contexts
from .rollout import rollout
from .data import atomic_npz

_MODEL=_ENV=_ARGS=None


def init_worker(args):
    global _MODEL,_ENV,_ARGS
    torch.set_num_threads(1);_ARGS=args
    _MODEL,_,_=load_adapter(args.checkpoint,args.base,'cpu');_ENV=make_env()


def collect_worker(item):
    case,code,index=item;env=_ENV;skill=_ARGS.skill
    row,traj=rollout(env,_MODEL,case,case['spec'],skill,code,save_snapshots=True)
    output=Path(_ARGS.out);output.mkdir(parents=True,exist_ok=True)
    # Save physical trajectories for separate encoder calibration/audits.
    atomic_npz(output/f'learner_{index:04d}.npz',**traj,base_z=case['base_z'],new_z=code,
               initial=case['initial_cube'],goal=case['goal'],spec=case['spec'],seed=case['seed'])
    if getattr(_ARGS,'rollouts_only',False):
        row.update(context_seed=int(case['seed']),new_z=np.asarray(code).tolist(),expert_action_queries=0,chunks=0)
        (output/f'record_{index:04d}.json').write_text(json.dumps(row,indent=2)+'\n')
        return row
    rng=np.random.default_rng(_ARGS.seed+index);available=np.arange(len(traj['snapshot_frames']))
    if len(available)>_ARGS.frames:available=np.sort(rng.choice(available,_ARGS.frames,replace=False))
    samples={k:[] for k in ['states','contexts','actions','mask','base_z','new_z']}
    mode,amp,speed=decode_style(torch.as_tensor(case['base_z']))
    for chosen in available:
        t=int(traj['snapshot_frames'][chosen]);ob,info=restore(env,traj['snapshots'][chosen])
        use_forces=bool(case.get('force_contacts',False))
        if use_forces:contact_info(env,info)
        observer=Observer(case['initial_cube'],case['goal'],case['spec'],skill,force_holding if use_forces else None)
        observer.has_lifted=bool(traj['contexts'][t,14]>.5)
        observer.visited=bool(traj['contexts'][t,15]>.5)
        observer.previous_holding=observer.is_holding(info)
        expert=SkillExpert(case['initial_cube'],case['goal'],Style(int(mode),float(amp),float(speed)),case['spec'],code,skill)
        cube=np.asarray(info['privileged/block_0_pos']);progress=float(traj['contexts'][t,10])
        expert.ready=bool(observer.has_lifted and observer.is_holding(info) and cube[2]>.08 and (cube[2]>=.165 or progress>.12))
        # A learner-state query must retain the release commitment once it has
        # already delivered the cube inside the native 4cm goal tolerance.
        # Recreating a fresh expert otherwise asked it to re-grasp settled
        # cubes a few mm outside the demonstrations' 2.2cm release threshold.
        can_finish=skill=='keepout' or observer.visited
        expert.releasing=bool(can_finish and observer.has_lifted and cube[2]<.04 and np.linalg.norm(cube[:2]-case['goal'][:2])<.035)
        target=[]
        plan_visited=observer.visited
        for j in range(16):
            if getattr(_ARGS,'macro_waypoint',False):
                if j%8==0:plan_visited=observer.visited
                action,_=expert.action(info,observer,visited_override=plan_visited)
            else:action,_=expert.action(info,observer)
            target.append(action)
            before=info['privileged/block_0_pos'].copy()
            result,path=step_traced(env,action);ob,_,_,_,info=result
            if use_forces:contact_info(env,info)
            observer.update(before,info['privileged/block_0_pos'],info,path)
        samples['states'].append(traj['states'][t]);samples['contexts'].append(traj['contexts'][t])
        samples['actions'].append(target);samples['mask'].append(np.ones(16,np.float32))
        samples['base_z'].append(case['base_z']);samples['new_z'].append(code)
    arrays={k:np.asarray(v,np.float32) for k,v in samples.items()}
    atomic_npz(output/f'queries_{index:04d}.npz',**arrays)
    row.update(context_seed=int(case['seed']),new_z=np.asarray(code).tolist(),expert_action_queries=16*len(available),chunks=len(available))
    (output/f'record_{index:04d}.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--skill',required=True,choices=['keepout','waypoint'])
    p.add_argument('--base',default='cube/artifacts/final/base_policy.pt');p.add_argument('--contexts',required=True)
    p.add_argument('--count',type=int,default=32);p.add_argument('--codes',type=int,default=4);p.add_argument('--frames',type=int,default=12)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=731);p.add_argument('--out',required=True)
    p.add_argument('--rollouts-only',action='store_true',help='Collect actual trajectories for E calibration without querying the expert')
    p.add_argument('--macro-waypoint',action='store_true',help='Commit each queried eight-step prefix to its initially observed waypoint phase')
    args=p.parse_args()
    if Path(args.contexts).name!='train_contexts.npz':raise ValueError('Feedback must use training contexts, never validation/test contexts')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);cases=load_contexts(args.contexts,args.count)
    rng=np.random.default_rng(args.seed);items=[]
    for c in cases:
        values=np.abs(rng.standard_normal((args.codes+1)//2));codes=np.r_[-values,values][:args.codes]
        for code in codes:items.append((c,np.array([code],np.float32),len(items)))
    started=time.time();records=[];source_sha=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    with mp.get_context('spawn').Pool(args.workers,initializer=init_worker,initargs=(args,)) as pool:
        for row in pool.imap_unordered(collect_worker,items):
            records.append(row)
            if len(records)%8==0:print(json.dumps({'learner_episodes':len(records),'elapsed':time.time()-started}),flush=True)
    arrays={}
    if not args.rollouts_only:
        for k in ['states','contexts','actions','mask','base_z','new_z']:
            arrays[k]=np.concatenate([np.load(out/f'queries_{i:04d}.npz')[k] for i in range(len(items))])
        atomic_npz(out/'feedback.npz',**arrays)
    if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()!=source_sha:raise RuntimeError('Actor checkpoint changed during collection')
    summary={'args':vars(args),'actual_learner_rollouts':len(records),'expert_action_queries':sum(r['expert_action_queries'] for r in records),
             'feedback_chunks':len(arrays['states']) if arrays else 0,'generator_checkpoint_sha256':source_sha,
             'learner_full_success':float(np.mean([r['success'] for r in records])),
             'elapsed_seconds':time.time()-started,'records':records}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
