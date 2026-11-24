"""Causal original-code sweeps with initial physics and constraint held fixed."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from .evaluate import load_adapter,load_contexts
from .env import make_env
from .rollout import rollout
from .geometry import axis_frame

_MODEL=_ENV=_SKILL=None


def initialize(checkpoint,base,skill):
    global _MODEL,_ENV,_SKILL
    torch.set_num_threads(1);_MODEL,_,_=load_adapter(checkpoint,base,'cpu');_ENV=make_env();_SKILL=skill


def sweep(item):
    index,original=item;rows=[]
    sign=1 if original['base_z'][0]>=0 else -1
    for axis,settings in [('speed',[-1.5,1.5]),('side',[-1.,1.]),('amplitude',[sign*.3,sign*1.5])]:
        for modifier in [-1.,1.]:
            for setting in settings:
                case={**original,'base_z':original['base_z'].copy()}
                case['base_z'][1 if axis=='speed' else 0]=setting
                row,traj=rollout(_ENV,_MODEL,case,case['spec'],_SKILL,[modifier])
                xyz=traj['poses'][:-1,3:];values=np.array([axis_frame(case['initial_cube'],case['goal'],point)[:2] for point in xyz])
                early=(values[:,0]>.15)&(values[:,0]<.35)&(traj['contexts'][:,13]>.5)
                row.update(context=index,sweep=axis,setting=setting,modifier=modifier,
                           early_lateral=float(values[early,1].mean()) if early.any() else None,
                           base_z=case['base_z'].tolist())
                rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--contexts',required=True);p.add_argument('--out',required=True)
    p.add_argument('--base',default='cube/artifacts/final/base_policy.pt');p.add_argument('--skill',required=True,choices=['keepout','waypoint'])
    p.add_argument('--count',type=int,default=8);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);cases=load_contexts(args.contexts,args.count);rows=[]
    with mp.get_context('spawn').Pool(args.workers,initializer=initialize,initargs=(args.checkpoint,args.base,args.skill)) as pool:
        for batch in pool.imap(sweep,enumerate(cases)):rows.extend(batch)
    speed=[];amplitude=[];side=[]
    for context in range(len(cases)):
        for modifier in [-1.,1.]:
            get=lambda axis:[r for r in rows if r['context']==context and r['modifier']==modifier and r['sweep']==axis]
            pair=sorted(get('speed'),key=lambda r:r['setting'])
            if all(r['success'] for r in pair):speed.append(pair[1]['steps']-pair[0]['steps'])
            pair=sorted(get('amplitude'),key=lambda r:abs(r['setting']))
            if all(r['success'] and r['early_lateral'] is not None for r in pair):amplitude.append(abs(pair[1]['early_lateral'])-abs(pair[0]['early_lateral']))
            pair=get('side')
            side.extend(np.sign(r['original_route_lateral'])==np.sign(r['setting']) for r in pair if r['success'])
    summary={'args':vars(args),'episodes':len(rows),'full_success':float(np.mean([r['success'] for r in rows])),
             'successful_speed_pairs':len(speed),'fast_code_completes_sooner_rate':float(np.mean(np.array(speed)<0)) if speed else None,
             'fast_minus_slow_steps':float(np.mean(speed)) if speed else None,
             'successful_amplitude_pairs':len(amplitude),'larger_code_increases_early_lateral_excursion_rate':float(np.mean(np.array(amplitude)>0)) if amplitude else None,
             'early_lateral_excursion_gain_m':float(np.mean(amplitude)) if amplitude else None,
             'successful_side_sweep_alignment':float(np.mean(side)) if side else None,
             'scope':'Counterfactual stress test: initial physics and sphere fixed while original code changes. A waypoint can force a particular midpoint side; no exact unconstrained path preservation is claimed.'}
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
