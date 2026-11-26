"""Generate independent diverse Cube demonstrations in native physics only."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import importlib
import json
import multiprocessing as mp
from pathlib import Path
from statistics import NormalDist
import time
import numpy as np
import torch
from .cube_diverse_demonstrations import DiverseSkillExpert,independent_spec,angle_from_code
from cube_addons.env import make_env,close_env
from cube_addons.probe import case
from cube_addons.data import atomic_npz
from cube_addons.geometry import MAX_STEPS

_ENV=None


def initialize():
    global _ENV
    torch.set_num_threads(1);_ENV=make_env()


def worker(job):
    seed,skill,directory,codes_count,jitter=job
    directory=Path(directory);stem=directory/str(seed)
    assert not stem.with_suffix('.json').exists()
    c,style=case(_ENV,seed);spec=independent_spec(c,style,skill,seed)
    c.update(force_contacts=True,macro_waypoint=skill=='waypoint',planning_interval=8)
    rng=np.random.default_rng(seed+433)
    quantiles=(np.arange(codes_count)+rng.uniform((1-jitter)/2,(1+jitter)/2,codes_count))/codes_count
    codes=np.array([[NormalDist().inv_cdf(float(q))] for q in quantiles],np.float32)
    module=importlib.import_module('cube_addons.rollout');original=module.SkillExpert
    module.SkillExpert=DiverseSkillExpert
    rows=[];trajectories=[]
    try:
        for code in codes:
            row,traj=module.rollout(_ENV,None,c,spec,skill,code,expert=True)
            row['latent_angle']=angle_from_code(code);rows.append(row);trajectories.append(traj)
    finally:module.SkillExpert=original
    arrays={**c,'spec':spec,'new_z':codes}
    for key in ['states','contexts','actions','poses','phase']:
        shape=(len(codes),MAX_STEPS+int(key=='poses'),*trajectories[0][key].shape[1:])
        arrays[key]=np.zeros(shape,trajectories[0][key].dtype)
        for i,traj in enumerate(trajectories):arrays[key][i,:len(traj[key])]=traj[key]
    arrays['lengths']=np.array([len(t['actions']) for t in trajectories],np.int32)
    atomic_npz(stem.with_suffix('.npz'),**arrays)
    meta={'seed':seed,'skill':skill,'records':rows,'demonstrations':len(rows),
          'successful':sum(r['success'] for r in rows),'data_only_teacher':True,'other_skill_observed':False}
    stem.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--contexts',type=int,default=8);p.add_argument('--codes',type=int,default=8)
    p.add_argument('--seed',type=int,default=1700001);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--jitter',type=float,default=.9,help='Within-stratum Gaussian-quantile spread; pilot used .2')
    p.add_argument('--skills',nargs='+',default=['keepout','waypoint']);args=p.parse_args()
    out=Path(args.out);assert not out.exists(),out;out.mkdir(parents=True)
    assert 0<args.jitter<1
    jobs=[]
    for skill in args.skills:
        folder=out/skill/'raw';folder.mkdir(parents=True)
        jobs.extend((args.seed+i,skill,str(folder),args.codes,args.jitter) for i in range(args.contexts))
    (out/'config.json').write_text(json.dumps({'args':vars(args),'scope':__doc__,
        'policy_dimensions_unchanged':{'state':47,'own_context':20,'skill_z':1,'action_chunk':[16,5]},
        'training_only_paths_never_policy_inputs':True,'independent_single_skill_only':True},indent=2)+'\n')
    started=time.time();records=[]
    with mp.get_context('spawn').Pool(args.workers,initializer=initialize) as pool:
        for row in pool.imap_unordered(worker,jobs,chunksize=1):
            records.append(row)
            progress={'contexts_done':len(records),'contexts_total':len(jobs),'elapsed':time.time()-started,
                'skills':{skill:{'episodes':sum(r['demonstrations'] for r in records if r['skill']==skill),
                    'successful':sum(r['successful'] for r in records if r['skill']==skill)} for skill in args.skills}}
            (out/'status.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps(progress),flush=True)
    summary={'seconds':time.time()-started,'skills':{}}
    for skill in args.skills:
        rows=[x for r in records if r['skill']==skill for x in r['records']]
        summary['skills'][skill]={'episodes':len(rows),'success_count':sum(r['success'] for r in rows),
            'goal_success_count':sum(r['released_success'] for r in rows),
            'constraint_success_count':sum(r['constraint_success'] for r in rows),
            'keepout_violations':sum(r['violated'] for r in rows),
            'table_violations':sum(r['table_violation'] for r in rows),
            'mean_steps':float(np.mean([r['steps'] for r in rows])),
            'angle_bin_counts':np.histogram([r['latent_angle'] for r in rows],np.linspace(-np.pi,np.pi,9))[0].tolist(),
            'successful_angle_bin_counts':np.histogram([r['latent_angle'] for r in rows if r['success']],np.linspace(-np.pi,np.pi,9))[0].tolist()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'contexts':len(records),'episodes':len(records)*args.codes,'native_physics_only':True})+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
