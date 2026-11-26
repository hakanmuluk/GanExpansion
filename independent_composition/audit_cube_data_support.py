"""OFFLINE teacher-family support audit, never an inference composition rule.

Each expert receives its own constraint only. Both constraints are monitored
after native execution. Any-code rates are capacity diagnostics, not deployable
success rates, and no selected joint witness enters the training dataset.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import multiprocessing as mp
from pathlib import Path
from statistics import NormalDist
import time
import numpy as np
import torch
from cube_addons.env import make_env,step_traced,completion
from cube_addons.contacts import contact_info,force_holding
from cube_addons.geometry import Observer
from cube_addons.expert import SkillExpert
from cube.cube_gan.style import decode_style
from cube.cube_gan.expert import Style
from cube_composition.scene import prepare
from cube_composition.geometry import JointObserver
from cube_composition.evaluate import load_cases
from .cube_diverse_demonstrations import DiverseSkillExpert
from .legacy import digest_file

_ENV=None


def initialize():
    global _ENV
    torch.set_num_threads(1);_ENV=make_env()


def execute(job):
    case,skill,family,code,context=job
    ob,info=prepare(_ENV,case);contact_info(_ENV,info)
    own=Observer(case['initial_cube'],case['goal'],case[skill],skill,force_holding)
    joint=JointObserver(case);joint.previous_held=force_holding(info)
    joint.k.previous_holding=joint.w.previous_holding=joint.previous_held
    m,a,s=decode_style(torch.as_tensor(case['base_z']))
    expert_type=SkillExpert if family=='original' else DiverseSkillExpert
    # Only these arguments reach the training teacher; no other-skill region.
    expert=expert_type(case['initial_cube'],case['goal'],Style(int(m),float(a),float(s)),case[skill],[code],skill)
    stable=0;plan_visited=False
    for t in range(240):
        if t%8==0:plan_visited=own.visited
        action,phase=expert.action(info,own,visited_override=plan_visited if skill=='waypoint' else None)
        before=info['privileged/block_0_pos'].copy()
        result,path=step_traced(_ENV,action);ob,_,_,_,info=result;contact_info(_ENV,info)
        own.update(before,info['privileged/block_0_pos'],info,path)
        joint.update(before,info['privileged/block_0_pos'],info,path,_ENV)
        native,released=completion(_ENV,info);stable=stable+1 if released else 0
        if stable>=10:break
    result=joint.result(stable>=10,native)
    return {'context':context,'skill':skill,'family':family,'code':float(code),
            'own_success':bool(stable>=10 and own.satisfied()),'joint_success':bool(result['success']),
            'goal':bool(stable>=10),'keepout':bool(result['keepout_satisfied']),
            'waypoint':bool(result['waypoint_satisfied']),'steps':t+1}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=4);p.add_argument('--codes',type=int,default=8)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args();out=Path(args.out)
    assert not out.exists();out.mkdir(parents=True);jobs=[];sources={}
    for group in ['supported_easy','supported_conflict']:
        path=Path('cube_composition_artifacts/tasks_v2')/('val_'+group+'.npz');sources[group]=digest_file(path)
        for i,c in enumerate(load_cases(path,args.count)):
            for skill in ['keepout','waypoint']:
                for family in ['original','diverse']:
                    for j in range(args.codes):jobs.append((c,skill,family,NormalDist().inv_cdf((j+.5)/args.codes),(group,i)))
    (out/'config.json').write_text(json.dumps({'args':vars(args),'scope':__doc__,'task_hashes':sources,'source_sha256':digest_file(__file__)},indent=2)+'\n')
    started=time.time();records=[]
    with mp.get_context('spawn').Pool(args.workers,initializer=initialize) as pool:
        for row in pool.imap_unordered(execute,jobs,chunksize=1):
            records.append(row)
            if len(records)%32==0:
                progress={'finished':len(records),'total':len(jobs),'seconds':time.time()-started}
                (out/'status.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
    metrics=[]
    for group in sources:
        for skill in ['keepout','waypoint']:
            for family in ['original','diverse']:
                rows=[r for r in records if r['context'][0]==group and r['skill']==skill and r['family']==family]
                metrics.append({'group':group,'skill':skill,'family':family,'episodes':len(rows),
                    'own_success_count':sum(r['own_success'] for r in rows),
                    'joint_success_count':sum(r['joint_success'] for r in rows),
                    'contexts_with_any_joint_code':sum(any(r['joint_success'] for r in rows if r['context'][1]==i) for i in range(args.count)),
                    'contexts':args.count})
    result={'scope':__doc__,'metrics':metrics,'seconds':time.time()-started,'joint_witnesses_used_for_training':False}
    (out/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':len(records),'offline_capacity_only':True})+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
