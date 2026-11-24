"""Actual native closed-loop inference with frozen learned manifold agreement."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from cube_addons.env import make_env,close_env
from cube_composition.evaluate import load_cases,summarize
from cube_composition.rollout import rollout
from .policy import Bank,NeuralPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=16)
    p.add_argument('--keepout',default='cube_agreement_artifacts/decoders/keepout_s42/best.pt');p.add_argument('--waypoint',default='cube_agreement_artifacts/decoders/waypoint_s42/best.pt');p.add_argument('--steering',default='cube_agreement_artifacts/shared_steering_v1.pt');p.add_argument('--backbone',default='cube_addon_artifacts/final/waypoint_s42/adapter.pt')
    p.add_argument('--method',choices=['agreement','single_keepout','single_waypoint'],default='agreement');p.add_argument('--family',choices=['keepout','waypoint'],default='waypoint');p.add_argument('--objective',choices=['path','delta'],default='path');p.add_argument('--steps',type=int,default=120);p.add_argument('--starts',type=int,default=12);p.add_argument('--seed',type=int,default=42);p.add_argument('--tolerance',type=float,default=.010);p.add_argument('--threshold',type=float,default=.15);p.add_argument('--backbone-execute',type=int,choices=[1,2,4,8,16],default=1);args=p.parse_args();torch.set_num_threads(2)
    bank=Bank(args.keepout,args.waypoint,args.steering,args.backbone);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=[];planning=[];env=make_env();started=time.perf_counter()
    config={'args':vars(args),'frozen_files':{v:hashlib.sha256(Path(v).read_bytes()).hexdigest() for v in [args.keepout,args.waypoint,args.steering,args.backbone,'cube/artifacts/final/base_policy.pt']},'task_sha256':hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'mechanism':'Latent-only frozen neural trajectory agreement, generic path cursor, learned single-skill-data steering and frozen neural grasp/release backbone. No hand-coded expert, joint task cost, candidate physics or geometric correction at inference.','execute_steps':1}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    try:
        for i,case in enumerate(load_cases(args.tasks,args.count)):
            controller=NeuralPolicy(bank,case,args.method,args.family,args.steps,args.starts,args.seed+9973*i,args.objective,args.tolerance,args.threshold,args.backbone_execute)
            row,trace,plans=rollout(env,None,case,planner_override=controller,execute=1)
            held=trace['held'];poses=trace['poses'];near=np.linalg.norm(poses[1:-1,3:5]-case['goal'][:2],axis=1)<.06
            transitions=np.flatnonzero(held[:-1]&~held[1:]&near)
            release_height=float(poses[transitions[-1]+1,5]) if row['released_success'] and len(transitions) else None
            row.update(context=i,last_supported_release_height_m=release_height,controlled_release_success=bool(row['success'] and release_height is not None and release_height<=.065))
            np.savez_compressed(out/f'rollout_{i:04d}.npz',**case,**trace,reference_path=controller.path)
            rows.append(row);planning.append(plans);(out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'planning.json').write_text(json.dumps(planning,indent=2)+'\n')
            print('EPISODE',i,'joint',row['success'],'controlled',row['controlled_release_success'],'goal',row['released_success'],'K',row['keepout_satisfied'],'W',row['waypoint_satisfied'],'release',release_height,flush=True)
        result=summarize(rows,planning);result['controlled_release_success']=float(np.mean([r['controlled_release_success'] for r in rows]));result['elapsed_seconds']=time.perf_counter()-started
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');bank.assert_frozen();(out/'DONE.json').write_text(json.dumps({'complete':True,'episodes':len(rows),'weights_frozen':True}));print(json.dumps(result,indent=2),flush=True)
    finally:close_env(env)


if __name__=='__main__':main()
