"""Closed-loop Cube composition evaluation with full physical trace retention."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,hashlib,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from cube_addons.env import make_env,close_env
from .models import FrozenBank
from .search import SearchConfig
from .rollout import rollout


def load_cases(path,count=0):
    with np.load(path) as data:return [{k:data[k][i] for k in data.files} for i in range(min(count or len(data['seed']),len(data['seed'])))]


def summarize(rows,plans):
    result={'episodes':len(rows)}
    for key in ['success','joint_without_drop_collision_extension','released_success','native_success','keepout_satisfied','waypoint_satisfied','visited',
        'keepout_violation','table_violation','min_cube_clearance','ee_region_violation_diagnostic','grasped','has_lifted','dropped','grasp_losses',
        'robot_table_contact_in_transport','supported_transport_path_fraction','steps','final_goal_error','max_abs_latent','latent_total_variation','planning_seconds','planning_calls','stationary']:
        result[key]=float(np.mean([r[key] for r in rows]))
    selected=[r['action_disagreement_mean'] for r in rows if r['action_disagreement_mean'] is not None];result['action_disagreement_mean']=float(np.mean(selected)) if selected else None
    times=[p['seconds'] for episode in plans for p in episode];result.update(planning_seconds_p50=float(np.median(times)),planning_seconds_p95=float(np.quantile(times,.95)),
        latent_sign_flips_mean=np.mean([r['latent_sign_flips'] for r in rows],axis=0).tolist(),geometric_strategy_switches_mean=np.mean([r['geometric_strategy_switches'] for r in rows],axis=0).tolist())
    initial=[episode[0]['seconds'] for episode in plans]
    result.update(initial_planning_seconds_p50=float(np.median(initial)),initial_planning_seconds_p95=float(np.quantile(initial,.95)),
        planning_seconds_max=max(times),native_forecast_prefix_max_error_m=max([r['native_forecast_prefix_max_error_m'] for r in rows if r.get('native_forecast_prefix_max_error_m') is not None],default=None))
    counts={}
    for episode in plans:
        for plan in episode:
            for name,c in plan['calls'].items():
                row=counts.setdefault(name,{'batched_calls':0,'candidate_equivalents':0})
                for key,val in c.items():row[key]+=val
    result['generator_evaluations']=counts;result['backward_steps']=sum(p['backward_steps'] for episode in plans for p in episode)
    for key in ['native_controls','native_forks','native_worker_seconds','encoder_calls','encoder_seconds']:
        result[key]=sum(p.get(key,0) for episode in plans for p in episode)
    for key in ['prior','trajectory_disagreement','executable_grip_disagreement']:
        values=[p[key] for episode in plans for p in episode if p.get(key) is not None]
        result[key+'_mean']=float(np.mean(values)) if values else None
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=0)
    p.add_argument('--config',required=True);p.add_argument('--model-seed',type=int,default=42);p.add_argument('--seed',type=int,default=41);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(2);raw=json.loads(args.config if args.config.startswith('{') else Path(args.config).read_text());cfg=SearchConfig(**raw)
    bank=FrozenBank(args.model_seed,args.device);prior_path=Path(f'cube_composition_artifacts/priors/seed{args.model_seed}.json');prior_record=json.loads(prior_path.read_text()) if prior_path.exists() else None
    if prior_record is not None:assert prior_record['frozen_files']==bank.files
    priors=prior_record['skills'] if prior_record else None;cases=load_cases(args.tasks,args.count);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({'args':vars(args),'search':asdict(cfg),'frozen_files':bank.files,'task_sha256':hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
        'prior_sha256':hashlib.sha256(prior_path.read_bytes()).hexdigest() if prior_record else None,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'dynamics_sha256':hashlib.sha256(Path(f'cube_composition_artifacts/dynamics/seed{args.model_seed}.json').read_bytes()).hexdigest() if cfg.family in ['projected_waypoint','common'] else None,
        'device':args.device,'gpu':torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        'success':'Released settled original goal and both independently audited force-supported cube constraints, no distant drop or robot/table contact during transport.',
        'constraint_scope':'Keep-out covers complete cube bound; gripper-center clearance is a separate50ms diagnostic.'},indent=2)+'\n')
    rows=[];plans=[];env=make_env();started=time.perf_counter()
    try:
        for i,case in enumerate(cases):
            row,traj,log=rollout(env,bank,case,cfg,priors,args.seed+9973*i);row['context']=i;rows.append(row);plans.append(log)
            np.savez_compressed(out/f'rollout_{i:04d}.npz',**traj,**case)
            (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'planning.json').write_text(json.dumps(plans,indent=2)+'\n')
            print('EPISODE',i,'success',row['success'],'goal',row['released_success'],'keep',row['keepout_satisfied'],'waypoint',row['waypoint_satisfied'],flush=True)
        measured=summarize(rows,plans);measured['total_seconds']=time.perf_counter()-started;(out/'summary.json').write_text(json.dumps(measured,indent=2)+'\n')
        bank.assert_frozen();(out/'DONE.json').write_text(json.dumps({'complete':True,'frozen_unchanged':True,'episodes':len(rows)}));print(json.dumps(measured,indent=2),flush=True)
    finally:
        from .simulator import close_pool
        close_pool();close_env(env)


if __name__=='__main__':main()
