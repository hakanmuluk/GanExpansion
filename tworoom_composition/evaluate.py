"""Reproducible real-dynamics closed-loop composition experiments."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .models import FrozenBank
from .search import SearchConfig
from .rollout import rollout


def load_cases(path,device,count=0):
    with np.load(path) as data:
        return {k:torch.as_tensor(data[k][:count or None],device=device) for k in data.files if data[k].dtype.kind not in ['U','S','O']}


def summary(rows,info):
    result={'episodes':len(rows)}
    for key in ['success','permissive_success','goal_success','visited','keepout_violation','strict_wall_violation','early_goal_entry',
                'native_collision','steps','disagreement_initial','disagreement_mean','disagreement_max','max_abs_latent','original_door_preserved','planning_steps','stalled',
                'terminal_stall','mean_action_norm','final_goal_distance']:
        result[key]=float(np.mean([r[key] for r in rows]))
    times=np.array([r['seconds'] for r in info['planning']]);result.update(planning_batch_seconds_mean=float(times.mean()),
                planning_batch_seconds_p50=float(np.median(times)),planning_batch_seconds_p95=float(np.quantile(times,.95)),
                total_seconds=info['total_seconds'],batch_size=info['batch_size'],planning_iterations=sum(r['iterations'] for r in info['planning']))
    calls={}
    for row in info['planning']:
        for key,value in row['calls'].items():
            c=calls.setdefault(key,{'batched_calls':0,'candidate_equivalents':0})
            for k,v in value.items():c[k]+=v
    result['generator_evaluations']=calls
    result['latent_sign_flips_mean']=np.mean([r['latent_sign_flips'] for r in rows],axis=0).tolist()
    for key in ['latent_total_variation','latent_prior_cost_mean']+[s+k for s in ['keepout','waypoint'] for k in ['_strategy_switch','_strategy_matches_code','_amount']]:
        result[key]=float(np.mean([r[key] for r in rows]))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=0)
    p.add_argument('--keepout',default='addon_artifacts/final/keepout.pt');p.add_argument('--waypoint',default='addon_artifacts/final/waypoint.pt')
    p.add_argument('--config',help='JSON SearchConfig, or a file containing it');p.add_argument('--seed',type=int,default=41);p.add_argument('--device',default='cuda')
    p.add_argument('--priors',default='composition_artifacts/priors/seed42.json')
    args=p.parse_args();torch.set_num_threads(2)
    config_data={}
    if args.config:config_data=json.loads(Path(args.config).read_text() if not args.config.startswith('{') else args.config)
    config=SearchConfig(**config_data);bank=FrozenBank(args.keepout,args.waypoint,device=args.device)
    prior_path=Path(args.priors);prior_record=json.loads(prior_path.read_text()) if prior_path.exists() else None
    if prior_record is not None and prior_record['frozen_files']!=bank.files:raise ValueError('Latent statistics were fitted to different frozen checkpoints')
    priors=prior_record['skills'] if prior_record is not None else None
    case=load_cases(args.tasks,args.device,args.count)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({'args':vars(args),'search':asdict(config),'frozen_files':bank.files,
        'task_sha256':hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'termination':'Stop at first native goal completion; any earlier swept goal-region entry before waypoint invalidates strict success.',
        'strict_geometry':'Swept circular body against wall rectangles and forbidden disk; observed swept waypoint visit; no projections.'},indent=2)+'\n')
    rows,traj,info=rollout(bank,case,config,priors,args.seed)
    with np.load(args.tasks) as data:
        for row,name in zip(rows,data['kind']):row['kind']=str(name)
    np.savez_compressed(out/'trajectories.npz',**traj,**{k:v.cpu().numpy() for k,v in case.items()})
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'planning.json').write_text(json.dumps(info,indent=2)+'\n')
    measured=summary(rows,info);(out/'summary.json').write_text(json.dumps(measured,indent=2)+'\n');print(json.dumps(measured,indent=2),flush=True)
    (out/'DONE.json').write_text(json.dumps({'frozen_unchanged':True,'episodes':len(rows)})+'\n')


if __name__=='__main__':main()
