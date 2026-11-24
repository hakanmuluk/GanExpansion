"""Unfiltered held-out closed-loop evaluation, including physical latent recovery."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from cube.cube_gan.evaluate import load_models
from .env import make_env,close_env
from .geometry import MAX_STEPS
from .models import GatedGenerator,AdapterConfig,SkillEncoder
from .rollout import rollout


def load_contexts(path,count=None):
    with np.load(path) as data:
        return [{k:data[k][i] for k in data.files} for i in range(min(len(data['seed']),count or len(data['seed'])))]


def load_adapter(path,base='cube/artifacts/final/base_policy.pt',device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False)
    if hashlib.sha256(Path(base).read_bytes()).hexdigest()!=ck['base_sha256']:
        raise ValueError('Original policy SHA256 does not match adapter')
    model=GatedGenerator(base,AdapterConfig(**ck['adapter_config']),device)
    if set(ck['addon'])!=set(model.addon_state()):raise ValueError('Incomplete adapter state')
    missing,unexpected=model.load_state_dict(ck['addon'],strict=False)
    model.execute_steps=int(ck.get('execute_steps',ck.get('args',{}).get('execute',8)))
    if unexpected or any(not k.startswith(('base.','original_encoder.')) for k in missing):raise ValueError('Invalid adapter keys')
    ec=ck.get('encoder_config',{})
    encoder=SkillEncoder(model.base,model.config.new_z_dim,hidden=ec.get('hidden',128),context_dim=model.config.context_dim,geometry_only=ec.get('geometry_only',False)).to(device)
    encoder.load_state_dict(ck['encoder'])
    return model.eval(),encoder.eval(),ck


def encode_physical(encoder,traj,base_z):
    device=next(encoder.parameters()).device;t=len(traj['actions'])
    inputs=[torch.as_tensor(traj[k],device=device)[None] for k in ['states','actions','contexts']]
    with torch.no_grad():mu,_=encoder(*inputs,torch.as_tensor(base_z,device=device)[None],torch.tensor([t],device=device))
    return mu[0].cpu().numpy()


def summarize(rows):
    metrics={'episodes':len(rows),'contexts':len(set(r['context'] for r in rows))}
    for key in ['success','released_success','native_success','constraint_success','visited','violated','table_violation','has_lifted',
                'steps','action_change_l2','outer_action_change_l2','height_excursion','mode_consistent_fraction','final_goal_error']:
        metrics[key]=float(np.mean([r[key] for r in rows]))
    metrics['strategy_matches_code']=float(np.mean([r['strategy']==(1 if r['code']>0 else -1) for r in rows]))
    both=[];paired=[]
    for context in sorted(set(r['context'] for r in rows)):
        group=[r for r in rows if r['context']==context]
        strategies={r['strategy'] for r in group if r['success']}
        both.append(-1 in strategies and 1 in strategies)
        negative=[r['signed_height'] for r in group if r['code']<0]
        positive=[r['signed_height'] for r in group if r['code']>0]
        if negative and positive:paired.append(float(np.mean(positive)-np.mean(negative)))
    metrics['both_successful_strategies_per_context']=float(np.mean(both))
    metrics['paired_vertical_separation_m']=float(np.mean(paired)) if paired else None
    recovered=[r for r in rows if 'recovered_code' in r]
    if recovered:
        x=np.array([r['code'] for r in recovered]);y=np.array([r['recovered_code'] for r in recovered])
        metrics['physical_latent_rmse']=float(np.sqrt(np.mean((x-y)**2)))
        metrics['physical_latent_sign_accuracy']=float(np.mean(np.sign(x)==np.sign(y)))
        metrics['physical_latent_correlation']=float(np.corrcoef(x,y)[0,1]) if np.std(y)>1e-9 else 0.
    metrics['successful_original_route_matches_base_z']=float(np.mean([np.sign(r['original_route_lateral'])==np.sign(r['base_z'][0]) for r in rows if r['success']])) if any(r['success'] for r in rows) else 0.
    successful=[r for r in rows if r['success']]
    metrics['successful_strategy_matches_code']=float(np.mean([r['strategy']==np.sign(r['code']) for r in successful])) if successful else None
    if all('supported_transport_path_fraction' in r for r in rows):
        metrics['supported_transport_path_fraction']=float(np.mean([r['supported_transport_path_fraction'] for r in rows]))
        metrics['waypoint_entry_matches_code']=float(np.mean([r['waypoint_entry_strategy']==np.sign(r['code']) for r in successful])) if successful else None
        metrics['waypoint_backtracked_before_visit']=float(np.mean([r['waypoint_backtracked_before_visit'] for r in rows]))
    if all('pretransport_action_change_l2' in r for r in rows):
        metrics['pretransport_action_change_l2']=float(np.mean([r['pretransport_action_change_l2'] for r in rows]))
        metrics['strategy_sign_flips']=float(np.mean([r['strategy_sign_flips'] for r in rows]))
    paired_ok=[];paired_control=[];separations=[];gains=[];xs=[];ys=[]
    for context in sorted(set(r['context'] for r in rows)):
        group=[r for r in rows if r['context']==context]
        for negative in [r for r in group if r['code']<0]:
            positive=next((r for r in group if r['code']>0 and abs(r['code']+negative['code'])<1e-7),None)
            if positive is None:continue
            valid=negative['success'] and positive['success'];paired_ok.append(valid)
            paired_control.append(valid and negative['strategy']==-1 and positive['strategy']==1)
            if valid:separations.append(positive['signed_height']-negative['signed_height'])
        for sign in [-1,1]:
            valid=sorted([r for r in group if r['success'] and np.sign(r['code'])==sign],key=lambda r:abs(r['code']))
            if len(valid)<2:continue
            gains.append(valid[-1]['height_excursion']-valid[0]['height_excursion'])
            x=np.array([abs(r['code']) for r in valid]);y=np.array([r['height_excursion'] for r in valid])
            xs.extend(x-x.mean());ys.extend(y-y.mean())
    metrics['paired_both_successful_rate']=float(np.mean(paired_ok)) if paired_ok else None
    metrics['paired_successful_code_control_rate']=float(np.mean(paired_control)) if paired_control else None
    metrics['successful_paired_vertical_separation_m']=float(np.mean(separations)) if separations else None
    metrics['within_context_magnitude_excursion_gain_m']=float(np.mean(gains)) if gains else None
    metrics['within_context_magnitude_increases_excursion_rate']=float(np.mean(np.asarray(gains)>0)) if gains else None
    metrics['within_context_magnitude_excursion_correlation']=float(np.corrcoef(xs,ys)[0,1]) if len(xs)>2 and np.std(xs)>1e-9 and np.std(ys)>1e-9 else None
    return metrics


def evaluate(model,encoder,cases,skill,codes=4,seed=123,*,out=None,gate=1,expert=False,resample=False,force_contacts=None,execute_steps=None):
    """Fixed Gaussian codes per context; positive/negative pairs share magnitude."""
    started=time.time();rng=np.random.default_rng(seed);rows=[];trajectories=[];env=make_env()
    if out:out=Path(out);out.mkdir(parents=True,exist_ok=True)
    try:
        for i,c in enumerate(cases):
            values=rng.standard_normal((codes+1)//2)
            values=np.r_[-np.abs(values),np.abs(values)][:codes]
            for j,value in enumerate(values):
                row,traj=rollout(env,model,c,c['spec'],skill,[value],gate=gate,expert=expert,resample=resample,rng=rng,force_contacts=force_contacts,execute_steps=execute_steps)
                row.update(context=i,context_seed=int(c['seed']),code=float(value),code_index=j,base_z=c['base_z'].tolist())
                if encoder is not None and not expert and not resample:row['recovered_code']=float(encode_physical(encoder,traj,c['base_z'])[0])
                rows.append(row)
                if out:
                    np.savez_compressed(out/f'rollout_{i:04d}_{j:02d}.npz',**traj,base_z=c['base_z'],new_z=np.array([value],np.float32),
                                        initial=c['initial_cube'],goal=c['goal'],spec=c['spec'],initial_state=c['initial_state'])
                else:trajectories.append(traj)
            if (i+1)%8==0:print(json.dumps({'evaluated_contexts':i+1,'elapsed':time.time()-started}),flush=True)
        summary=summarize(rows);summary.update(elapsed_seconds=time.time()-started,geometry='2ms physics steps; conservative cube bounding sphere; swept adjacent samples',stop_rule='First released/settled original-goal placement, irrespective of constraint',
                                              force_contacts=all(r['force_contacts'] for r in rows),support_sampling='50ms control interval; both pad groups normal force >=0.05N; gripper/cube centers within 6.5cm',
                                              transport_gate=bool(getattr(getattr(model,'config',None),'transport_gate',False)),intervention_metric='Unsquared L2 between continuous normalized five-channel model outputs at executed chunk indices, on the same observed state')
        if out:
            (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary,rows
    finally:close_env(env)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--base',default='cube/artifacts/final/base_policy.pt')
    p.add_argument('--contexts',required=True);p.add_argument('--count',type=int);p.add_argument('--codes',type=int,default=8)
    p.add_argument('--skill',choices=['keepout','waypoint'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--device',default='cpu');p.add_argument('--off',action='store_true')
    p.add_argument('--expert',action='store_true');p.add_argument('--resample',action='store_true')
    p.add_argument('--transport-gate-override',action='store_true',help='Validation diagnostic: use observed transport gate without changing weights')
    p.add_argument('--force-contacts',action='store_true',default=None,help='Override legacy context banks with actual bilateral contact evidence')
    p.add_argument('--legacy-grip-proxy',action='store_false',dest='force_contacts',help='Reproduce the superseded observer for diagnostic records only')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--execute',type=int,choices=[1,2,4,8,16],help='Active-policy execution interval override; disabled adapters always retain eight')
    args=p.parse_args();torch.set_num_threads(1)
    if args.checkpoint:
        model,encoder,_=load_adapter(args.checkpoint,args.base,args.device)
        if args.transport_gate_override:model.config.transport_gate=True
        if model.config.no_new_latent or args.off:encoder=None
    else:model,encoder,_=load_models(args.base,device=args.device);encoder=None
    Path(args.out).mkdir(parents=True,exist_ok=True)
    (Path(args.out)/'evaluation_settings.json').write_text(json.dumps({'args':vars(args),'adapter_config':asdict(model.config) if hasattr(model,'base') else None},indent=2)+'\n')
    if args.workers>1:
        from .parallel_eval import evaluate_parallel
        summary,_=evaluate_parallel(args.checkpoint,args.base,load_contexts(args.contexts,args.count),args.skill,args.codes,args.seed,out=args.out,
                                    workers=args.workers,gate=0 if args.off else 1,expert=args.expert,resample=args.resample,
                                    force_contacts=args.force_contacts,transport_gate_override=args.transport_gate_override,execute_steps=args.execute)
    else:
        summary,_=evaluate(model,encoder,load_contexts(args.contexts,args.count),args.skill,args.codes,args.seed,out=args.out,gate=0 if args.off else 1,expert=args.expert,resample=args.resample,force_contacts=args.force_contacts,execute_steps=args.execute)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
