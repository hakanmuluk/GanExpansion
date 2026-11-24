"""Parallel GPU queries, independent native scenes, pure action-chunk agreement.

Every candidate retains its own loss, Adam moments, seed and latent parameters.
Only neural matrix operations are batched. Each episode is independently replayed
through the sealed scalar native monitor before its trace/result is committed.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from cube_addons.env import make_env,close_env,features,completion
from cube_addons.contacts import contact_info,force_holding
from cube_composition.scene import prepare
from cube_composition.rollout import rollout
from cube_composition.audit import Replay
from cube_composition.evaluate import load_cases,summarize
from .actor_policy import Bank,ActorPolicy
from .action_model import path_context
from .batch_search import refine_batch
from .manifolds import geometry_cases,world_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=64);p.add_argument('--batch',type=int,default=8);p.add_argument('--model-seed',type=int,default=42);p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=80);p.add_argument('--action-steps',type=int,default=20);p.add_argument('--starts',type=int,default=12);p.add_argument('--action-starts',type=int,default=0);p.add_argument('--continuity-weight',type=float,default=.001);p.add_argument('--action-interval',type=int,default=1);p.add_argument('--actor');args=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;device='cuda';r=Path('cube_agreement_artifacts');actor=args.actor or str(r/('action_adapter_v3' if args.model_seed==42 else f'action_adapter_v3_s{args.model_seed}')/'best.pt');k=str(r/f'decoders/keepout_s{args.model_seed}/best.pt');w=str(r/f'decoders/waypoint_s{args.model_seed}/best.pt');bank=Bank(k,w,actor);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'config.json').exists(),'Use a fresh output folder'
    warm_starts=args.action_starts or args.starts
    config={'args':vars(args),'method':'Pure raw 16x5 action MSE, generic Gaussian prior and temporal latent continuity. No whole-route agreement term, task cost, joint geometry correction or expert control. Neural forwards/backwards are batched; episodes, losses, seeds and optimizer moments remain independent. All controls are the frozen StyleGAN action network.','frozen_files':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [k,w,actor,'cube/artifacts/final/base_policy.pt']},'task_sha256':hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');cases=load_cases(args.tasks,args.count);rows=[];all_plans=[];started=time.perf_counter();envs=[make_env() for _ in range(min(args.batch,len(cases)))];replay_env=make_env();max_replay_error=0.
    try:
        for first in range(0,len(cases),args.batch):
            batch_started=time.perf_counter()
            batch_cases=cases[first:first+args.batch];episodes=[]
            for i,c in enumerate(batch_cases):
                ob,info=prepare(envs[i],c);contact_info(envs[i],info);policy=ActorPolicy(bank,c,'local_action',seed=args.seed+9973*(first+i),execute=1)
                episodes.append({'env':envs[i],'case':c,'ob':ob,'info':info,'policy':policy,'previous':None,'stable':0,'actions':[],'continuous':[],'codes':[],'poses':[],'plans':[],'finished':False})
            g=geometry_cases(batch_cases,device)
            # Initial action-only optimization, matching scalar refine's seeds.
            states=torch.as_tensor(np.stack([features(e['env'],e['ob'],e['info'],e['case']['initial_cube']) for e in episodes]),device=device)
            refs=torch.as_tensor(np.stack([e['policy'].reference.features(e['info'],0) for e in episodes]),device=device);cubes=torch.as_tensor(np.stack([e['info']['privileged/block_0_pos'] for e in episodes]),device=device,dtype=torch.float32)
            torch.cuda.synchronize();t0=time.perf_counter();codes,_,paths,metrics=refine_batch(bank,g,batch_cases,states,refs,cubes,torch.zeros(len(episodes),device=device,dtype=torch.long),[e['policy'].seed for e in episodes],starts=args.starts,steps=args.steps,continuity_weight=args.continuity_weight);torch.cuda.synchronize();initial_seconds=time.perf_counter()-t0
            for i,e in enumerate(episodes):
                e['policy'].codes=codes[i].cpu().numpy();e['policy'].path=world_path(paths[i].detach().cpu().numpy(),e['case'])
            for t in range(240):
                active=[i for i,e in enumerate(episodes) if not e['finished']]
                if not active:break
                tick=time.perf_counter();states_np=[];refs_np=[];cube_np=[];search=[]
                for i in active:
                    e=episodes[i];pol=e['policy'];info=e['info'];cube=np.asarray(info['privileged/block_0_pos']);held=force_holding(info)
                    if pol.previous_cube is not None:pol.reference.update(pol.previous_cube,cube,info,np.stack([pol.previous_cube,cube]))
                    pol.reference.previous_holding=held;pol.reference.has_lifted|=bool(held and cube[2]>.10)
                    while pol.index<len(pol.path)-1 and np.linalg.norm(cube-pol.path[pol.index])<.010:pol.index+=1
                    if pol.index==len(pol.path)-1 and np.linalg.norm(cube-pol.path[-1])<.010:pol.done=True
                    pol.reference.visited=pol.done
                    states_np.append(features(e['env'],e['ob'],info,e['case']['initial_cube'],e['previous']));refs_np.append(pol.reference.features(info,t));cube_np.append(cube)
                    if t>0 and held and not pol.done and t>=pol.next_action_replan:search.append(i)
                    pol.previous_cube=cube.copy();e['poses'].append(np.r_[info['proprio/effector_pos'],cube])
                states=torch.as_tensor(np.stack(states_np),device=device);refs=torch.as_tensor(np.stack(refs_np),device=device);cubes=torch.as_tensor(np.stack(cube_np),device=device,dtype=torch.float32);search_metrics={}
                if search:
                    positions=[active.index(i) for i in search];pols=[episodes[i]['policy'] for i in search]
                    new,_,paths,ms=refine_batch(bank,g[search],[episodes[i]['case'] for i in search],states[positions],refs[positions],cubes[positions],torch.tensor([p.index for p in pols],device=device),[p.seed+1009*p.replans for p in pols],torch.as_tensor(np.stack([p.codes for p in pols]),device=device),warm_starts,args.action_steps,args.continuity_weight)
                    for j,i in enumerate(search):
                        pol=episodes[i]['policy'];pol.codes=new[j].cpu().numpy();pol.path=world_path(paths[j].detach().cpu().numpy(),episodes[i]['case']);pol.replans+=1;pol.next_action_replan=t+args.action_interval;search_metrics[i]=ms[j]
                paths=torch.as_tensor(np.stack([episodes[i]['policy'].path for i in active]),device=device,dtype=torch.float32);indices=torch.tensor([episodes[i]['policy'].index for i in active],device=device)
                with torch.inference_mode():
                    ctx=path_context(refs,cubes,paths,indices);bz=torch.as_tensor(np.stack([episodes[i]['case']['base_z'] for i in active]),device=device);pred=bank.actor(states,bz,torch.full((len(active),1),.8,device=device),ctx)[:,0].cpu().numpy()
                torch.cuda.synchronize();seconds=time.perf_counter()-tick
                for j,i in enumerate(active):
                    e=episodes[i];pol=e['policy'];e['continuous'].append(pred[j].copy());action=pred[j].astype(np.float32).copy();action[4]=1 if action[4]>=0 else -1;e['actions'].append(action);e['codes'].append(pol.codes.copy())
                    metric=metrics[i] if t==0 else search_metrics.get(i);n=args.steps if t==0 else (args.action_steps if i in search else 0)
                    candidates=args.starts if t==0 else warm_starts
                    calls={'StyleGAN_actor':{'batched_calls':1+(n+1 if metric else 0),'candidate_equivalents':1+(2*candidates*(n+1) if metric else 0)}}
                    e['plans'].append({'step':t,'seconds':seconds+(initial_seconds if t==0 else 0),'iterations':n,'disagreement':metric['action_disagreement'] if metric else None,'backward_steps':n,'calls':calls,'query_batch_size':len(episodes) if t==0 else (len(search) if metric else 0),'controller_batch_size':len(active),'path_index':pol.index,'path_done':pol.done,'codes':pol.codes.tolist(),**(metric or {})})
                    e['previous']=e['info']['proprio/effector_pos'].copy();e['ob'],_,_,_,e['info']=e['env'].step(action);contact_info(e['env'],e['info']);_,released=completion(e['env'],e['info']);e['stable']=e['stable']+1 if released else 0;e['finished']=e['stable']>=10 or t==239
            batch_online_seconds=time.perf_counter()-batch_started
            for i,e in enumerate(episodes):
                e['poses'].append(np.r_[e['info']['proprio/effector_pos'],e['info']['privileged/block_0_pos']]);actions=np.asarray(e['actions'],np.float32);row,trace,_=rollout(replay_env,None,e['case'],planner_override=Replay(actions),execute=1,steps=len(actions))
                error=float(np.max(np.abs(trace['poses']-np.asarray(e['poses'],np.float32))));assert error<2e-6,error;max_replay_error=max(max_replay_error,error)
                trace['codes']=np.asarray(e['codes'],np.float32);trace['continuous_actions']=np.asarray(e['continuous'],np.float32);held=trace['held'];poses=trace['poses'];near=np.linalg.norm(poses[1:-1,3:5]-e['case']['goal'][:2],axis=1)<.06;transitions=np.flatnonzero(held[:-1]&~held[1:]&near);height=float(poses[transitions[-1]+1,5]) if row['released_success'] and len(transitions) else None;zz=trace['codes'];plans=e['plans'];ds=[p['disagreement'] for p in plans if p['disagreement'] is not None]
                row.update(context=first+i,last_supported_release_height_m=height,controlled_release_success=bool(row['success'] and height is not None and height<=.065),latent_initial=zz[0].tolist(),latent_final=zz[-1].tolist(),max_abs_latent=float(abs(zz).max()),latent_sign_flips=(np.sign(zz[8::8])!=np.sign(zz[:-8:8])).sum(0).tolist() if len(zz)>8 else [0]*21,latent_total_variation=float(np.linalg.norm(np.diff(zz[::8],axis=0),axis=-1).sum()),action_disagreement_initial=ds[0] if ds else None,action_disagreement_mean=float(np.mean(ds)) if ds else None,planning_seconds=sum(p['seconds'] for p in plans),planning_calls=len(plans),total_seconds=batch_online_seconds,batch_online_wall_seconds=batch_online_seconds,batch_to_scalar_native_replay_error_m=error)
                np.savez_compressed(out/f'rollout_{first+i:04d}.npz',**e['case'],**trace,reference_path=e['policy'].path);rows.append(row);all_plans.append(plans);print('EPISODE',first+i,'joint',row['success'],'controlled',row['controlled_release_success'],flush=True)
            bank.assert_frozen();(out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'planning.json').write_text(json.dumps(all_plans,indent=2)+'\n')
        summary=summarize(rows,all_plans);summary.update(controlled_release_success=float(np.mean([r['controlled_release_success'] for r in rows])),elapsed_seconds=time.perf_counter()-started,maximum_batch_to_scalar_native_replay_error_m=max_replay_error,latency_scope='Batch query wall latency, not divided by scene count. Interactive single-scene latency is benchmarked separately.')
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'DONE.json').write_text(json.dumps({'complete':True,'episodes':len(rows),'weights_frozen':True}));print(json.dumps(summary),flush=True)
    finally:
        for env in [*envs,replay_env]:close_env(env)


if __name__=='__main__':main()
