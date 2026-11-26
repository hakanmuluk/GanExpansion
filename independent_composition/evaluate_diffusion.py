"""Physical execution/monitoring outside independent score-policy inference."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .diffusion import load_denoiser,ScoreComposer
from .legacy import digest_file
from .train_diffusion import load_base


def summarize(rows):
    keys=['success','goal_success','keepout_satisfied','waypoint_satisfied',
          'keepout_full_success','waypoint_full_success','steps','planning_seconds']
    return {'episodes':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in keys}}


def make_policy(args,models,base,seed):
    return ScoreComposer(models,base,args.method,args.guidance,args.sampling_steps,seed,args.warm_time,args.execute,cold_time=args.cold_time)


def evaluate_tworoom(args,models,base,out,policy_factory=make_policy):
    from tworoom_gan.env import observation,transition
    from tworoom_composition.evaluate import load_cases
    from tworoom_composition.geometry import contexts,update
    case=load_cases(args.tasks,args.device,args.count)
    n=len(case['start']);device=args.device
    policy=policy_factory(args,models,base,args.seed)
    pos=case['start'].clone();alive=torch.ones(n,device=device,dtype=torch.bool)
    visited=torch.zeros_like(alive);violated=visited.clone();wall=visited.clone();early=visited.clone()
    length=torch.full((n,),240,device=device,dtype=torch.long)
    previous=torch.zeros_like(pos);positions=[pos];actions=[];plans=[]
    for t in range(240):
        if t%args.execute==0:
            c=contexts(pos,case,visited,torch.full((n,),float(t),device=device),previous)
            torch.cuda.synchronize() if str(device).startswith('cuda') else None
            started=time.perf_counter();chunk=policy.sample(observation(pos,case['goal']),c,case['base_z'])
            torch.cuda.synchronize() if str(device).startswith('cuda') else None
            plans.append({'step':t,'seconds':time.perf_counter()-started,'simulator_queries':0})
            plans[-1].update(getattr(policy,'last_plan',{}))
        action=torch.where(alive[:,None],chunk[:,t%args.execute],0.)
        if hasattr(policy,'observe_action'):policy.observe_action(action)
        nxt,_=transition(pos,action)
        v,k,w,e,g=update(pos,nxt,case,visited,violated,wall,early)
        visited=torch.where(alive,v,visited);violated=torch.where(alive,k,violated)
        wall=torch.where(alive,w,wall);early=torch.where(alive,e,early)
        length=torch.where(alive&g,t+1,length);pos=torch.where(alive[:,None],nxt,pos)
        alive=alive&~g;previous=action;positions.append(pos);actions.append(action)
        if not bool(alive.any()):break
    goal=(pos-case['goal']).norm(dim=-1)<16
    rows=[]
    for i in range(n):
        rows.append({'context':i,'success':bool(goal[i]&visited[i]&~violated[i]&~wall[i]&~early[i]),
                     'goal_success':bool(goal[i]),'keepout_satisfied':bool(~violated[i]),
                     'waypoint_satisfied':bool(visited[i]&~early[i]),'visited':bool(visited[i]),
                     'keepout_full_success':bool(goal[i]&~violated[i]&~wall[i]),
                     'waypoint_full_success':bool(goal[i]&visited[i]&~early[i]&~wall[i]),
                     'early_goal_entry':bool(early[i]),'wall_violation':bool(wall[i]),
                     'steps':int(length[i]),'planning_seconds':sum(p['seconds'] for p in plans if p['step']<int(length[i]))})
    np.savez_compressed(out/'trajectories.npz',positions=torch.stack(positions,1).cpu().numpy(),
                        actions=torch.stack(actions,1).cpu().numpy(),lengths=length.cpu().numpy(),
                        **{getattr(policy,'trace_noise_name','initial_diffusion_noise'):policy.noise.cpu().numpy()},
                        **(policy.trajectory_fields() if hasattr(policy,'trajectory_fields') else {}),
                        **{k:v.cpu().numpy() for k,v in case.items()})
    (out/'planning.json').write_text(json.dumps(plans,indent=2)+'\n')
    return rows


def evaluate_cube(args,models,base,out,policy_factory=make_policy):
    from cube_addons.env import make_env,close_env,features,completion,step_traced
    from cube_addons.contacts import contact_info,force_holding
    from .observation_replay import prepare
    from cube_composition.geometry import JointObserver
    from cube_composition.evaluate import load_cases
    env=make_env();rows=[];allplans=[];initial_observations={};paired_errors=[]
    try:
        for i,case in enumerate(load_cases(args.tasks,args.count)):
            policy=policy_factory(args,models,base,args.seed+i*9973)
            ob,info=prepare(env,case);contact_info(env,info)
            observer=JointObserver(case);observer.previous_held=force_holding(info)
            observer.k.previous_holding=observer.w.previous_holding=observer.previous_held
            previous=None;stable=0;states=[];contexts=[];poses=[];actions=[];raw=[];micro=[];plans=[]
            for t in range(240):
                state=features(env,ob,info,case['initial_cube'],previous);c=observer.features(info,t)
                if t==0 and getattr(args,'paired',False):
                    pair_id=int(case['paired_id'])
                    if pair_id in initial_observations:
                        error=float(np.max(np.abs(state-initial_observations[pair_id])))
                        paired_errors.append(error)
                        assert error<2e-6, ('Different initial observations within pair',pair_id,error)
                    else:initial_observations[pair_id]=state.copy()
                states.append(state);contexts.append(np.stack(c));poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']])
                if t%args.execute==0:
                    s=torch.as_tensor(state,dtype=torch.float32,device=args.device)[None]
                    ctx=tuple(torch.as_tensor(x,dtype=torch.float32,device=args.device)[None] for x in c)
                    bz=torch.as_tensor(case['base_z'],dtype=torch.float32,device=args.device)[None]
                    torch.cuda.synchronize() if args.device.startswith('cuda') else None
                    started=time.perf_counter();chunk=policy.sample(s,ctx,bz)[0].cpu().numpy()
                    plans.append({'step':t,'seconds':time.perf_counter()-started,'simulator_queries':0})
                    plans[-1].update(getattr(policy,'last_plan',{}))
                continuous=chunk[t%args.execute].copy();action=continuous.copy();action[4]=1. if action[4]>=0 else -1.
                if hasattr(policy,'observe_action'):policy.observe_action(action)
                before=info['privileged/block_0_pos'].copy();previous=info['proprio/effector_pos'].copy()
                result,path=step_traced(env,action);ob,_,_,_,info=result;contact_info(env,info)
                observer.update(before,info['privileged/block_0_pos'],info,path,env)
                native,released=completion(env,info);stable=stable+1 if released else 0
                actions.append(action);raw.append(continuous);micro.append(path.astype(np.float32))
                if stable>=10:break
            poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']])
            row=observer.result(stable>=10,native)
            row.update(context=i,goal_success=bool(stable>=10),steps=len(actions),planning_seconds=sum(p['seconds'] for p in plans))
            common=bool(row['goal_success'] and not row['dropped'] and not row['robot_table_contact_in_transport'])
            row['keepout_full_success']=bool(common and row['keepout_satisfied'])
            row['waypoint_full_success']=bool(common and row['waypoint_satisfied'])
            rows.append(row);allplans.append(plans)
            np.savez_compressed(out/f'rollout_{i:04d}.npz',states=np.asarray(states,np.float32),contexts=np.asarray(contexts,np.float32),
                                poses=np.asarray(poses,np.float32),actions=np.asarray(actions,np.float32),continuous_actions=np.asarray(raw,np.float32),
                                micro_cube=np.asarray(micro,np.float32),**{getattr(policy,'trace_noise_name','initial_diffusion_noise'):policy.noise.cpu().numpy()},
                                **(policy.trajectory_fields() if hasattr(policy,'trajectory_fields') else {}),**case)
            (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({'episode':i,'joint':row['success'],'goal':row['goal_success'],
                              'keepout':row['keepout_satisfied'],'waypoint':row['waypoint_satisfied']}),flush=True)
    finally:close_env(env)
    (out/'planning.json').write_text(json.dumps(allplans,indent=2)+'\n')
    (out/'observation_replay.json').write_text(json.dumps({
        'native_derived_contact_force_refreshed_at_reset':True,
        'paired_initial_observation_checks':len(paired_errors),
        'maximum_paired_initial_difference':max(paired_errors,default=0.),
        'extra_integration_steps':0})+'\n')
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--tasks',required=True);p.add_argument('--checkpoints',nargs=2,required=True);p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['keepout','waypoint','mean','sum','ratio'],default='ratio')
    p.add_argument('--count',type=int,default=16);p.add_argument('--guidance',type=float,default=1.)
    p.add_argument('--sampling-steps',type=int,default=20);p.add_argument('--warm-time',type=int,default=99)
    p.add_argument('--cold-time',type=int,default=99)
    p.add_argument('--execute',type=int,default=8);p.add_argument('--seed',type=int,default=41);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(2)
    if Path(args.tasks).name.startswith('test'):raise ValueError('Development validation only')
    if args.execute not in [1,2,4,8,16]:raise ValueError('Unsupported execution interval')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh evaluation directory required')
    out.mkdir(parents=True,exist_ok=True)
    models=[];records=[]
    base,source=load_base(args.env,args.device)
    for path,skill in zip(args.checkpoints,['keepout','waypoint']):
        model,saved=load_denoiser(path,args.device)
        assert saved['config']['args']['env']==args.env and saved['config']['args']['skill']==skill
        assert saved['config']['base']==source,'Base provenance differs'
        models.append(model);records.append({'path':path,'sha256':digest_file(path),'step':saved['step']})
    settings={'args':vars(args),'models':records,'base':source,'task_sha256':digest_file(args.tasks),
              'scope':'Current observations only. Independent denoisers, zero inference environment queries, no shared learned new network or analytic geometry cost.',
              'score_interpretation':'Heuristic noisy-score mean/product/density-ratio approximation; not an exact product-sampling guarantee.',
              'source_sha256':{p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
    source_copy=out/'source';source_copy.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,source_copy/path.name)
    started=time.time()
    rows=evaluate_tworoom(args,models,base,out) if args.env=='tworoom' else evaluate_cube(args,models,base,out)
    measured=summarize(rows);measured['total_seconds']=time.time()-started
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(measured,indent=2)+'\n')
    assert digest_file(source['path'])==source['sha256']
    assert all(p.grad is None and not p.requires_grad for m in [base,*models] for p in m.parameters())
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'inference_environment_queries':0,'models_frozen':True})+'\n')
    print(json.dumps(measured,indent=2),flush=True)


if __name__=='__main__':main()
