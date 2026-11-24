"""Actual native MuJoCo composition execution, with 2ms swept object geometry."""
import time
import numpy as np
from cube_addons.env import features,completion,step_traced
from cube_addons.contacts import contact_info,force_holding
from cube_addons.geometry import axis_frame
from .scene import prepare
from .geometry import JointObserver


def rollout(env,bank,case,config=None,priors=None,seed=41,planner_override=None,execute=8,steps=240):
    if planner_override is None:
        from .search import LatentSearch
        planner=LatentSearch(bank,config,priors,seed)
    else:planner=planner_override
    ob,info=prepare(env,case);contact_info(env,info);observer=JointObserver(case);observer.previous_held=force_holding(info);observer.k.previous_holding=observer.w.previous_holding=observer.previous_held
    previous=None;stable=0;states=[];contexts=[];actions=[];raw=[];poses=[];codes=[];forces=[];held=[];micro=[];plans=[];started=time.perf_counter()
    for t in range(steps):
        state=features(env,ob,info,case['initial_cube'],previous);ctx=observer.features(info,t)
        states.append(state);contexts.append(np.stack(ctx));poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']]);forces.append([info['addon/left_cube_normal_force'],info['addon/right_cube_normal_force']]);held.append(force_holding(info))
        if t%execute==0:
            runtime={'env':env,'observer':observer,'ob':ob,'info':info,'time':t,'previous_ee':previous,'case':case,'stable':stable}
            if config is not None and config.initialization=='encoder_prefix':
                runtime['history']={'states':np.asarray(states[:len(actions)],np.float32),'contexts':np.asarray(contexts[:len(actions)],np.float32),'actions':np.asarray(actions,np.float32)}
            chunk,z,meta=planner.plan(state,ctx,case['base_z'],runtime);plans.append({'step':t,**meta,'codes':np.asarray(z).tolist()})
        continuous=np.asarray(chunk[t%execute],np.float32).copy();action=continuous.copy();action[4]=1. if action[4]>=0 else -1.
        before=info['privileged/block_0_pos'].copy();previous=info['proprio/effector_pos'].copy();result,path=step_traced(env,action);ob,_,_,_,info=result;contact_info(env,info)
        observer.update(before,info['privileged/block_0_pos'],info,path,env);native,released=completion(env,info);stable=stable+1 if released else 0
        actions.append(action);raw.append(continuous);codes.append(np.asarray(z));micro.append(path.astype(np.float32))
        if stable>=10:break
    poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']]);poses=np.asarray(poses,np.float32);zz=np.asarray(codes,np.float32)
    progress=np.array([axis_frame(case['initial_cube'],case['goal'],p)[0] for p in poses[:-1,3:]])
    heights=poses[:-1,5]-.2;carried=np.asarray(held);strategies=[];switches=[]
    for j,mask in enumerate([(progress>.2)&(progress<.8)&carried,(progress>.15)&(progress<.4)&carried]):
        values=heights[mask];mean=values.mean() if len(values) else 0.;strategies.append(int(np.sign(mean)) if abs(mean)>.01 else 0)
        signs=np.sign(values[np.abs(values)>.01]);switches.append(int((signs[1:]!=signs[:-1]).sum()))
    row=observer.result(stable>=10,native);agreements=[p['disagreement'] for p in plans if p.get('disagreement') is not None]
    forecast_errors=[]
    for plan in plans:
        if 'predicted_prefix' not in plan:continue
        predicted=np.asarray(plan['predicted_prefix']);actual=poses[plan['step']:plan['step']+len(predicted)]
        error=float(np.max(np.abs(actual-predicted[:len(actual)])))
        plan['actual_prefix_max_error_m']=error;forecast_errors.append(error)
    row.update(seed=int(case['seed']),kind=str(case['kind']),steps=len(actions),final_goal_error=float(np.linalg.norm(poses[-1,3:]-case['goal'])),
        action_disagreement_mean=float(np.mean(agreements)) if agreements else None,action_disagreement_initial=agreements[0] if agreements else None,
        latent_initial=zz[0].tolist(),latent_final=zz[-1].tolist(),max_abs_latent=float(abs(zz).max()),
        latent_sign_flips=(np.sign(zz[8::8])!=np.sign(zz[:-8:8])).sum(0).tolist() if len(zz)>8 else [0,0],
        latent_total_variation=float(np.linalg.norm(np.diff(zz[::8],axis=0),axis=-1).sum()),
        keepout_strategy=strategies[0],waypoint_strategy=strategies[1],geometric_strategy_switches=switches,
        planning_seconds=sum(p['seconds'] for p in plans),planning_calls=len(plans),total_seconds=time.perf_counter()-started,
        stationary=bool(np.linalg.norm(np.diff(poses[-min(33,len(poses)):,:3],axis=0),axis=-1).mean()<.0001 and stable<10),
        native_forecast_prefix_max_error_m=max(forecast_errors) if forecast_errors else None)
    traj={'states':np.asarray(states,np.float32),'contexts':np.asarray(contexts,np.float32),'actions':np.asarray(actions,np.float32),
        'continuous_actions':np.asarray(raw,np.float32),'poses':poses,'codes':zz,'forces':np.asarray(forces,np.float32),'held':carried,'micro_cube':np.asarray(micro,np.float32)}
    if bank is not None:bank.assert_frozen()
    if hasattr(planner,'close'):planner.close()
    return row,traj,plans
