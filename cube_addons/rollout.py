"""Receding-horizon execution and metrics from actual MuJoCo episodes."""
import numpy as np
import torch
from cube.cube_gan.style import decode_style
from cube.cube_gan.expert import Style
from .env import restore,features,completion,step_traced,snapshot,reset_case,reset_context
from .geometry import Observer,MAX_STEPS,axis_frame
from .expert import SkillExpert
from .contacts import contact_info,force_holding


@torch.no_grad()
def rollout(env, model, case, spec, skill, new_z, *, gate=1,
            expert=False, resample=False, rng=None, max_steps=MAX_STEPS,
            save_snapshots=False,force_contacts=None,replay_actions=None,execute_steps=None):
    """Stop at the first released/settled placement, even if constraint failed."""
    # A full integration snapshot does not contain Python-side task caches.
    # Initialize those through the native reset before restoring paired physics.
    context_key=(int(case['seed']),int(case['task_id']) if 'task_id' in case else None)
    if getattr(env,'_addon_context_key',None)!=context_key:
        if 'task_id' in case:reset_context(env,int(case['seed']),int(case['task_id']))
        else:reset_case(env,int(case['seed']))
        env._addon_context_key=context_key
    ob,info=restore(env,case['initial_state'])
    initial,goal,bz=case['initial_cube'],case['goal'],case['base_z']
    force_contacts=bool(case.get('force_contacts',True)) if force_contacts is None else force_contacts
    if force_contacts:contact_info(env,info)
    observer=Observer(initial,goal,spec,skill,force_holding if force_contacts else None)
    code=np.asarray(new_z,dtype=np.float32).reshape(-1)
    rng=rng or np.random.default_rng(0)
    execute_steps=8 if not gate else (execute_steps or getattr(model,'execute_steps',8))
    if expert and replay_actions is None:
        m,a,s=decode_style(torch.as_tensor(bz))
        teacher=SkillExpert(initial,goal,Style(int(m),float(a),float(s)),spec,code,skill)
    elif replay_actions is None:
        device=next(model.parameters()).device
        tz=torch.as_tensor(bz,device=device)[None]
    previous=None;stable=0;states=[];contexts=[];actions=[];poses=[];phases=[];codes=[];deltas=[];frames=[];snapshots=[];plan_visited=False
    for t in range(min(max_steps,len(replay_actions)) if replay_actions is not None else max_steps):
        state=features(env,ob,info,initial,previous)
        ctx=observer.features(info,t)
        states.append(state);contexts.append(ctx)
        poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']])
        if save_snapshots and t%execute_steps==0:
            snapshots.append(snapshot(env));frames.append(t)
        if replay_actions is not None:
            action=np.asarray(replay_actions[t],np.float32);phase=-2
        elif expert:
            if bool(case.get('macro_waypoint',False)):
                if t%int(case.get('planning_interval',8))==0:plan_visited=observer.visited
                action,phase=teacher.action(info,observer,visited_override=plan_visited)
            else:action,phase=teacher.action(info,observer)
        else:
            if t%execute_steps==0:
                if resample and t>0:code=rng.standard_normal(code.shape).astype(np.float32)
                ts=torch.as_tensor(state,device=device)[None]
                tc=torch.as_tensor(ctx,device=device)[None]
                tn=torch.as_tensor(code,device=device)[None]
                if hasattr(model,'base'):
                    chunk=model(ts,tz,tn,tc,gate=gate)[0].cpu().numpy()
                    base_chunk=model(ts,tz,gate=0)[0].cpu().numpy()
                    difference=np.linalg.norm(chunk-base_chunk,axis=-1)
                else:
                    chunk=model(ts,tz)[0].cpu().numpy();difference=np.zeros(len(chunk))
            action=chunk[t%execute_steps].copy();action[4]=1. if action[4]>=0 else -1.;phase=-1
            deltas.append(difference[t%execute_steps])
        codes.append(code.copy());actions.append(action);phases.append(phase)
        before=info['privileged/block_0_pos'].copy();previous=info['proprio/effector_pos'].copy()
        result,path=step_traced(env,action);ob,_,_,_,info=result
        if force_contacts:contact_info(env,info)
        observer.update(before,info['privileged/block_0_pos'],info,path)
        native,released=completion(env,info);stable=stable+1 if released else 0
        if stable>=10:break
    poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']])
    trajectory={
        'states':np.asarray(states,np.float32),'contexts':np.asarray(contexts,np.float32),
        'actions':np.asarray(actions,np.float32),'poses':np.asarray(poses,np.float32),
        'phase':np.asarray(phases,np.int16),'codes':np.asarray(codes,np.float32),
        'action_delta':np.asarray(deltas,np.float32),
        'snapshots':np.asarray(snapshots),'snapshot_frames':np.asarray(frames,np.int16),
    }
    cubes=trajectory['poses'][:-1,3:];p=np.array([axis_frame(initial,goal,c)[0] for c in cubes])
    # Strategy classifiers use the executed object path, not a latent or E.
    mode_mask=(p>.2)&(p<.8) if skill=='keepout' else ((p>.15)&(p<.4))
    mode_mask &= trajectory['contexts'][:,13]>.5
    height=cubes[:,2]-.20
    values=height[mode_mask]
    signed_height=float(values.mean()) if len(values) else 0.
    strategy=1 if signed_height>.010 else (-1 if signed_height<-.010 else 0)
    lateral=np.array([axis_frame(initial,goal,c)[1] for c in cubes])
    middle=(p>.2)&(p<.8)&(cubes[:,2]>.07)
    local=((p<.12)|(p>.88))
    before_transport=(trajectory['contexts'][:,14]<.5)&(p<.12)
    delta=trajectory['action_delta']
    significant=np.sign(values[np.abs(values)>.010])
    visit=np.flatnonzero(trajectory['contexts'][:,15]>.5)
    entry_height=None;entry_strategy=0;backtracked=False
    if skill=='waypoint' and len(visit):
        first=int(visit[0]);window=height[max(0,first-6):first]
        entry_height=float(window.mean()) if len(window) else 0.
        entry_strategy=1 if entry_height>.010 else -1 if entry_height<-.010 else 0
        backtracked=bool(p[:first].max()>.70) if first else False
    row={
        'released_success':bool(stable>=10),'native_success':bool(native),
        'constraint_success':observer.satisfied(),
        'success':bool(stable>=10 and observer.satisfied()),
        'visited':observer.visited,'violated':observer.violated,'table_violation':observer.table_violation,
        'has_lifted':observer.has_lifted,'held_samples':observer.held_transport_samples,
        'min_clearance':observer.minimum_clearance if skill=='keepout' else None,
        'steps':t+1,'strategy':strategy,'signed_height':signed_height,
        'height_excursion':float(np.max(np.abs(values))) if len(values) else 0.,
        'original_route_lateral':float(lateral[middle].mean()) if middle.any() else 0.,
        'carry_steps':int(middle.sum()),
        'action_change_l2':float(delta.mean()) if len(delta) else 0.,
        'outer_action_change_l2':float(delta[local].mean()) if len(delta) and local.any() else 0.,
        'pretransport_action_change_l2':float(delta[before_transport].mean()) if len(delta) and before_transport.any() else 0.,
        'strategy_sign_flips':int(np.sum(significant[1:]!=significant[:-1])) if len(significant)>1 else 0,
        'mode_consistent_fraction':float(np.mean(np.sign(values)==strategy)) if len(values) and strategy else 0.,
        'final_goal_error':float(np.linalg.norm(cubes[-1]-goal)),
        'force_contacts':bool(force_contacts),
        'supported_transport_path_fraction':float(observer.supported_fraction()),
        'waypoint_entry_height':entry_height,'waypoint_entry_strategy':entry_strategy,
        'waypoint_backtracked_before_visit':backtracked,
        'execute_steps':1 if expert or replay_actions is not None else int(execute_steps),
    }
    return row,trajectory
