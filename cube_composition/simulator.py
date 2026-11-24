"""Persistent native CPU workers for exact, model-in-the-loop candidate futures."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
from copy import deepcopy
import multiprocessing as mp
import time
import numpy as np
from cube_addons.env import make_env,features,completion,step_traced
from cube_addons.contacts import contact_info
from .scene import prepare
from .physics import capture,restore_branch

_ENV=None
_POOL=None
_WORKERS=None


def init_worker():
    global _ENV
    _ENV=make_env()


def advance(item):
    packet,actions=item;env=_ENV;started=time.perf_counter();prepare(env,packet['case']);ob,info=restore_branch(env,packet['physics'])
    observer=deepcopy(packet['observer']);previous=packet['previous_ee'];stable=packet['stable'];poses=[];executed=0
    for i,continuous in enumerate(actions):
        if stable<10 and packet['time']+i<240:
            action=np.asarray(continuous,np.float32).copy();action[4]=1. if action[4]>=0 else -1.
            before=info['privileged/block_0_pos'].copy();previous=info['proprio/effector_pos'].copy()
            result,path=step_traced(env,action);ob,_,_,_,info=result;contact_info(env,info);observer.update(before,info['privileged/block_0_pos'],info,path,env)
            _,released=completion(env,info);stable=stable+1 if released else 0;executed+=1
        poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']])
    clock=min(240,packet['time']+len(actions));state=features(env,ob,info,packet['case']['initial_cube'],previous);ctx=observer.features(info,clock)
    return {'case':packet['case'],'physics':capture(env,ob,info),'observer':observer,'previous_ee':previous,'stable':stable,'time':clock,
        'state':state,'contexts':np.stack(ctx),'poses':np.asarray(poses),'native_controls':executed,'worker_seconds':time.perf_counter()-started}


def get_pool(workers=4):
    global _POOL,_WORKERS
    if _POOL is None:
        _POOL=mp.get_context('spawn').Pool(workers,initializer=init_worker);_WORKERS=workers
    elif workers!=_WORKERS:raise ValueError('Use one worker count per evaluator process')
    return _POOL


def close_pool():
    global _POOL,_WORKERS
    if _POOL is not None:_POOL.close();_POOL.join();_POOL=None;_WORKERS=None


def initial_packet(state,ctx,runtime):
    return {'case':runtime['case'],'physics':capture(runtime['env'],runtime['ob'],runtime['info']),
        'observer':deepcopy(runtime['observer']),'previous_ee':deepcopy(runtime['previous_ee']),'stable':runtime['stable'],'time':runtime['time'],
        'state':np.asarray(state),'contexts':np.stack(ctx)}


def task_cost(packet,poses):
    o=packet['observer'];case=packet['case'];cube=poses[-1,3:];ee=poses[-1,:3];held=o.previous_held
    if not o.w.visited:cost=float(np.sum((cube-case['waypoint'][:3])**2)/.15**2)
    else:cost=float(np.sum((cube-case['goal'])**2)/.2**2)-3.
    if not o.k.has_lifted:
        cost+=float(np.sum((ee-cube)**2)/.05**2)+max(0.,.17-cube[2])**2/.1**2+2*float(not held)
    if o.k.transport_path_length>.005:cost+=10*max(0.,.8-o.k.supported_fraction())**2
    cost+=100*max(0.,.003-o.k.minimum_clearance)**2/.01**2
    cost+=1000*float(o.k.violated)+1000*float(o.drop or o.robot_table or o.k.table_violation)
    if packet['stable']>=10:cost+=(-5. if o.w.visited else 100.)
    return cost


def forecast(bank,state,ctx,bz,codes,families,runtime,horizon=16,workers=4,action_sequences=None):
    """Predict native closed-loop fixed-code futures; only eight controls per NN call."""
    import torch
    if horizon<=0 or horizon%8:raise ValueError('Native horizon must be a positive multiple of8')
    n=len(codes);device=next(bank.parameters()).device;pool=get_pool(workers);initial=initial_packet(state,ctx,runtime)
    packets=[deepcopy(initial) for _ in range(n)];start_pose=np.r_[runtime['info']['proprio/effector_pos'],runtime['info']['privileged/block_0_pos']]
    paths=[np.asarray([start_pose]) for _ in range(n)];first=None;all_actions=[];controls=forks=0;worker_seconds=0.
    with torch.inference_mode():
        for t in range(0,horizon,8):
            if action_sequences is None:
                states=torch.as_tensor(np.stack([p['state'] for p in packets]),device=device,dtype=torch.float32)
                contexts=torch.as_tensor(np.stack([p['contexts'] for p in packets]),device=device,dtype=torch.float32)
                base=torch.as_tensor(bz,device=device,dtype=torch.float32)[None].expand(n,-1);z=torch.as_tensor(codes,device=device,dtype=torch.float32)
                chunk=np.zeros((n,16,5),np.float32)
                for family in sorted(set(families)):
                    ids=np.array([i for i,f in enumerate(families) if f==family]);index=torch.as_tensor(ids,device=device)
                    chunk[ids]=bank.execute(states[index],base[index],z[index],(contexts[index,0],contexts[index,1]),family).cpu().numpy()
                if first is None:first=chunk.copy()
                actions=chunk[:,:8]
            else:
                actions=np.asarray(action_sequences[:,t:t+8],np.float32)
                if first is None:first=np.asarray(action_sequences[:,:16],np.float32).copy()
            all_actions.append(actions);packets=pool.map(advance,list(zip(packets,actions)),chunksize=1);forks+=n
            for i,packet in enumerate(packets):
                paths[i]=np.concatenate([paths[i],packet['poses']]);controls+=packet['native_controls'];worker_seconds+=packet['worker_seconds']
    paths=np.stack(paths);scores=np.array([task_cost(p,pp) for p,pp in zip(packets,paths)])
    return {'scores':scores,'paths':paths,'first_chunks':first,'actions':np.concatenate(all_actions,axis=1),'packets':packets,
        'native_controls':controls,'native_forks':forks,'native_worker_seconds':worker_seconds}
