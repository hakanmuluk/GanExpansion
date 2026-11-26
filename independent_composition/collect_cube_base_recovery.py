"""Broader ORIGINAL-base data: fresh goal demos plus off-route recoveries.

Base style labels and the original goal-only expert are unchanged. Recovery
starts are native replay states from independent skill demonstrations; their
constraints are not passed to the base expert or included in base inputs.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from cube.cube_gan import data as original_data
from cube.cube_gan.env import TASK,features,completion,restore,goal_position,reset_context,close_env
from cube.cube_gan.expert import Style,select_action
from cube.cube_gan.style import decode_style
from cube_addons.env import reset_case,step_traced
from cube_addons.contacts import contact_info,force_holding
from cube_addons.data import atomic_npz
from .legacy import digest_file

_ENV=None


def initialize():
    global _ENV
    torch.set_num_threads(1);original_data.initialize_worker();_ENV=original_data._ENV


def save_episodes(path,episodes,extra):
    arrays={k:np.stack([e[k] for e in episodes]) for k in ['states','actions','poses','phase']}
    for field,name,dtype in [('length','lengths',np.int64),('mode','mode',np.int64),('amplitude','amplitude',np.float32),('speed','speed',np.float32)]:
        arrays[name]=np.array([e[field] for e in episodes],dtype)
    atomic_npz(path,**arrays,**extra)


def goal_worker(job):
    spec,folder=job;result=original_data.collect_context(spec);episodes=result['episodes'];seed=spec['seed']
    path=Path(folder)/str(seed)
    save_episodes(path.with_suffix('.npz'),episodes,{k:result[k] for k in ['initial_state','initial_cube','goal']})
    row={'seed':seed,'task_id':spec['task_id'],'paired_success':all(e['success'] for e in episodes),
         'successful':sum(e['success'] for e in episodes),'episodes':len(episodes)}
    path.with_suffix('.json').write_text(json.dumps(row)+'\n');return row


def recovery_worker(job):
    source,index,time_index,destination=job
    with np.load(source) as data:
        initial=data['initial_cube'];goal=data['goal'];seed=int(data['seed']);bz=data['base_z']
        prefix=data['actions'][index,:time_index];expected=data['poses'][index,time_index,3:]
        initial_state=data['initial_state']
    reset_case(_ENV,seed);ob,info=restore(_ENV,initial_state);previous=None
    for action in prefix:
        previous=info['proprio/effector_pos'].copy();result,_=step_traced(_ENV,action);ob,_,_,_,info=result
    error=float(np.max(np.abs(info['privileged/block_0_pos']-expected)));assert error<5e-6
    assert np.allclose(goal_position(_ENV),goal,atol=1e-10)
    contact_info(_ENV,info);initially_held=force_holding(info)
    m,a,s=decode_style(torch.as_tensor(bz));style=Style(int(m),float(a),float(s))
    states=np.zeros((TASK.max_steps+1,47),np.float32);actions=np.zeros((TASK.max_steps,5),np.float32)
    poses=np.zeros((TASK.max_steps+1,6),np.float32);phases=np.zeros(TASK.max_steps,np.int8)
    stable=0;total_path=0.;supported_path=0.;was_held=initially_held
    for t in range(TASK.max_steps):
        states[t]=features(_ENV,ob,info,initial,previous);poses[t]=np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']]
        action,phase=select_action(_ENV,info,initial,style);actions[t]=action;phases[t]=phase
        previous=info['proprio/effector_pos'].copy();result,path=step_traced(_ENV,action);ob,_,_,_,info=result;contact_info(_ENV,info)
        held=force_holding(info);delta=goal[:2]-initial[:2];p=((path[:,:2]-initial[:2])@delta)/max(float(delta@delta),1e-12)
        middle=(p[:-1]+p[1:])/2;length=float(np.linalg.norm(np.diff(path,axis=0),axis=1)[(middle>.2)&(middle<.8)].sum())
        total_path+=length
        if held and was_held:supported_path+=length
        was_held=held;native,released=completion(_ENV,info);stable=stable+1 if released else 0
        if stable>=TASK.stable_steps:break
    length=t+1;states[length:]=features(_ENV,ob,info,initial,previous)
    poses[length:]=np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']];phases[length:]=7
    support=supported_path/max(total_path,1e-9)
    row={'length':length,'mode':style.mode,'amplitude':style.amplitude,'speed':style.speed,
        'success':bool(stable>=TASK.stable_steps and initially_held and (total_path<.005 or support>=.8)),
        'released_success':bool(stable>=TASK.stable_steps),'supported_fraction':support,
        'native_prefix_replay_error_m':error,'source_code_index':index,'source_time':time_index,'source':str(source)}
    save_episodes(Path(destination).with_suffix('.npz'),[dict(states=states,actions=actions,poses=poses,phase=phases,**row)],{})
    Path(destination).with_suffix('.json').write_text(json.dumps(row)+'\n');return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--skill-raw',required=True)
    p.add_argument('--goal-contexts',type=int,default=1000);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--recoveries-per-context',type=int,default=2);args=p.parse_args();out=Path(args.out)
    assert not out.exists();out.mkdir(parents=True);(out/'goal_raw').mkdir();(out/'recovery_raw').mkdir()
    started=time.time();state={'pid':os.getpid(),'args':vars(args),'scope':__doc__,'stages':{}}
    def save():
        temp=out/'status.tmp.json';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'status.json')
    save();specs=original_data.context_specs(args.goal_contexts,1739,.5);goal_rows=[]
    with mp.get_context('spawn').Pool(args.workers,initializer=initialize) as pool:
        for row in pool.imap_unordered(goal_worker,[(s,str(out/'goal_raw')) for s in specs],chunksize=1):
            goal_rows.append(row);state['stages']['goal']={'complete':len(goal_rows),'total':len(specs),'successful_episodes':sum(r['successful'] for r in goal_rows)};save()
    raw=Path(args.skill_raw);state['stages']['recovery']={'status':'waiting_for_skill_raw_completion'};save()
    while not (raw/'DONE.json').exists():
        parent=raw.parent/'status.json'
        if parent.exists():
            status=json.loads(parent.read_text())
            if any(r['status']=='failed' for r in status['stages']):raise RuntimeError('Skill data pipeline failed; inspect its receipt')
            if not Path('/proc/'+str(status['pid'])+'/cmdline').exists():raise RuntimeError('Upstream data pipeline stopped before completion')
        time.sleep(10)
    jobs=[];rng=np.random.default_rng(8137)
    for source in sorted(raw.glob('*/raw/*.npz')):
        with np.load(source) as data:
            count=len(data['lengths']);order=rng.permutation(count)
            for index in order[:args.recoveries_per_context]:
                n=int(data['lengths'][index]);c=data['contexts'][index,:n];phase=data['phase'][index,:n]
                valid=np.flatnonzero((phase==4)&(c[:,13]>.5)&(c[:,14]>.5)&(c[:,10]>.2)&(c[:,10]<.7))
                if not len(valid):continue
                t=int(rng.choice(valid));destination=out/'recovery_raw'/str(len(jobs))
                jobs.append((str(source),int(index),t,str(destination)))
    recovery_rows=[]
    with mp.get_context('spawn').Pool(args.workers,initializer=initialize) as pool:
        for row in pool.imap_unordered(recovery_worker,jobs,chunksize=1):
            recovery_rows.append(row);state['stages']['recovery']={'complete':len(recovery_rows),'total':len(jobs),'successful_episodes':sum(r['success'] for r in recovery_rows)};save()
    selected=[]
    for path in sorted((out/'goal_raw').glob('*.json')):
        if json.loads(path.read_text())['paired_success']:selected.append(path.with_suffix('.npz'))
    selected.extend(path.with_suffix('.npz') for path in sorted((out/'recovery_raw').glob('*.json')) if json.loads(path.read_text())['success'])
    keys=['states','actions','poses','phase','lengths','mode','amplitude','speed'];parts={k:[] for k in keys}
    for path in selected:
        with np.load(path) as data:
            for k in keys:parts[k].append(data[k])
    dataset={k:np.concatenate(v) for k,v in parts.items()};atomic_npz(out/'train.npz',**dataset)
    valid=np.arange(TASK.max_steps)[None]<dataset['lengths'][:,None];x=dataset['states'][:,:-1][valid].astype(np.float64)
    atomic_npz(out/'normalization.npz',mean=x.mean(0).astype(np.float32),std=np.maximum(x.std(0),.05).astype(np.float32))
    original_data.initialize_worker()
    val=original_data.context_specs(64,2871,.5);contexts={k:[] for k in ['initial_state','initial_cube','goal','task_id','seed']}
    for spec in val:
        ob,info=reset_context(original_data._ENV,spec['seed'],spec['task_id'])
        from cube.cube_gan.env import snapshot
        values={'initial_state':snapshot(original_data._ENV),'initial_cube':info['privileged/block_0_pos'],
                'goal':goal_position(original_data._ENV),'task_id':spec['task_id'],'seed':spec['seed']}
        for k,v in values.items():contexts[k].append(v)
    atomic_npz(out/'val_contexts.npz',**{k:np.stack(v) for k,v in contexts.items()})
    close_env(original_data._ENV)
    report={'scope':__doc__,'goal_contexts':len(goal_rows),'goal_successes':sum(r['successful'] for r in goal_rows),
        'recovery_requested':len(jobs),'recovery_successes':sum(r['success'] for r in recovery_rows),
        'training_episodes':len(dataset['lengths']),'maximum_native_prefix_replay_error_m':max(r['native_prefix_replay_error_m'] for r in recovery_rows),
        'train_sha256':digest_file(out/'train.npz'),'source_sha256':digest_file(__file__),'seconds':time.time()-started}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(out/'DONE.json').write_text(json.dumps({'base_data_ready':True,'new_neural_architecture':False})+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
