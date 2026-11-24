"""Read-only native replay sensitivity: conservative sphere versus actual box.

The primary frozen success definition is unchanged. Secondary box/sphere gaps
use actual free-joint orientation and native collision-geom dimensions at2ms
samples. No model parameter, collision geom, action or state is edited in replay.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,hashlib,multiprocessing as mp,time,math
from pathlib import Path
from copy import deepcopy
import numpy as np
import mujoco
from cube_addons.env import make_env,close_env,completion
from cube_addons.contacts import contact_info,force_holding
from .scene import prepare
from .geometry import JointObserver

_ENV=None


def init_worker():
    global _ENV
    _ENV=make_env()


def replay(job):
    source,original,candidate=job;source=Path(source);started=time.perf_counter();env=_ENV
    with np.load(source) as f:
        keys=['seed','kind','initial_state','initial_cube','goal','base_z','keepout','waypoint','force_contacts','feasible']
        case={k:f[k] for k in keys if k in f.files}
        actions=f['actions'] if candidate is None else f['actions'][candidate]
        poses=f['poses'] if candidate is None else f['paths'][candidate]
    assert np.isfinite(actions).all() and np.isfinite(poses).all(),'Nonfinite recorded native trace'
    ob,info=prepare(env,case);contact_info(env,info);observer=JointObserver(case);observer.previous_held=force_holding(info)
    observer.k.previous_holding=observer.w.previous_holding=observer.previous_held
    geom=env._model.geom('object_0');body_id=int(env._model.geom_bodyid[geom.id]);assert int(env._model.body_parentid[body_id])==0 and int(env._model.geom_type[geom.id])==6
    halfsize=geom.size.copy();offset=geom.pos.copy();local_rotation=np.empty(9);mujoco.mju_quat2Mat(local_rotation,geom.quat);local_rotation=local_rotation.reshape(3,3)
    qpos=env._data.joint('object_joint_0').qpos;rotation=np.empty(9);minimum=float('inf');first_violation=None;clock=0;points=0
    mujoco.mju_quat2Mat(rotation,qpos[3:7]);initial_rotation=rotation.reshape(3,3)
    np.testing.assert_allclose(initial_rotation@local_rotation,env._data.geom_xmat[geom.id].reshape(3,3),atol=1e-8,rtol=0)
    np.testing.assert_allclose(qpos[:3]+initial_rotation@offset,env._data.geom_xpos[geom.id],atol=1e-8,rtol=0)
    def sample():
        nonlocal minimum,first_violation,points
        mujoco.mju_quat2Mat(rotation,qpos[3:7]);r=rotation.reshape(3,3);center=qpos[:3]+r@offset
        local=(r@local_rotation).T@(case['keepout'][:3]-center);distance=np.abs(local)-halfsize
        gap=float(np.linalg.norm(np.maximum(distance,0))+min(float(distance.max()),0)-case['keepout'][3])
        assert math.isfinite(gap),'Nonfinite native box geometry'
        if gap<minimum:minimum=gap
        if gap < -1e-7 and first_violation is None:first_violation=clock*.002
        points+=1
        return qpos[:3].copy()
    stable=0;max_error=0.;sample()
    for t,action in enumerate(actions):
        action=np.asarray(action,np.float32).copy();action[4]=1. if action[4]>=0 else -1.
        before=info['privileged/block_0_pos'].copy();path=[qpos[:3].copy()];native_step=mujoco.mj_step
        def traced(model,data,nstep=1):
            nonlocal clock
            if data is not env._data:return native_step(model,data,nstep=nstep)
            for _ in range(nstep):
                native_step(model,data,nstep=1);clock+=1;path.append(sample())
        mujoco.mj_step=traced
        try:ob,_,_,_,info=env.step(action)
        finally:mujoco.mj_step=native_step
        contact_info(env,info);observer.update(before,info['privileged/block_0_pos'],info,np.asarray(path),env)
        native,released=completion(env,info);stable=stable+1 if released else 0
        pose=np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']];max_error=max(max_error,float(np.max(np.abs(pose-poses[t+1]))))
        if candidate is not None and stable>=10:break
    primary=observer.result(stable>=10,native)
    for key in ['success','released_success','native_success','keepout_violation','waypoint_satisfied','visited','dropped','robot_table_contact_in_transport']:
        if candidate is not None and key=='native_success':continue # Original capability file used a placeholder here, not in its success criterion.
        assert primary[key]==original[key],(str(source),key,primary[key],original[key])
    assert max_error<2e-6,(str(source),max_error)
    assert minimum>=primary['min_cube_clearance']-2e-6,(minimum,primary['min_cube_clearance'])
    if not primary['keepout_violation']:assert minimum>=-1e-7
    secondary=deepcopy(observer);secondary.k.violated=minimum < -1e-7;secondary.k.minimum_clearance=minimum;alternate=secondary.result(stable>=10,native)
    assert alternate['success'] or not primary['success']
    return {'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seed':int(case['seed']),'kind':str(case['kind']),
        'source_id':str(source) if candidate is None else str(source)+'#'+str(candidate),'candidate_index':candidate,'family':original.get('family'),
        'context':original['context'],'primary_joint_success':primary['success'],'primary_keepout_violation':primary['keepout_violation'],
        'audited_native_success':bool(native),'audited_released_success':bool(stable>=10),
        'secondary_sampled_box_joint_success':alternate['success'],'secondary_sampled_box_keepout_satisfied':alternate['keepout_satisfied'],
        'sampled_box_sphere_violation':minimum < -1e-7,'minimum_sampled_box_sphere_gap_m':minimum,'primary_minimum_sphere_bound_gap_m':primary['min_cube_clearance'],
        'first_box_violation_seconds':first_violation,'native_replay_max_pose_error_m':max_error,'native_control_steps':t+1,'geometry_samples':points,'seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',nargs='+',default=[]);p.add_argument('--capability-root',nargs='+',default=[]);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--count-per-evaluation',type=int,default=0);args=p.parse_args();root=Path(args.out);root.mkdir(parents=True,exist_ok=True);groups=[]
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for base in map(Path,args.root):
        for done in sorted(base.rglob('DONE.json')):
            folder=done.parent
            if not (folder/'records.json').exists() or not (folder/'config.json').exists():continue
            config=json.loads((folder/'config.json').read_text());rows=json.loads((folder/'records.json').read_text());rows=rows[:args.count_per_evaluation or len(rows)]
            groups.append((folder,config,rows))
    jobs=[(folder/f"rollout_{row['context']:04d}.npz",row,None) for folder,_,rows in groups for row in rows];capability=[]
    for folder in map(Path,args.capability_root):
        assert (folder/'DONE.json').exists();config=json.loads((folder/'config.json').read_text());parents=json.loads((folder/'records.json').read_text());parents=parents[:args.count_per_evaluation or len(parents)]
        capability.append((folder,config,parents))
        for parent in parents:
            for index,row in enumerate(parent['candidates']):jobs.append((folder/f"paths_{parent['context']:04d}.npz",{**row,'context':parent['context'],'seed':parent['seed']},index))
    jobs.sort(key=lambda job:(job[1]['seed'],str(job[0]),job[2] if job[2] is not None else -1));out=[];start=time.perf_counter()
    if not jobs:raise ValueError('No completed evaluation traces')
    descriptor=hashlib.sha256()
    for source,row,candidate in jobs:descriptor.update(json.dumps([str(source),row,candidate],sort_keys=True).encode())
    checkpoint={'source_sha256':source_sha,'jobs_sha256':descriptor.hexdigest(),'mujoco':mujoco.__version__,'count':len(jobs)}
    config_path=root/'checkpoint_config.json';events=root/'checkpoint_events.jsonl'
    if config_path.exists():
        assert json.loads(config_path.read_text())==checkpoint,'Audit source/input changed: use a fresh output directory'
        if events.exists():out=[json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    else:config_path.write_text(json.dumps(checkpoint,indent=2)+'\n')
    completed={v['source_id'] for v in out};checked={}
    for record in out:
        path=record['source']
        if path not in checked:checked[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert checked[path]==record['source_sha256'],'Changed replay input'
    remaining=[job for job in jobs if (str(job[0]) if job[2] is None else str(job[0])+'#'+str(job[2])) not in completed]
    print('RESUME',len(out),'of',len(jobs),flush=True)
    with events.open('a',buffering=1) as stream,mp.get_context('spawn').Pool(args.workers,initializer=init_worker) as pool:
        for record in pool.imap(replay,remaining,chunksize=1):
            out.append(record);stream.write(json.dumps(record)+'\n')
            if len(out)%64==0:print('AUDITED',len(out),'of',len(jobs),'seconds',time.perf_counter()-start,flush=True)
    by_source={v['source_id']:v for v in out};summary=[];capability_summary=[]
    for folder,config,rows in groups:
        records=[by_source[str(folder/f"rollout_{row['context']:04d}.npz")] for row in rows]
        summary.append({'source_evaluation':str(folder),'method':folder.name,'model_seed':config['args']['model_seed'],'search_seed':config['args']['seed'],'kind':records[0]['kind'],'episodes':len(records),
            'primary_joint_success':float(np.mean([x['primary_joint_success'] for x in records])),
            'secondary_sampled_box_joint_success':float(np.mean([x['secondary_sampled_box_joint_success'] for x in records])),
            'primary_keepout_violation':float(np.mean([x['primary_keepout_violation'] for x in records])),
            'sampled_box_sphere_violation':float(np.mean([x['sampled_box_sphere_violation'] for x in records]))})
    for folder,config,parents in capability:
        cases=[]
        for parent in parents:
            records=[by_source[str(folder/f"paths_{parent['context']:04d}.npz")+'#'+str(i)] for i in range(len(parent['candidates']))]
            cases.append({'context':parent['context'],'seed':parent['seed'],
                'any_primary_success_by_family':{family:any(r['primary_joint_success'] for r in records if r['family']==family) for family in ['keepout','waypoint','sum']},
                'any_secondary_box_success_by_family':{family:any(r['secondary_sampled_box_joint_success'] for r in records if r['family']==family) for family in ['keepout','waypoint','sum']}})
        capability_summary.append({'source':str(folder),'model_seed':config['args']['seed'],'contexts':len(cases),'cases':cases})
    (root/'records.json').write_text(json.dumps(out,indent=2)+'\n');(root/'summary.json').write_text(json.dumps({'args':vars(args),'runs':summary,'capability':capability_summary,'episodes':len(out),
        'native_controls':sum(v['native_control_steps'] for v in out),'geometry_samples':sum(v['geometry_samples'] for v in out),'max_replay_pose_error_m':max(v['native_replay_max_pose_error_m'] for v in out),
        'scope':'Primary frozen task definition unchanged. Secondary exact oriented-box versus sphere geometry at2ms sample times, preserving all original grasp/support/drop/goal criteria; no continuous-time safety guarantee.',
        'source_sha256':source_sha},indent=2)+'\n');(root/'DONE.json').write_text(json.dumps({'complete':True,'episodes':len(out)}));print('DONE',len(out),time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
