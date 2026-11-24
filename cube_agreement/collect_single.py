"""Actual independent single-skill demonstrations for shared manipulation control.

Only ONE active region is generated/recorded per episode. No joint scene, joint
witness or composed policy supplies training actions. Failed attempts remain.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time,multiprocessing as mp,math,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from cube_addons.env import make_env,reset_case,snapshot,features,completion,step_traced
from cube_addons.contacts import contact_info,force_holding
from cube_addons.geometry import Observer,axis_frame
from .manifolds import oracle,world_path,FREE
from .physical import PathTeacher

_ENV=None
_ACTOR=None
_SETTINGS=None


def init(settings=None):
    global _ENV,_ACTOR,_SETTINGS
    torch.set_num_threads(1);_ENV=make_env()
    _SETTINGS=settings or {}
    if _SETTINGS.get('actor'):
        from .action_model import load
        _ACTOR=load(_SETTINGS['actor'])


def collect(job):
    skill,seed,index,out,codes=job;env=_ENV;rng=np.random.default_rng(seed+19)
    ob,info=reset_case(env,seed);contact_info(env,info);start=info['privileged/block_0_pos'].copy()
    from cube.cube_gan.env import goal_position
    goal=goal_position(env);bz=np.random.default_rng(seed+53).standard_normal(2).astype(np.float32)
    amp=(.035+.04*math.erf(abs(float(bz[0]))/math.sqrt(2)))*(1 if bz[0]>=0 else -1)
    _,_,axis,perp,length=axis_frame(start,goal,start);state0=snapshot(env)
    progress=rng.uniform(.30,.70) if skill=='keepout' else .5
    lateral=amp+rng.uniform(-.01,.01) if skill=='keepout' else amp+np.sign(amp)*rng.uniform(.045,.075)
    height=.2+(rng.uniform(-.01,.01) if skill=='keepout' else rng.uniform(-.005,.005))
    radius=rng.uniform(.018,.030) if skill=='keepout' else rng.uniform(.020,.028)
    spec=np.r_[start[:2]+progress*(goal[:2]-start[:2])+lateral*perp,height,radius]
    geometry=np.zeros(10,np.float32);geometry[:2]=[length,amp];offset=2 if skill=='keepout' else 6;geometry[offset:offset+4]=[progress*length,lateral,height-.2,radius]
    case={'seed':np.int64(seed),'initial_cube':start,'goal':goal,'base_z':bz,'initial_state':state0,'waypoint':spec}
    rows=[]
    for code_index in range(codes):
        rejected=0
        while True:
            z=np.clip(rng.standard_normal(FREE+int(skill=='keepout'))*_SETTINGS.get('code_scale',.7),-3,3).astype(np.float32)
            if skill=='keepout':z[FREE]=rng.uniform(-math.pi,math.pi)
            with torch.no_grad():transverse=oracle(skill,torch.tensor(z)[None],torch.tensor(geometry)[None])[0].numpy()
            path=world_path(transverse,case)
            if ((path>=np.array([.255,-.30,.10]))&(path<=np.array([.595,.30,.32]))).all():break
            rejected+=1
            if rejected>=100:return {'skill':skill,'seed':seed,'context':index,'geometry_rejected_codes':rejected,'error':'No native workspace-feasible single-skill code in 100 draws','episodes':rows}
        from cube_addons.env import restore
        ob,info=restore(env,state0);contact_info(env,info);observer=Observer(start,goal,spec,skill,force_holding);observer.previous_holding=force_holding(info)
        teacher=PathTeacher(case,path);stable=0;previous=None;states=[];contexts=[];actions=[];executed=[];poses=[];forces=[];held=[];phases=[];steering=[];done=[];indices=[];micro=[];drop=False
        if _ACTOR is not None:
            from .action_model import path_context
            ref_spec=np.r_[(start[:2]+goal[:2])/2+(amp+np.sign(amp)*.060)*perp,.20,.022]
            reference=Observer(start,goal,ref_spec,'waypoint',force_holding);reference.previous_holding=force_holding(info)
            path_tensor=torch.tensor(path,device='cuda',dtype=torch.float32)[None]
        for t in range(240):
            contact_info(env,info);state=features(env,ob,info,start,previous);ctx=observer.features(info,t);cube=info['privileged/block_0_pos'].copy()
            runtime={'info':info,'observer':SimpleNamespace(w=observer)}
            chunk,_,meta=teacher.plan(state,None,bz,runtime);action=np.asarray(chunk[0],np.float32)
            states.append(state);contexts.append(ctx);poses.append(np.r_[info['proprio/effector_pos'],cube]);forces.append([info['addon/left_cube_normal_force'],info['addon/right_cube_normal_force']]);held.append(force_holding(info));phases.append(meta['teacher_phase']);indices.append(teacher.index);done.append(teacher.done)
            steering.append((teacher.style.speed*(path[teacher.index]-cube)*10).astype(np.float32))
            actual=action.copy()
            if _ACTOR is not None:
                reference.visited=teacher.done
                with torch.inference_mode():
                    ref=torch.tensor(reference.features(info,t),device='cuda')[None];pc=path_context(ref,torch.tensor(cube,device='cuda',dtype=torch.float32)[None],path_tensor,teacher.index)
                    pred=_ACTOR(torch.tensor(state,device='cuda')[None],torch.tensor(bz,device='cuda')[None],torch.tensor([[.8]],device='cuda'),pc)[0,0].cpu().numpy()
                mix=_SETTINGS.get('mix',.5);actual[:4]=mix*pred[:4]+(1-mix)*action[:4]
                actual[4]=(1 if pred[4]>=0 else -1) if rng.random()<mix else action[4]
            previous=info['proprio/effector_pos'].copy();result,traced=step_traced(env,actual);ob,_,_,_,info=result;contact_info(env,info);after=info['privileged/block_0_pos'];observer.update(cube,after,info,traced)
            if _ACTOR is not None:reference.update(cube,after,info,traced)
            if observer.has_lifted and np.linalg.norm(after[:2]-goal[:2])>.06 and (np.linalg.norm(info['proprio/effector_pos']-after)>.09 or after[2]<.065):drop=True
            native,released=completion(env,info);stable=stable+1 if released else 0;actions.append(action);executed.append(actual);micro.append(traced.astype(np.float32))
            if stable>=10:break
        poses.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']]);success=bool(stable>=10 and observer.satisfied() and not drop)
        folder=Path(out)/skill;folder.mkdir(parents=True,exist_ok=True);target=folder/f'episode_{index:04d}_{code_index:02d}.npz'
        np.savez_compressed(target,skill=np.array(skill),seed=seed,initial_state=state0,initial_cube=start,goal=goal,base_z=bz,spec=spec,new_z=z,path=path,geometry=geometry,
            states=np.asarray(states,np.float32),contexts=np.asarray(contexts,np.float32),actions=np.asarray(actions,np.float32),executed_actions=np.asarray(executed,np.float32),is_feedback=np.array(_ACTOR is not None),poses=np.asarray(poses,np.float32),forces=np.asarray(forces,np.float32),held=np.asarray(held),teacher_phase=np.asarray(phases),steering_inputs=np.asarray(steering,np.float32),path_done=np.asarray(done),path_indices=np.asarray(indices),micro_cube=np.asarray(micro,np.float32))
        rows.append({'source':str(target),'skill':skill,'seed':seed,'context':index,'code':code_index,'success':success,'released_success':bool(stable>=10),'skill_satisfied':observer.satisfied(),'dropped':drop,'steps':len(actions),'rejected_workspace_codes':rejected,'supported_path_fraction':observer.supported_fraction(),'minimum_keepout_gap_m':observer.minimum_clearance if skill=='keepout' else None})
    return {'skill':skill,'seed':seed,'context':index,'episodes':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--contexts',type=int,default=128);p.add_argument('--codes',type=int,default=2);p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=8100001);p.add_argument('--actor');p.add_argument('--mix',type=float,default=.5);p.add_argument('--code-scale',type=float,default=.7);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();rows=[];context_records=[]
    jobs=[(skill,args.seed+j*1000000+i,i,str(out),args.codes) for j,skill in enumerate(['keepout','waypoint']) for i in range(args.contexts)]
    (out/'config.json').write_text(json.dumps({'args':vars(args),'scope':'Independent single-skill physical teacher data. One active region per episode; no joint contexts or joint expert demonstrations. Workspace filtering is geometry-only and rejected code counts are retained.','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2)+'\n')
    with mp.get_context('spawn').Pool(args.workers,initializer=init,initargs=(vars(args),)) as pool:
        for record in pool.imap(collect,jobs,chunksize=1):
            context_records.append(record);rows.extend(record['episodes'])
            if len(context_records)%16==0:print('CONTEXTS',len(context_records),'EPISODES',len(rows),'SUCCESS',sum(r['success'] for r in rows),'seconds',time.perf_counter()-started,flush=True)
            (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'context_records.json').write_text(json.dumps(context_records,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'complete':True,'contexts':len(context_records),'episodes':len(rows),'seconds':time.perf_counter()-started}))


if __name__=='__main__':main()
