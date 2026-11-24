"""Dedicated inference timing on identical, natively replayed phase states."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,hashlib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from cube_addons.env import make_env,close_env,features,step_traced,completion
from cube_addons.contacts import contact_info,force_holding
from .scene import prepare
from .geometry import JointObserver
from .physics import capture,restore_branch
from .models import FrozenBank
from .search import LatentSearch,SearchConfig
from .simulator import forecast,close_pool


def main():
    p=argparse.ArgumentParser();p.add_argument('--trajectory',required=True);p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model-seed',type=int,default=42);p.add_argument('--repetitions',type=int,default=10);args=p.parse_args();torch.set_num_threads(2)
    with np.load(args.trajectory) as f:data={k:f[k] for k in f.files}
    case={k:data[k] for k in ['seed','kind','initial_state','initial_cube','goal','base_z','keepout','waypoint','force_contacts','feasible'] if k in data}
    mask=(data['contexts'][:,0,13]>.5)&(data['contexts'][:,0,14]>.5)&(data['contexts'][:,1,15]<.5)
    transport=np.flatnonzero(mask);after=np.flatnonzero((data['contexts'][:,1,15]>.5)&(data['contexts'][:,0,13]>.5))
    if not len(transport) or not len(after):raise ValueError('Choose a successful trace spanning all manipulation phases')
    targets={'reach':0,'held_transport':int(transport[len(transport)//2]),'after_waypoint':int(after[0])};env=make_env();bank=FrozenBank(args.model_seed,'cuda')
    priors=json.loads(Path(f'cube_composition_artifacts/priors/seed{args.model_seed}.json').read_text())['skills'];jobs=json.loads(Path(args.plan).read_text());snapshots={};records=[]
    try:
        ob,info=prepare(env,case);contact_info(env,info);observer=JointObserver(case);observer.previous_held=force_holding(info);previous=None;stable=0;max_error=0.
        for t in range(max(targets.values())+1):
            if t in targets.values():
                snapshots[t]={'physics':capture(env,ob,info),'observer':deepcopy(observer),'state':features(env,ob,info,case['initial_cube'],previous),
                    'contexts':observer.features(info,t),'previous_ee':deepcopy(previous),'stable':stable}
            before=info['privileged/block_0_pos'].copy();previous=info['proprio/effector_pos'].copy();result,path=step_traced(env,data['actions'][t]);ob,_,_,_,info=result;contact_info(env,info)
            observer.update(before,info['privileged/block_0_pos'],info,path,env);_,released=completion(env,info);stable=stable+1 if released else 0
            pose=np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']];max_error=max(max_error,float(np.max(np.abs(pose-data['poses'][t+1]))))
        if max_error>2e-6:raise AssertionError('Benchmark states do not reproduce the physical source')
        # Charge native worker startup separately, with an explicit real query.
        workers=next((j['search'].get('workers',4) for j in jobs if j['search'].get('objective','agreement')!='agreement'),4)
        t=targets['held_transport'];saved=snapshots[t];ob,info=restore_branch(env,saved['physics']);runtime={'env':env,'ob':ob,'info':info,'case':case,'time':t,
            'observer':deepcopy(saved['observer']),'previous_ee':saved['previous_ee'],'stable':saved['stable']}
        started=time.perf_counter();warmup=forecast(bank,saved['state'],saved['contexts'],case['base_z'],np.tile([.8,.8],(workers,1)).astype(np.float32),['waypoint']*workers,runtime,8,workers)
        warmup_seconds=time.perf_counter()-started
        for phase,t in targets.items():
            saved=snapshots[t]
            for job in jobs:
                cfg=SearchConfig(**job['search']);cfg.workers=workers;planner=LatentSearch(bank,cfg,priors,41);calls=[]
                for repetition in range(args.repetitions+1):
                    ob,info=restore_branch(env,saved['physics']);runtime={'env':env,'ob':ob,'info':info,'case':case,'time':t,'observer':deepcopy(saved['observer']),
                        'previous_ee':saved['previous_ee'],'stable':saved['stable'],'history':{'states':data['states'][:t],'contexts':data['contexts'][:t],'actions':data['actions'][:t]}}
                    _,z,meta=planner.plan(saved['state'],saved['contexts'],case['base_z'],runtime);calls.append({'repetition':repetition,'codes':z.tolist(),**meta})
                planner.close();steady=[c['seconds'] for c in calls[1:]];record={'phase':phase,'native_step':t,'method':job['name'],'search':asdict(cfg),
                    'initial_seconds':calls[0]['seconds'],'steady_p50_seconds':float(np.median(steady)),'steady_p95_seconds':float(np.quantile(steady,.95)),
                    'steady_exceeds_400ms_fraction':float(np.mean(np.asarray(steady)>.4)),'calls':calls}
                records.append(record);print(phase,job['name'],record['initial_seconds'],record['steady_p50_seconds'],flush=True)
        bank.assert_frozen();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps({'args':vars(args),'records':records,'source_trajectory_sha256':hashlib.sha256(Path(args.trajectory).read_bytes()).hexdigest(),
            'source_replay_max_error_m':max_error,'frozen_files':bank.files,'native_worker_warmup_seconds':warmup_seconds,'native_worker_warmup_controls':warmup['native_controls'],
            'measurement':'Dedicated sequential methods on identical native phase states. First controller query separate from repeated-state warm queries. Physical closed-loop latency remains in evaluation logs. Run only after concurrent experiment jobs finish.',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2)+'\n')
    finally:close_pool();close_env(env)


if __name__=='__main__':main()
