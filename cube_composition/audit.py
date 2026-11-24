"""Disjointness, feasibility-certificate and independent native replay audits."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,hashlib,multiprocessing as mp
from pathlib import Path
import numpy as np
from .evaluate import load_cases
from .rollout import rollout
from cube_addons.env import make_env

_ENV=None


def init_worker():
    global _ENV
    _ENV=make_env()


class Replay:
    def __init__(self,actions):self.actions=actions
    def plan(self,state,ctx,bz,runtime):
        a=self.actions[runtime['time']]
        return np.repeat(a[None],16,axis=0),np.zeros(2,np.float32),{'seconds':0.,'calls':{},'backward_steps':0,'iterations':0,'disagreement':None}


def replay(job):
    case,path=job
    with np.load(path) as f:source={k:f[k] for k in f.files}
    row,traj,_=rollout(_ENV,None,case,planner_override=Replay(source['actions']),execute=1,steps=len(source['actions']))
    error=float(np.max(np.abs(traj['poses']-source['poses'])))
    if not row['success'] or error>2e-6:raise AssertionError((str(path),row,error))
    return {'source':str(path),'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'seed':int(case['seed']),'success':row['success'],'max_pose_error_m':error,'metrics':row}


def key(case):
    return b''.join(np.asarray(case[k],np.float32).tobytes() for k in ['initial_cube','goal','base_z'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--replay-per-bank',type=int,default=3);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=Path(args.tasks)
    if not (root/'DONE.json').exists():raise ValueError('Audit only complete task sets')
    manifest=json.loads((root/'manifest.json').read_text());old_keys=set();old_seeds=set();old_sources={}
    for path in sorted(Path('cube_addon_artifacts/data_force').glob('*/*_contexts.npz')):
        cases=load_cases(path);old_keys.update(map(key,cases));old_seeds.update(int(c['seed']) for c in cases);old_sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    seen=set();seen_seeds=set();banks={};jobs=[]
    for name,record in manifest['banks'].items():
        path=root/(name+'.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'];cases=load_cases(path)
        keys=set(map(key,cases));seeds={int(c['seed']) for c in cases};assert len(keys)==len(cases) and len(seeds)==len(cases)
        assert not(keys&seen or keys&old_keys or seeds&seen_seeds or seeds&old_seeds);seen.update(keys);seen_seeds.update(seeds)
        eligible=[x for x in record['candidate_results'] if x.get('geometric_eligible')];accepted=[x for x in eligible if x.get('accepted')]
        assert [x['seed'] for x in accepted[:len(cases)]]==[int(c['seed']) for c in cases]
        infeasible='infeasible' in name
        if infeasible:
            for c in cases:assert np.linalg.norm(c['waypoint'][:3]-c['keepout'][:3])+c['waypoint'][3] < c['keepout'][3]+np.sqrt(3)*.02
        else:
            for x in accepted[:len(cases)]:assert any(a['row']['success'] for a in x['attempts'])
            for index in np.linspace(0,len(cases)-1,min(args.replay_per_bank,len(cases)),dtype=int):jobs.append((cases[index],root/'witnesses'/name/f'{index:04d}.npz'))
        banks[name]={'count':len(cases),'sha256':record['sha256'],'eligible_native_candidates':len(eligible),'certified_candidates':len(accepted),
            'native_certification_rate':len(accepted)/len(eligible),'analytic_infeasibility':infeasible,
            'selected_first_n_certificates_only':True,'disjoint_from_old_contexts_and_other_new_banks':True,
            'base_z_second_coordinate_range':[min(float(c['base_z'][1]) for c in cases),max(float(c['base_z'][1]) for c in cases)]}
    with mp.get_context('spawn').Pool(args.workers,initializer=init_worker) as pool:replays=pool.map(replay,jobs)
    (root/'audit.json').write_text(json.dumps({'args':vars(args),'banks':banks,'old_context_sources':old_sources,
        'independent_native_replays':replays,'joint_training_episodes':0,'witnesses_used_by_optimizer':False},indent=2)+'\n')
    print(json.dumps({'banks':banks,'native_replays':len(replays),'max_pose_error_m':max([r['max_pose_error_m'] for r in replays],default=0)},indent=2))


if __name__=='__main__':main()
