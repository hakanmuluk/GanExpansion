"""New joint contexts selected only by geometry and independent native witnesses."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,hashlib,time,multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from cube_addons.env import make_env,close_env,snapshot
from cube.cube_gan.env import goal_position
from cube.cube_gan.style import decode_style
from cube.cube_gan.expert import Style
from cube_addons.geometry import axis_frame
from .scene import reset_scene
from .planner import joint_path,WitnessPolicy
from .rollout import rollout

_ENV=None


def init_worker():
    global _ENV
    torch.set_num_threads(1);_ENV=make_env()


def candidate(item):
    seed,kind=item;rng=np.random.default_rng(seed+79);radius=rng.uniform(.020,.025);offset=rng.uniform(.050,.065);wp_radius=rng.uniform(.020,.025)
    if kind=='supported_easy' and not(radius<=.022 and offset>=.061):return {'seed':seed,'geometric_eligible':False}
    if kind=='supported_conflict' and not(radius>=.024 and offset<=.054):return {'seed':seed,'geometric_eligible':False}
    env=_ENV;_,info=reset_scene(env,seed,kind=='coupled_ood');initial=info['privileged/block_0_pos'].copy();goal=goal_position(env)
    bz=np.random.default_rng(seed+53).standard_normal(2).astype(np.float32);mode,amplitude,speed=decode_style(torch.tensor(bz));style=Style(int(mode),float(amplitude),float(speed))
    _,_,axis,perp,length=axis_frame(initial,goal,initial);side=-1 if style.mode==0 else 1
    center=(initial[:2]+goal[:2])/2+side*style.amplitude*perp;wp=np.r_[center+side*offset*perp,.20,wp_radius]
    keep=np.r_[center,.20,radius]
    if kind=='coupled_ood':keep[:2]+=axis*length*(.15 if seed%2 else -.15)
    if kind=='infeasible':wp[:3]=keep[:3]
    case={'seed':np.int64(seed),'kind':np.array(kind),'initial_state':snapshot(env),'initial_cube':initial,'goal':goal,'base_z':bz,
        'keepout':keep,'waypoint':wp,'force_contacts':np.bool_(True),'feasible':np.bool_(kind!='infeasible')}
    if kind=='infeasible':return {'seed':seed,'geometric_eligible':True,'accepted':True,'case':case,'witness':None,'certificate':'Waypoint sphere lies entirely in forbidden sphere inflated by complete cube bound.'}
    attempts=[]
    for preferred in [1,-1]:
        graph_started=time.perf_counter();nodes=joint_path(case,preferred);graph_seconds=time.perf_counter()-graph_started
        if nodes is None:continue
        start=time.perf_counter();row,traj,_=rollout(env,None,case,planner_override=WitnessPolicy(case,style,nodes),execute=1)
        attempts.append({'preferred':preferred,'row':row,'seconds':time.perf_counter()-start,'geometric_planning_seconds':graph_seconds})
        if row['success']:
            return {'seed':seed,'geometric_eligible':True,'accepted':True,'case':case,'witness':{**traj,'nodes':nodes},'attempts':attempts}
    return {'seed':seed,'geometric_eligible':True,'accepted':False,'attempts':attempts}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='cube_composition_artifacts/tasks');p.add_argument('--count',type=int,default=16)
    p.add_argument('--splits',nargs='+',default=['dev']);p.add_argument('--kinds',nargs='+',default=['supported_easy','supported_conflict','coupled_ood','infeasible'])
    p.add_argument('--workers',type=int,default=6);args=p.parse_args();root=Path(args.out);root.mkdir(parents=True,exist_ok=True);started=time.time();manifest={}
    with mp.get_context('spawn').Pool(args.workers,initializer=init_worker) as pool:
        for split,offset in [('dev',2100001),('val',3100001),('test',4100001)]:
            if split not in args.splits:continue
            for k,kind in enumerate(['supported_easy','supported_conflict','coupled_ood','infeasible']):
                if kind not in args.kinds:continue
                accepted=[];attempts=[];next_seed=offset+k*100000
                while len(accepted)<args.count:
                    jobs=[(s,kind) for s in range(next_seed,next_seed+max(32,args.workers*8))];next_seed+=len(jobs)
                    for result in pool.imap(candidate,jobs,chunksize=1):
                        attempts.append({key:value for key,value in result.items() if key not in ['case','witness']})
                        if result.get('accepted') and len(accepted)<args.count:
                            index=len(accepted);accepted.append(result['case']);proof=result['witness']
                            if proof is not None:
                                folder=root/'witnesses'/f'{split}_{kind}';folder.mkdir(parents=True,exist_ok=True)
                                np.savez_compressed(folder/f'{index:04d}.npz',**proof)
                    if next_seed-offset-k*100000>args.count*400:raise RuntimeError('Insufficient independently verified feasible tasks')
                arrays={key:np.stack([c[key] for c in accepted]) for key in accepted[0]};path=root/f'{split}_{kind}.npz';np.savez_compressed(path,**arrays)
                record={'count':len(accepted),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'candidate_results':attempts,
                    'selection':'Geometry and independent native expert only; no pretrained-policy outcome filter. Witnesses are not optimizer/training inputs.'}
                manifest[split+'_'+kind]=record;(root/'manifest.json').write_text(json.dumps({'args':vars(args),'banks':manifest,'elapsed_seconds':time.time()-started,
                    'generator_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                    'witness_controller':{'transport_gain':30,'max_transport_action':.65,'node_tolerance_m':.006}},indent=2)+'\n')
                print('SAVED',split,kind,len(accepted),'elapsed',time.time()-started,flush=True)
    (root/'DONE.json').write_text(json.dumps({'complete':True,'banks':len(manifest)}))


if __name__=='__main__':main()
