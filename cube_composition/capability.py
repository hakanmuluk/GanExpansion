"""Finite-domain native capability diagnostic, not a non-intersection proof."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from cube_addons.env import make_env,close_env,features
from cube_addons.contacts import contact_info,force_holding
from .evaluate import load_cases
from .models import FrozenBank
from .scene import prepare
from .geometry import JointObserver
from .simulator import forecast,close_pool


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=6)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--workers',type=int,default=4);args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);bank=FrozenBank(args.seed,'cuda');env=make_env();rows=[]
    values=np.array([-2.,-1.2,-.6,-.2,.2,.6,1.2,2.],np.float32);n=len(values);grid=np.array(np.meshgrid(values,values,indexing='ij')).reshape(2,-1).T
    codes=np.concatenate([np.stack([values,np.full(n,.8,np.float32)],-1),np.stack([np.full(n,.8,np.float32),values],-1),grid]);families=['keepout']*n+['waypoint']*n+['sum']*len(grid)
    (out/'config.json').write_text(json.dumps({'args':vars(args),'values':values.tolist(),'frozen_files':bank.files,'task_sha256':hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
        'scope':'80 exact native closed-loop futures per context, horizon240. Finite code grid and fixed codes only; no global continuous or time-varying reachability claim.',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2)+'\n')
    try:
        for index,case in enumerate(load_cases(args.tasks,args.count)):
            start=time.perf_counter();ob,info=prepare(env,case);contact_info(env,info);observer=JointObserver(case);observer.previous_held=force_holding(info)
            state=features(env,ob,info,case['initial_cube']);ctx=observer.features(info,0)
            runtime={'env':env,'ob':ob,'info':info,'observer':observer,'case':case,'previous_ee':None,'stable':0,'time':0};bank.reset_counts()
            result=forecast(bank,state,ctx,case['base_z'],codes,families,runtime,240,args.workers)
            with torch.inference_mode():
                s=torch.as_tensor(state,device='cuda')[None].expand(len(grid),-1);b=torch.as_tensor(case['base_z'],device='cuda')[None].expand(len(grid),-1);c=tuple(torch.as_tensor(v,device='cuda')[None].expand(len(grid),-1) for v in ctx)
                a,w=bank.pair(s,b,torch.as_tensor(grid,device='cuda'),c);agreement=(a-w).square().mean((1,2)).cpu().numpy()
            metrics=[p['observer'].result(p['stable']>=10,bool(np.linalg.norm(p['physics']['info']['privileged/block_0_pos']-case['goal'])<=.04)) for p in result['packets']]
            trajectory=(((result['paths'][:n,None]-result['paths'][None,n:2*n])/.1)**2).mean((2,3));imin=int(trajectory.argmin());ia=int(agreement.argmin())
            row={'context':index,'seed':int(case['seed']),'seconds':time.perf_counter()-start,'native_controls':result['native_controls'],'native_forks':result['native_forks'],'generator_evaluations':bank.calls,
                'any_joint_success_by_family':{f:any(m['success'] for m,ff in zip(metrics,families) if ff==f) for f in sorted(set(families))},
                'successful_codes_by_family':{f:int(sum(m['success'] for m,ff in zip(metrics,families) if ff==f)) for f in sorted(set(families))},
                'minimum_initial_action_disagreement':float(agreement[ia]),'minimum_initial_action_pair':grid[ia].tolist(),
                'minimum_closed_loop_trajectory_disagreement':float(trajectory.flat[imin]),'minimum_trajectory_pair':grid[imin].tolist(),
                'trajectory_minimizer_joint_success_if_execute_K':metrics[imin//n]['success'],'trajectory_minimizer_joint_success_if_execute_W':metrics[n+imin%n]['success'],
                'initial_agreement_minimizer_joint_success_K':metrics[ia//n]['success'],'initial_agreement_minimizer_joint_success_W':metrics[n+ia%n]['success'],
                'candidates':[{'family':f,'codes':z.tolist(),**m} for f,z,m in zip(families,codes,metrics)]}
            rows.append(row);np.savez_compressed(out/f'paths_{index:04d}.npz',codes=codes,families=np.array(families),paths=result['paths'].astype(np.float32),actions=result['actions'],initial_action_mse=agreement,trajectory_mse=trajectory,**case)
            (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');print(index,row['any_joint_success_by_family'],flush=True)
        bank.assert_frozen();(out/'DONE.json').write_text(json.dumps({'complete':True,'contexts':len(rows),'frozen_unchanged':True}))
    finally:close_pool();close_env(env)


if __name__=='__main__':main()
