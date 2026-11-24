"""Rescore unchanged saved actions with real bilateral contact evidence.

Unlike rerunning a policy with corrected features, this isolates the measurement
change: original actions and physical trajectories must replay identically.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import numpy as np
from . import geometry
from .contacts import contact_info,force_holding,SUPPORT_FORCE_N
from .env import make_env,reset_case,restore,completion,step_traced,close_env


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--skill',required=True,choices=['keepout','waypoint']);args=p.parse_args()
    source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    records=json.loads((source/'records.json').read_text());result=[];env=make_env()
    # Process-local replacement supports auditing earlier source versions too.
    # No actor is evaluated here; saved actions remain byte-for-byte unchanged.
    original=geometry.holding;geometry.holding=force_holding
    try:
        for row in records:
            filename=f"rollout_{row['context']:04d}_{row['code_index']:02d}.npz"
            with np.load(source/filename) as f:data={k:f[k] for k in f.files}
            reset_case(env,row['context_seed']);_,info=restore(env,data['initial_state']);contact_info(env,info)
            observer=geometry.Observer(data['initial'],data['goal'],data['spec'],args.skill)
            stable=0;forces=[];errors=[]
            for t,action in enumerate(data['actions']):
                before=info['privileged/block_0_pos'].copy();step,path=step_traced(env,action);_,_,_,_,info=step
                contact_info(env,info);after=info['privileged/block_0_pos']
                observer.update(before,after,info,path)
                native,released=completion(env,info);stable=stable+1 if released else 0
                forces.append([info['addon/left_cube_normal_force'],info['addon/right_cube_normal_force'],float(force_holding(info))])
                pose=np.r_[info['proprio/effector_pos'],after]
                errors.append(float(np.max(np.abs(pose-data['poses'][t+1]))))
            if max(errors)>1e-6:raise RuntimeError(f'Physical replay differs: {filename}: {max(errors)}')
            updated={**row,'legacy_success':row['success'],'success':bool(stable>=10 and observer.satisfied()),
                     'constraint_success':observer.satisfied(),'has_lifted':observer.has_lifted,'held_samples':observer.held_transport_samples,
                     'visited':observer.visited,'violated':observer.violated,'table_violation':observer.table_violation,
                     'force_contacts':True,'replay_pose_error_m':max(errors)}
            result.append(updated);np.savez_compressed(out/filename,contact_forces=np.asarray(forces),pose_error=np.asarray(errors))
        summary={'episodes':len(result),'released_success':float(np.mean([r['released_success'] for r in result])),
                 'legacy_full_success':float(np.mean([r['legacy_success'] for r in result])),
                 'force_contact_full_success':float(np.mean([r['success'] for r in result])),
                 'force_contact_constraint_success':float(np.mean([r['constraint_success'] for r in result])),
                 'maximum_replay_pose_error_m':max(r['replay_pose_error_m'] for r in result),
                 'support_definition':f'Both gripper pad groups have solved normal force >= {SUPPORT_FORCE_N}N and cube/EE centers within 0.065m; sampled at 50ms',
                 'interpretation':'Unchanged saved actions in identical native physics; this isolates observer correction, not retraining or improved policy.'}
        (out/'records.json').write_text(json.dumps(result,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    finally:geometry.holding=original;close_env(env)


if __name__=='__main__':main()
