"""Replay new Cube policies with native physics and oriented-box diagnostics."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--count-per-run',type=int,default=16)
    p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    from cube_composition.geometry_audit import init_worker,replay
    jobs=[];inputs=[]
    for path in map(Path,args.runs):
        assert (path/'DONE.json').exists()
        rows=json.loads((path/'records.json').read_text())
        # Include every failure as well as a prefix, without changing outcomes.
        selected=[row for i,row in enumerate(rows) if i<args.count_per_run or not row['success']]
        jobs.extend((path/f"rollout_{row['context']:04d}.npz",row,None) for row in selected)
        inputs.append({'run':str(path),'records_sha256':digest_file(path/'records.json'),
                       'available_episodes':len(rows),'replayed_episodes':len(selected)})
    (out/'config.json').write_text(json.dumps({'args':vars(args),'inputs':inputs,
        'source_sha256':digest_file(__file__),'replay_source_sha256':digest_file('cube_composition/geometry_audit.py')},indent=2)+'\n')
    started=time.time();results=[]
    with mp.get_context('spawn').Pool(args.workers,initializer=init_worker) as pool:
        for row in pool.imap(replay,jobs,chunksize=1):
            results.append(row)
            if len(results)%32==0:print(json.dumps({'native_replays':len(results),'total':len(jobs)}),flush=True)
    (out/'records.json').write_text(json.dumps(results,indent=2)+'\n')
    summary={'episodes':len(results),'all_primary_outcomes_reproduced':True,
             'maximum_native_replay_pose_error_m':max(row['native_replay_max_pose_error_m'] for row in results),
             'geometry_samples':sum(row['geometry_samples'] for row in results),
             'primary_joint_success':float(np.mean([row['primary_joint_success'] for row in results])),
             'secondary_oriented_box_joint_success':float(np.mean([row['secondary_sampled_box_joint_success'] for row in results])),
             'seconds':time.time()-started,
             'scope':'Replay validation and independent oriented-box diagnostics, with all original strict success criteria retained. Prefix plus all failures is an audit sample, not an unbiased success-rate estimate.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'complete':True,'episodes':len(results)})+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
