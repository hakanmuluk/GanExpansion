"""Independently replay new policy traces using the sealed native geometry audit.

The imported audit only reads its sources. Every output is in the new namespace;
failed episodes are audited as well as successes. No policy is rerun or repaired.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
from cube_composition.geometry_audit import init_worker,replay


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    groups=[];jobs=[]
    for root in map(Path,args.root):
        for done in sorted(root.rglob('DONE.json')):
            folder=done.parent
            if not (folder/'records.json').exists() or not (folder/'config.json').exists():continue
            rows=json.loads((folder/'records.json').read_text());groups.append((folder,rows))
            jobs.extend((folder/f"rollout_{r['context']:04d}.npz",r,None) for r in rows)
    if not jobs:raise ValueError('No completed native evaluations')
    descriptor=hashlib.sha256(json.dumps([(str(p),r) for p,r,_ in jobs],sort_keys=True).encode()).hexdigest()
    config={'args':vars(args),'jobs_sha256':descriptor,'episodes':len(jobs),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'native_audit_source_sha256':hashlib.sha256(Path('cube_composition/geometry_audit.py').read_bytes()).hexdigest()}
    config_path=out/'config.json';events=out/'events.jsonl'
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Inputs changed: use a new output folder'
    else:config_path.write_text(json.dumps(config,indent=2)+'\n')
    records=[json.loads(x) for x in events.read_text().splitlines() if x.strip()] if events.exists() else []
    completed={r['source']:r for r in records}
    for source,r in completed.items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==r['source_sha256']
    remaining=[j for j in jobs if str(j[0]) not in completed];started=time.perf_counter()
    with events.open('a',buffering=1) as stream,mp.get_context('spawn').Pool(args.workers,initializer=init_worker) as pool:
        for record in pool.imap(replay,remaining,chunksize=1):
            records.append(record);stream.write(json.dumps(record)+'\n')
            if len(records)%64==0:print('AUDITED',len(records),'of',len(jobs),flush=True)
    (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    summary={'complete':True,'episodes':len(records),'geometry_samples':sum(r['geometry_samples'] for r in records),'native_controls':sum(r['native_control_steps'] for r in records),'maximum_replay_pose_error_m':max(r['native_replay_max_pose_error_m'] for r in records),'seconds':time.perf_counter()-started,'scope':'Independent action replay. Conservative task predicates and pose trajectories match for both successes and failures. Additional actual oriented-box geometry sampled every native 2ms step; no continuous-time safety guarantee.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'DONE.json').write_text(json.dumps(summary));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
