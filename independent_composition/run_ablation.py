"""Recorded sequential validation sweeps; never open a final test bank."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from .latent_search import LatentConfig
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=16)
    p.add_argument('--strata',nargs='+',default=['supported_easy','supported_conflict'])
    p.add_argument('--families',nargs='+',default=['keepout','sum','waypoint'])
    p.add_argument('--steps',type=int,default=15);p.add_argument('--seed',type=int,default=41)
    p.add_argument('--window-root',default='independent_composition_artifacts/windows')
    p.add_argument('--variants',nargs='+',default=['agreement','self_01','self_1','executed_01','executed_1','executed_reconstruct_1','executed_nll_1'])
    args=p.parse_args();out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh ablation directory required')
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir()
    changes={'agreement':dict(cycle='none'),
             'self_01':dict(cycle='self',cycle_weight=.1),
             'self_1':dict(cycle='self',cycle_weight=1.),
             'executed_01':dict(cycle='executed',cycle_weight=.1),
             'executed_1':dict(cycle='executed',cycle_weight=1.),
             'executed_reconstruct_1':dict(cycle='executed',cycle_weight=1.,reconstruction_weight=1.),
             'executed_nll_1':dict(cycle='executed',cycle_weight=1.,cycle_metric='nll')}
    encoders=[str(Path(args.window_root)/f'{args.env}_{skill}_v1'/'best.pt') for skill in ['keepout','waypoint']]
    taskroot='composition_artifacts/tasks' if args.env=='tworoom' else 'cube_composition_artifacts/tasks_v2'
    jobs=[]
    for family in args.families:
        for name in args.variants:
            cfg=LatentConfig(family=family,steps=args.steps,**changes[name])
            for kind in args.strata:
                job_id=f'{family}_{name}_{kind}'
                command=[sys.executable,'-u','-m','independent_composition.evaluate_legacy',
                         '--env',args.env,'--tasks',f'{taskroot}/val_{kind}.npz',
                         '--out',str(out/'results'/job_id),'--count',str(args.count),
                         '--seed',str(args.seed),'--config',json.dumps(asdict(cfg)),
                         '--encoders',*encoders]
                jobs.append({'id':job_id,'command':command,'config':asdict(cfg),'kind':kind,'status':'pending'})
    state={'args':vars(args),'created':time.time(),
           'encoder_sha256':{p:digest_file(p) for p in encoders},
           'source_sha256':{p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')},
           'jobs':jobs,'scope':'Development validation only; all methods use current observations and tensor-only inference.'}
    def save():
        pending=out/'status.tmp.json';pending.write_text(json.dumps(state,indent=2)+'\n');pending.replace(out/'status.json')
    save()
    for job in jobs:
        job['started']=time.time();job['status']='running';save()
        with (out/'logs'/f"{job['id']}.log").open('w') as log:
            process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
            job['pid']=process.pid;save();code=process.wait()
        job['returncode']=code;job['finished']=time.time()
        job['status']='complete' if code==0 else 'failed'
        summary=out/'results'/job['id']/'summary.json'
        if code==0:job['summary']=json.loads(summary.read_text())
        save();print(json.dumps({'id':job['id'],'status':job['status'],
                               'success':job.get('summary',{}).get('success')}),flush=True)
        if code:
            print((out/'logs'/f"{job['id']}.log").read_text()[-5000:],flush=True)
            raise RuntimeError(f"Ablation failed: {job['id']}; inspect its existing output before continuing")
    (out/'DONE.json').write_text(json.dumps({'jobs':len(jobs),'successfully_completed':len(jobs)})+'\n')


if __name__=='__main__':main()
