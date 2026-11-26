"""Recorded own-skill and composition validation for independent denoisers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=16)
    p.add_argument('--strata',nargs='+',default=['supported_easy','supported_conflict'])
    p.add_argument('--model-root',default='independent_composition_artifacts/diffusion')
    p.add_argument('--sampling-steps',type=int,default=20)
    p.add_argument('--own-only',action='store_true');args=p.parse_args()
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh validation root required')
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir()
    models=[str(Path(args.model_root)/f'{args.env}_{s}_v1'/'best.pt') for s in ['keepout','waypoint']]
    variants=[('keepout',1.,99),('waypoint',1.,99)]
    if not args.own_only:variants += [('mean',1.,99),('mean',1.,50),('ratio',.5,99),('ratio',1.,99),('ratio',1.5,99),('ratio',1.,50),('sum',1.,99)]
    taskroot='composition_artifacts/tasks' if args.env=='tworoom' else 'cube_composition_artifacts/tasks_v2'
    jobs=[]
    for method,guidance,warm in variants:
        for kind in args.strata:
            name=f'{method}_g{guidance}_warm{warm}_{kind}'
            command=[sys.executable,'-u','-m','independent_composition.evaluate_diffusion',
                     '--env',args.env,'--tasks',f'{taskroot}/val_{kind}.npz','--checkpoints',*models,
                     '--out',str(out/'results'/name),'--method',method,'--guidance',str(guidance),
                     '--warm-time',str(warm),'--count',str(args.count),'--sampling-steps',str(args.sampling_steps)]
            jobs.append({'id':name,'method':method,'kind':kind,'command':command,'status':'pending'})
    state={'args':vars(args),'created':time.time(),'models':{x:digest_file(x) for x in models},'jobs':jobs,
           'scope':'Development validation; independent skill success is checked separately from joint success.'}
    def save():
        tmp=out/'status.tmp.json';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'status.json')
    save()
    for job in jobs:
        job['status']='running';job['started']=time.time();save()
        with (out/'logs'/f"{job['id']}.log").open('w') as log:
            proc=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
            job['pid']=proc.pid;save();code=proc.wait()
        job['returncode']=code;job['finished']=time.time();job['status']='complete' if code==0 else 'failed'
        if code==0:job['summary']=json.loads((out/'results'/job['id']/'summary.json').read_text())
        save();print(json.dumps({'id':job['id'],'status':job['status'],'summary':job.get('summary')}),flush=True)
        if code:
            print((out/'logs'/f"{job['id']}.log").read_text()[-5000:],flush=True)
            raise RuntimeError(f"Failed {job['id']}; inspect terminal status before continuing")
    (out/'DONE.json').write_text(json.dumps({'complete':True,'jobs':len(jobs)})+'\n')


if __name__=='__main__':main()
