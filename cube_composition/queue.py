"""Recorded sequential native experiment queue; completion is never inferred."""
import argparse,json,subprocess,sys,time,os,datetime
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model-seed',type=int,default=42);p.add_argument('--seed',type=int,default=41);p.add_argument('--count',type=int,default=0);p.add_argument('--producer-pid',type=int)
    p.add_argument('--evaluator',default=__package__+'.evaluate')
    args=p.parse_args();root=Path(args.out);root.mkdir(parents=True,exist_ok=True);jobs=json.loads(Path(args.plan).read_text())
    for task in args.tasks:
        while not Path(task).exists():
            if not args.producer_pid:raise FileNotFoundError(task)
            try:os.kill(args.producer_pid,0)
            except ProcessLookupError:raise RuntimeError('Task producer terminated without '+task)
            time.sleep(2)
        for job in jobs:
            out=root/Path(task).stem/job['name']
            if (out/'DONE.json').exists():print('CACHED',out,flush=True);continue
            out.mkdir(parents=True,exist_ok=True)
            argv=[sys.executable,'-m',args.evaluator,'--tasks',task,'--out',str(out),'--config',json.dumps(job['search']),
                '--model-seed',str(args.model_seed),'--seed',str(args.seed),'--count',str(job.get('count',args.count))]
            print('START',out,flush=True)
            with (out/'console.log').open('a') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            event={'out':str(out),'argv':argv,'returncode':r.returncode,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            with (root/'events.jsonl').open('a') as log:log.write(json.dumps(event)+'\n')
            if r.returncode:raise RuntimeError('Experiment failed: '+str(out))
            print('DONE',out,json.loads((out/'summary.json').read_text())['success'],flush=True)
    (root/'DONE.json').write_text(json.dumps({'complete':True,'experiments':len(jobs)*len(args.tasks)}))


if __name__=='__main__':main()
