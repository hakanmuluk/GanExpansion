"""Finite sequential evaluation queue; only complete receipts permit skipping."""
import argparse,json,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=8);p.add_argument('--module',default='cube_agreement.evaluate');args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    jobs=json.loads(Path(args.plan).read_text())
    with (out/(Path(args.plan).stem+'_events.jsonl')).open('a',buffering=1) as events:
        for bank in args.tasks:
            for job in jobs:
                folder=out/Path(bank).stem/job['name']
                if (folder/'DONE.json').exists():continue
                folder.mkdir(parents=True,exist_ok=True);argv=['python','-m',args.module,'--tasks',bank,'--out',str(folder),'--count',str(job.get('count',args.count))]
                for key,value in job.get('options',{}).items():argv.extend(['--'+key.replace('_','-'),str(value)])
                print('START',folder,flush=True);events.write(json.dumps({'event':'START','folder':str(folder),'argv':argv,'time':time.time()})+'\n')
                result=subprocess.run(argv,stdout=(folder/'console.log').open('w'),stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError('Evaluation failed: '+str(folder))
                summary=json.loads((folder/'summary.json').read_text());print('DONE',folder,summary['success'],summary['controlled_release_success'],flush=True);events.write(json.dumps({'event':'DONE','folder':str(folder),'time':time.time()})+'\n')
        (out/(Path(args.plan).stem+'_DONE.json')).write_text(json.dumps({'complete':True}))


if __name__=='__main__':main()
