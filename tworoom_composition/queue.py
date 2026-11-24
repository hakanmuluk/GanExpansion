"""Recorded sequential experiment queue; completed artifacts are not overwritten."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--tasks',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=41);p.add_argument('--count',type=int,default=0)
    p.add_argument('--keepout',default='addon_artifacts/final/keepout.pt');p.add_argument('--waypoint',default='addon_artifacts/final/waypoint.pt')
    p.add_argument('--priors',default='composition_artifacts/priors/seed42.json')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    jobs=json.loads(Path(args.plan).read_text())
    for task in args.tasks:
        for job in jobs:
            target=out/Path(task).stem/job['name']
            if (target/'DONE.json').exists():print('CACHED',target,flush=True);continue
            command=[sys.executable,'-m','tworoom_composition.evaluate','--tasks',task,'--out',str(target),
                     '--config',json.dumps(job['search']),'--seed',str(args.seed),'--count',str(args.count),
                     '--keepout',args.keepout,'--waypoint',args.waypoint,'--priors',args.priors]
            print('START',target,flush=True);target.mkdir(parents=True,exist_ok=True)
            with (target/'console.log').open('a') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            row={'target':str(target),'argv':command,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':result.returncode}
            if (target/'summary.json').exists():row['summary']=json.loads((target/'summary.json').read_text())
            with (out/'events.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
            if result.returncode:raise RuntimeError('Experiment failed: '+str(target))
            print('DONE',target,row['summary']['success'],flush=True)
    (out/'DONE.json').write_text(json.dumps({'complete':True,'jobs':len(jobs)*len(args.tasks)})+'\n')


if __name__=='__main__':main()
