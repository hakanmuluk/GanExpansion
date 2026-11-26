"""Wait on specific live recorded queues; never infer restart from a timeout."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--queues',nargs='+',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    while True:
        ready=True;receipts=[]
        for directory in args.queues:
            path=Path(directory)/'status.json'
            state=json.loads(path.read_text())
            if any(j['status']=='failed' for j in state['jobs']):raise RuntimeError(f'Training queue failed: {directory}')
            complete=all(j['status']=='complete' for j in state['jobs'])
            if not complete:
                ready=False
                try:os.kill(state['pid'],0)
                except ProcessLookupError:
                    updated=json.loads(path.read_text())
                    if all(j['status']=='complete' for j in updated['jobs']):continue
                    raise RuntimeError(f'Recorded queue is missing without a terminal receipt: {directory}')
                command=Path(f"/proc/{state['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
                if 'independent_composition.run_jobs' not in command or directory not in command:
                    raise RuntimeError(f'Recorded queue process does not match: {directory}')
            receipts.append({'directory':directory,'pid':state['pid'],'complete':complete})
        if ready:
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps({'ready':True,'queues':receipts,'time':time.time()},indent=2)+'\n')
            return
        print(json.dumps({'waiting_for_live_queues':receipts}),flush=True);time.sleep(10)


if __name__=='__main__':main()
