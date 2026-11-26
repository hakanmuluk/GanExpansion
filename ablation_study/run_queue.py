"""Durable, serial GPU queue. A failed experiment is recorded, never retuned.

Only ready jobs run. Waiting dependencies are tied to verified live process
handles, not inferred from a status file alone. The manifest may gain independent
jobs, but a launched job's exact command/config hash is frozen in its receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def live(pid):
    root=Path('/proc')/str(pid)
    try:return bool((root/'cmdline').read_bytes()) and (root/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def atomic(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    status={'pid':os.getpid(),'manifest':args.manifest,'started':time.time(),'jobs':{}};atomic(out/'status.json',status)
    while True:
        spec=json.loads(Path(args.manifest).read_text());ready=[];waiting=[]
        for job in spec['jobs']:
            key=job['id']
            if key in status['jobs']:continue
            if any(not Path(x).is_file() for x in job.get('required_files',[])):
                waiting.append(key);continue
            # Recorded external references/data jobs must be terminal before
            # dependent work or another GPU-training process starts.
            busy=False
            for name in job.get('wait_for_processes',[]):
                receipt=Path(name)
                if not receipt.is_file():busy=True;continue
                record=json.loads(receipt.read_text())
                if live(record['pid']):busy=True
            if busy:waiting.append(key);continue
            dependencies=job.get('after_jobs',[])
            if any(status['jobs'].get(x,{}).get('state')!='complete' for x in dependencies):
                waiting.append(key);continue
            ready.append(job)
        if not ready:
            status['waiting_jobs']=waiting;status['last_poll']=time.time();atomic(out/'status.json',status)
            if not waiting and spec.get('closed',False):
                atomic(out/'DONE.json',{'jobs':status['jobs'],'complete':True});return
            # This is the queue process's own short poll, not an agent tool wait.
            time.sleep(5);continue
        job=min(ready,key=lambda x:(x.get('priority',100),x['id']));key=job['id']
        original_config_hashes={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in job.get('configs',[])}
        row={'state':'running','command':job['command'],'started':time.time(),'configuration_sha256':original_config_hashes}
        status['jobs'][key]=row;atomic(out/'status.json',status)
        with (out/(key+'.log')).open('w') as log:
            env={**os.environ,'MUJOCO_GL':'egl','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                 'PYTHONPATH':str(Path.cwd())+':'+str(Path.cwd()/'cube')+':'+os.environ.get('PYTHONPATH','')}
            process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,env=env)
            row['pid']=process.pid;atomic(out/'status.json',status)
            code=process.wait()
        row.update(returncode=code,finished=time.time(),state='complete' if code==0 else 'failed')
        missing=[f for f in job.get('expected_files',[]) if not Path(f).is_file()]
        unchanged=all(hashlib.sha256(Path(s).read_bytes()).hexdigest()==h for s,h in original_config_hashes.items())
        row.update(missing_expected_files=missing,configuration_unchanged=unchanged)
        if missing or not unchanged:row['state']='failed'
        atomic(out/'status.json',status)
        print(json.dumps({'id':key,**row}),flush=True)


if __name__=='__main__':main()
