"""Finite inspectable experiment queues with explicit dependencies and workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--launch',action='store_true')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--worker',type=int,default=0)
    a=p.parse_args();protocol=Path(a.protocol);study=json.loads((protocol/'study.json').read_text())
    root=Path('/content/cs585_regularization')
    if a.launch:
        assert not (protocol/'launch.json').exists(),'Inspect previous handles rather than launch twice'
        workers=[]
        for i in range(a.workers):
            with (protocol/f'worker_{i}.log').open('w') as log:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--protocol',str(protocol),
                    '--workers',str(a.workers),'--worker',str(i)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append({'worker':i,'pid':process.pid})
        receipt={'workers':workers,'jobs':len(study['jobs']),'protocol':str(protocol)}
        (protocol/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    jobs=[dict(j,status='pending') for i,j in enumerate(study['jobs']) if i%a.workers==a.worker]
    status={'pid':os.getpid(),'worker':a.worker,'status':'running','started':time.time(),'jobs':jobs}
    target=protocol/f'worker_{a.worker}_status.json'
    def save():
        status['updated']=time.time();target.write_text(json.dumps(status,indent=2)+'\n')
    save()
    for j in jobs:
        if j.get('dependency'):
            while True:
                d=json.loads(Path(j['dependency']).read_text())
                if d['status']=='complete':break
                if d['status']=='failed' or not Path('/proc',str(d['pid'])).exists():
                    status.update(status='failed',reason='Dependency failed or stopped',dependency=j['dependency']);save();sys.exit(2)
                j['status']='waiting_for_live_dependency';save();time.sleep(2)
        for path in j.get('require_files',[]):assert Path(path).is_file(),path
        j['status']='running';save()
        with (protocol/(j['id']+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-m',j.get('module','independent_composition.evaluate_regularized_latents')]+j['argv'],
                cwd=root,stdout=log,stderr=subprocess.STDOUT)
            j['child_pid']=child.pid;save();code=child.wait()
        j['returncode']=code;j['status']='complete' if code==0 else 'failed';save()
        if code:status['status']='failed';save();sys.exit(code)
    status['status']='complete';status['completed']=time.time();save()


if __name__=='__main__':main()
