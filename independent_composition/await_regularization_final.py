"""Start a frozen final only after the known development workers terminate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['cube','tworoom'],required=True);p.add_argument('--launch',action='store_true')
    a=p.parse_args();root=Path('/content/cs585_regularization');art=root/'independent_composition_artifacts'
    control=art/('regularization_v142_'+a.env+'_await')
    if a.launch:
        assert not control.exists();control.mkdir()
        with (control/'await.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--env',a.env],cwd=root,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'control':str(control),'environment':a.env}
        (control/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    status={'pid':os.getpid(),'status':'waiting_on_known_development_workers','started':time.time()}
    def save():(control/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    while True:
        paths=list((art/('regularization_v141_'+a.env+'_study')).glob('worker_*_status.json'));assert len(paths)==2
        workers=[json.loads(path.read_text()) for path in paths]
        status['workers']=[{'pid':w['pid'],'status':w['status'],'live':Path('/proc',str(w['pid'])).exists()} for w in workers]
        status['updated']=time.time();save()
        if any(w['status']=='failed' or (w['status']!='complete' and not Path('/proc',str(w['pid'])).exists()) for w in workers):
            status['status']='failed_dependency';save();sys.exit(2)
        if all(w['status']=='complete' for w in workers):break
        time.sleep(2)
    result=subprocess.run([sys.executable,'/content/run_regularization_final.py','--env',a.env,'--launch'],
                          cwd=root,text=True,capture_output=True)
    status['returncode']=result.returncode;status['stdout']=result.stdout;status['stderr']=result.stderr
    status['status']='final_launched' if result.returncode==0 else 'failed_to_launch_final';save()
    print(result.stdout,flush=True)
    if result.returncode:sys.exit(result.returncode)


if __name__=='__main__':main()
