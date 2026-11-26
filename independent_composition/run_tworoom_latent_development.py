"""Launch/resume a finite, inspectable queue on the authorized evaluation host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--launch',action='store_true')
    a=p.parse_args()
    root=Path('/content/cs585_independent_composition')
    artifacts=root/'independent_composition_artifacts'
    source=artifacts/'cube_diverse_v34_source_tree'
    protocol=artifacts/'tworoom_latent_v132_protocol';protocol.mkdir(exist_ok=True)
    if a.launch:
        target=source/'independent_composition/evaluate_tworoom_amplitude.py'
        uploaded=Path('/content/evaluate_tworoom_amplitude.py')
        if target.exists():assert target.read_bytes()==uploaded.read_bytes()
        else:shutil.copy2(uploaded,target)
        shutil.copy2('/content/cube_identical_config.json',protocol/'cube_identical_config.json')
        assert not (protocol/'launch.json').exists(), 'Inspect existing process; do not launch twice'
        with (protocol/'queue.log').open('w') as f:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=root,
                                     stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'source_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                 'config_sha256':hashlib.sha256((protocol/'cube_identical_config.json').read_bytes()).hexdigest(),
                 'cases_per_stratum':16,'status':'launched'}
        (protocol/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt));return
    jobs=[]
    for kind in ['supported_easy','supported_conflict','coupled_ood']:
        out=artifacts/'tworoom_latent_v132_dev'/kind
        argv=['--tasks',str(artifacts/'tworoom_transfer_v128_dev_bank'/('val_'+kind+'.npz')),
              '--config',str(protocol/'cube_identical_config.json'),'--out',str(out),
              '--count','16','--execute','8']
        code='import sys,runpy;sys.path.insert(0,'+repr(str(source))+');sys.argv='+repr(['independent_composition.evaluate_tworoom_amplitude']+argv)+';runpy.run_module("independent_composition.evaluate_tworoom_amplitude",run_name="__main__")'
        jobs.append({'id':kind,'command':[sys.executable,'-c',code],'out':str(out),'status':'pending'})
    status={'pid':os.getpid(),'status':'running','jobs':jobs,'started':time.time()}
    def save():(protocol/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    for j in jobs:
        j['status']='running';save()
        with (protocol/(j['id']+'.log')).open('w') as log:
            result=subprocess.run(j['command'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        j['returncode']=result.returncode;j['status']='complete' if result.returncode==0 else 'failed';save()
        if result.returncode:
            status['status']='failed';save();sys.exit(result.returncode)
    status['status']='complete';status['completed']=time.time();save()


if __name__=='__main__':main()
