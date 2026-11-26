"""Finite recorded TwoRoom latent-search configuration study, evaluation host only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',required=True);p.add_argument('--launch',action='store_true')
    a=p.parse_args();protocol=Path(a.protocol)
    manifest=json.loads((protocol/'study.json').read_text())
    root=Path('/content/cs585_independent_composition')
    source=root/'independent_composition_artifacts/cube_diverse_v34_source_tree'
    if a.launch:
        assert not (protocol/'launch.json').exists(), 'Inspect previous handle before a second launch'
        with (protocol/'queue.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--protocol',str(protocol)],
                                     cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'protocol':str(protocol),'jobs':len(manifest['jobs'])}
        (protocol/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt));return
    status={'pid':os.getpid(),'status':'running','started':time.time(),'jobs':manifest['jobs']}
    def save():(protocol/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    for job in status['jobs']:
        job['status']='running';save()
        argv=job['argv']
        code='import sys,runpy;sys.path.insert(0,'+repr(str(source))+');sys.argv='+repr(['independent_composition.evaluate_tworoom_latent_frozen']+argv)+';runpy.run_module("independent_composition.evaluate_tworoom_latent_frozen",run_name="__main__")'
        with (protocol/(job['id']+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,'-c',code],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        job['returncode']=result.returncode;job['status']='complete' if result.returncode==0 else 'failed';save()
        if result.returncode:
            status['status']='failed';save();sys.exit(result.returncode)
    status['status']='complete';status['completed']=time.time();save()


if __name__=='__main__':main()
