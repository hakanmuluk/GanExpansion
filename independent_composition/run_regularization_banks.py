"""Regenerate development-only banks after the runtime reset; no raw transfer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['cube','tworoom'],required=True);p.add_argument('--launch',action='store_true')
    a=p.parse_args();root=Path('/content/cs585_regularization');art=root/'independent_composition_artifacts'
    control=art/('regularization_v141_'+a.env+'_bank_protocol')
    if a.launch:
        name='generate_regularization_cube_dev.py';target=root/'independent_composition'/name
        if not target.exists():shutil.copy2(Path('/content')/name,target)
        assert not control.exists();control.mkdir()
        with (control/'queue.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--env',a.env],cwd=root,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'control':str(control),'environment':a.env}
        (control/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    bank=art/('regularization_v141_'+a.env+'_bank')
    module='tworoom_composition.data' if a.env=='tworoom' else 'independent_composition.generate_regularization_cube_dev'
    argv=['--out',str(bank),'--count','64' if a.env=='tworoom' else '16','--splits','val',
          '--kinds','supported_easy','supported_conflict','coupled_ood']
    argv+=['--device','cuda'] if a.env=='tworoom' else ['--workers','4']
    status={'pid':os.getpid(),'status':'running','started':time.time(),'module':module,'argv':argv}
    (control/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    with (control/'generation.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-m',module]+argv,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        status['child_pid']=child.pid;(control/'status.json').write_text(json.dumps(status,indent=2)+'\n');code=child.wait()
    status.update(status='complete' if code==0 else 'failed',returncode=code,completed=time.time())
    (control/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    if code:sys.exit(code)


if __name__=='__main__':main()
