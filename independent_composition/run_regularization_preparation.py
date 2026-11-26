"""Durable single-skill observation collection and existing-encoder calibration."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['cube','tworoom'],required=True)
    p.add_argument('--launch',action='store_true');p.add_argument('--worlds',type=int,default=32)
    a=p.parse_args();root=Path('/content/cs585_regularization');art=root/'independent_composition_artifacts'
    control=art/('private_encoder_v140_'+a.env+'_protocol')
    if a.launch:
        for name in ['regularized_latent_composition.py','regularization_models.py','collect_private_encoder_observations.py',
                     'audit_static_private_encoders.py','calibrate_static_private_encoder.py']:
            target=root/'independent_composition'/name;source=Path('/content')/name
            if target.exists():assert target.read_bytes()==source.read_bytes(),name
            else:shutil.copy2(source,target)
        assert not control.exists();control.mkdir()
        with (control/'queue.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--env',a.env,'--worlds',str(a.worlds)],
                cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'environment':a.env,'control':str(control),'worlds':a.worlds}
        (control/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    base='artifacts/final/base_policy.pt' if a.env=='tworoom' else 'cube/artifacts/final/base_policy.pt'
    keepout='addon_artifacts/final/keepout.pt' if a.env=='tworoom' else 'cube_addon_artifacts/final/keepout_s42/adapter.pt'
    waypoint='addon_artifacts/final/waypoint.pt' if a.env=='tworoom' else 'cube_addon_artifacts/final/waypoint_s42/adapter.pt'
    common=['--env',a.env,'--base',base,'--keepout',keepout,'--waypoint',waypoint]
    data=art/('private_encoder_v140_'+a.env+'_data')
    status={'pid':os.getpid(),'status':'running','started':time.time(),'jobs':[]}
    def save():(control/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def run(name,module,argv):
        command=[sys.executable,'-m',module]+argv
        row={'id':name,'status':'running','command':command};status['jobs'].append(row);save()
        with (control/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            row['child_pid']=child.pid;save();code=child.wait()
        row['returncode']=code;row['status']='complete' if code==0 else 'failed';save()
        if code:status['status']='failed';save();sys.exit(code)
    run('single_skill_observations','independent_composition.collect_private_encoder_observations',
        common+['--out',str(data),'--worlds',str(a.worlds),'--seed','1401001'])
    run('original_encoder_audit','independent_composition.audit_static_private_encoders',
        common+['--data',str(data),'--out',str(control/'original_encoder_audit.json')])
    paths=[]
    for skill in ['keepout','waypoint']:
        out=art/('private_encoder_v140_'+a.env+'_'+skill)
        run('calibrate_'+skill,'independent_composition.calibrate_static_private_encoder',
            common+['--skill',skill,'--data',str(data/skill/'train.npz'),'--out',str(out),'--steps','2000','--horizon','16'])
        paths.append(str(out/'best.pt'))
    run('calibrated_encoder_audit','independent_composition.audit_static_private_encoders',
        common+['--data',str(data),'--out',str(control/'calibrated_encoder_audit.json'),'--encoders']+paths+['--horizon','16'])
    status.update(status='complete',completed=time.time());save()


if __name__=='__main__':main()
