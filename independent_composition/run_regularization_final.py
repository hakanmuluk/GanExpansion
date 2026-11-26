"""Durable frozen regularization comparison with fresh native-certified scenes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True);p.add_argument('--launch',action='store_true')
    p.add_argument('--prior-method',default='prior01');p.add_argument('--combined-method',default='prior_cycle01')
    a=p.parse_args();root=Path('/content/cs585_regularization');art=root/'independent_composition_artifacts'
    control=art/('regularization_v142_'+a.env+'_launch');protocol=art/('regularization_v142_'+a.env+'_protocol')
    bank=art/('regularization_v142_'+a.env+'_bank');evaluations=art/('regularization_v142_'+a.env+'_eval')
    if a.launch:
        for name in ['freeze_regularized_composition.py']:
            target=root/'independent_composition'/name;source=Path('/content')/name
            if target.exists():assert target.read_bytes()==source.read_bytes()
            else:shutil.copy2(source,target)
        assert not control.exists();control.mkdir()
        with (control/'queue.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--env',a.env,
                '--prior-method',a.prior_method,'--combined-method',a.combined_method],cwd=root,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'environment':a.env,'control':str(control)}
        (control/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    status={'pid':os.getpid(),'status':'running','started':time.time(),'stages':[]}
    def save():(control/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def run(name,module,argv):
        row={'id':name,'status':'running'};status['stages'].append(row);save()
        with (control/(name+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-m',module]+argv,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            row['child_pid']=child.pid;save();code=child.wait()
        row['returncode']=code;row['status']='complete' if code==0 else 'failed';save()
        if code:status['status']='failed';save();sys.exit(code)
    run('freeze','independent_composition.freeze_regularized_composition',['--env',a.env,
        '--study',str(art/('regularization_v141_'+a.env+'_study')),'--out',str(protocol),
        '--prior-method',a.prior_method,'--combined-method',a.combined_method,
        '--scene-seed','1897500001' if a.env=='tworoom' else '1898500001'])
    generator='independent_composition.generate_final_tworoom' if a.env=='tworoom' else 'independent_composition.generate_cube_frozen_bank'
    argv=['--protocol',str(protocol/'freeze.json'),'--out',str(bank),'--count','64']
    if a.env=='cube':argv+=['--workers','4']
    run('fresh_data',generator,argv)
    frozen=json.loads((protocol/'freeze.json').read_text());jobs=[]
    for method in frozen['final_methods']:
        cfg=json.loads((protocol/(method+'.json')).read_text())
        for kind in ['supported_easy','supported_conflict','coupled_ood']:
            out=evaluations/method/kind
            argv=['--env',a.env,'--base',frozen['role_model_paths']['base'],'--keepout',frozen['role_model_paths']['keepout'],
                  '--waypoint',frozen['role_model_paths']['waypoint'],'--tasks',str(bank/('final_'+kind+'.npz')),
                  '--config',str(protocol/(method+'.json')),'--out',str(out),'--count','64','--execute','8',
                  '--freeze',str(protocol/'freeze.json'),'--bank-manifest',str(bank/'manifest.json'),'--method-name',method]
            if cfg['cycle_weight']:argv+=['--encoders',frozen['encoder_paths']['keepout'],frozen['encoder_paths']['waypoint']]
            jobs.append({'id':method+'_'+kind,'argv':argv,'out':str(out)})
    (protocol/'study.json').write_text(json.dumps({'scope':'Frozen final matched9cells.','jobs':jobs},indent=2)+'\n')
    row={'id':'comparisons','status':'running','children':[]};status['stages'].append(row);save();children=[]
    for i in range(2):
        log=(control/f'comparisons_worker{i}.log').open('w')
        child=subprocess.Popen([sys.executable,'/content/run_regularization_grid.py','--protocol',str(protocol),'--workers','2','--worker',str(i)],
            cwd=root,stdout=log,stderr=subprocess.STDOUT);children.append((child,log));row['children'].append(child.pid);save()
    codes=[]
    for child,log in children:codes.append(child.wait());log.close()
    row['returncodes']=codes;row['status']='complete' if not any(codes) else 'failed';save()
    if any(codes):status['status']='failed';save();sys.exit(3)
    if a.env=='tworoom':
        run('native_replay','independent_composition.audit_tworoom_latent_native',[
            '--protocol',str(protocol/'freeze.json'),'--evaluations',str(evaluations),'--out',str(art/'regularization_v142_tworoom_native_replay')])
    status.update(status='complete',completed=time.time());save()


if __name__=='__main__':main()
