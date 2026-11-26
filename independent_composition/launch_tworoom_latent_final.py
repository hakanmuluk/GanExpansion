"""Durable final freeze, new-scene generation, evaluation and native replay."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    root=Path('/content/cs585_independent_composition');art=root/'independent_composition_artifacts'
    source=art/'cube_diverse_v34_source_tree';control=art/'tworoom_latent_v136_launch'
    if a.launch:
        dev=json.loads((art/'tworoom_latent_v135_protocol/status.json').read_text())
        assert dev['status']=='complete' and all(j['status']=='complete' for j in dev['jobs'])
        for name in ['evaluate_tworoom_latent_frozen.py','freeze_tworoom_latent.py','audit_tworoom_latent_native.py']:
            target=source/'independent_composition'/name;uploaded=Path('/content')/name
            if target.exists():assert target.read_bytes()==uploaded.read_bytes()
            else:shutil.copy2(uploaded,target)
        assert not control.exists();control.mkdir()
        with (control/'queue.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=root,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'control':str(control),'stages':['freeze','fresh_data','comparisons','native_replay']}
        (control/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    protocol=art/'tworoom_latent_v136_protocol';bank=art/'tworoom_latent_v136_bank'
    evaluations=art/'tworoom_latent_v136_eval';audit=art/'tworoom_latent_v136_native_replay'
    status={'pid':os.getpid(),'status':'running','started':time.time(),'stages':[]}
    def save():(control/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def run(name,command):
        row={'id':name,'status':'running','command':command};status['stages'].append(row);save()
        with (control/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            row['child_pid']=child.pid;save();code=child.wait()
        row['returncode']=code;row['status']='complete' if code==0 else 'failed';save()
        if code:status['status']='failed';save();sys.exit(code)
    def module(name,argv):
        code='import sys,runpy;sys.path.insert(0,'+repr(str(source))+');sys.argv='+repr([name]+argv)+';runpy.run_module('+repr(name)+',run_name="__main__")'
        return [sys.executable,'-c',code]
    run('freeze',module('independent_composition.freeze_tworoom_latent',['--out',str(protocol)]))
    run('fresh_data',module('independent_composition.generate_final_tworoom',[
        '--protocol',str(protocol/'freeze.json'),'--out',str(bank),'--count','64']))
    jobs=[]
    for name,method in [('latent_residual','amplitude'),('coarse_only','amplitude'),('fixed_residual','fixed')]:
        for kind in ['supported_easy','supported_conflict','coupled_ood']:
            out=evaluations/name/kind
            argv=['--tasks',str(bank/('final_'+kind+'.npz')),'--config',str(protocol/(name+'.json')),
                  '--out',str(out),'--count','64','--execute','8','--method',method,
                  '--frozen-protocol',str(protocol/'freeze.json'),'--protocol-method',name,
                  '--bank-manifest',str(bank/'manifest.json')]
            jobs.append({'id':name+'_'+kind,'out':str(out),'argv':argv,'status':'pending'})
    (protocol/'study.json').write_text(json.dumps({'scope':'Final9cells, frozen before freshsceneconstruction.','jobs':jobs},indent=2)+'\n')
    run('comparisons',[sys.executable,'/content/run_tworoom_latent_final.py','--protocol',str(protocol)])
    run('native_replay',module('independent_composition.audit_tworoom_latent_native',[
        '--protocol',str(protocol/'freeze.json'),'--evaluations',str(evaluations),'--out',str(audit)]))
    status['status']='complete';status['completed']=time.time();save()


if __name__=='__main__':main()
