"""Build the portable regularizer and check original-output parity in isolation."""
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
    root=Path('/content/cs585_regularization');art=root/'independent_composition_artifacts'
    control=art/'regularization_v143_runtime_checks';bundle=art/'regularization_v143_runtime'
    installed=art/'regularization_v143_installed';wheels=art/'regularization_v143_wheels'
    if a.launch:
        assert not control.exists();control.mkdir()
        with (control/'queue.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=root,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'control':str(control)}
        (control/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    status={'pid':os.getpid(),'status':'running','started':time.time(),'jobs':[]}
    def save():(control/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def run(name,command,env=None):
        row={'id':name,'status':'running'};status['jobs'].append(row);save()
        with (control/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            row['child_pid']=child.pid;save();code=child.wait()
        row['returncode']=code;row['status']='complete' if code==0 else 'failed';save()
        if code:status['status']='failed';save();sys.exit(code)
    cube=art/'regularization_v142_cube_protocol/freeze.json'
    tworoom=art/'regularization_v142_tworoom_protocol/freeze.json'
    assert cube.exists() and tworoom.exists()
    run('export',[sys.executable,'-m','independent_composition.export_regularization_runtime',
        '--cube-freeze',str(cube),'--tworoom-freeze',str(tworoom),'--planning-dir',str(root/'independent_composition'),'--out',str(bundle)])
    run('build_wheel',[sys.executable,'-m','pip','wheel',str(bundle),'--no-deps','--no-build-isolation','--no-index','-w',str(wheels)])
    new=list(wheels.glob('*.whl'));assert len(new)==1
    old=list((art/'latent_optimization_release/wheels').glob('*.whl'));assert len(old)==2
    run('install_wheels',[sys.executable,'-m','pip','install','--no-deps','--no-index','--target',str(installed)]+[str(p) for p in old+new])
    isolated=dict(os.environ);isolated['PYTHONPATH']=str(installed);isolated['OMP_NUM_THREADS']='2';isolated['MKL_NUM_THREADS']='2'
    reference_summary={}
    for environment,freeze in [('tworoom',tworoom),('cube',cube)]:
        frozen=json.loads(freeze.read_text());models=frozen['role_model_paths']
        for mode in ['baseline','prior','prior_cycle']:
            common=['--env',environment,'--base',models['base'],'--keepout',models['keepout'],'--waypoint',models['waypoint'],
                    '--mode',mode,'--reference',str(control/(environment+'_'+mode+'_reference_raw_DO_NOT_TRANSFER.npz'))]
            if mode=='prior_cycle':common+=['--encoders',frozen['encoder_paths']['keepout'],frozen['encoder_paths']['waypoint']]
            run(environment+'_'+mode+'_reference',[sys.executable,'-m','independent_composition.check_regularization_runtime']+common+[
                '--write-reference',str(art/('regularization_v141_'+environment+'_eval/baseline/supported_easy')),
                '--config',str(freeze.parent/(mode+'.json')),'--out',str(control/(environment+'_'+mode+'_reference_created.json'))])
            for kind,path in [('source',bundle),('installed',installed)]:
                proof=control/(environment+'_'+mode+'_'+kind+'_proof.json')
                run(environment+'_'+mode+'_'+kind,[sys.executable,'/content/check_regularization_runtime.py']+common+[
                    '--bundle',str(path),'--out',str(proof)],env=isolated)
                reference_summary[environment+'/'+mode+'/'+kind]=json.loads(proof.read_text())
    (control/'parity_summary.json').write_text(json.dumps(reference_summary,indent=2)+'\n')
    (control/'DONE.json').write_text(json.dumps({'source_and_installed_parity_cells':len(reference_summary),
        'all_actions_and_codes_bitwise_equal':True,'simulator_imports_blocked':True,'raw_arrays_exported':False})+'\n')
    status.update(status='complete',completed=time.time());save()


if __name__=='__main__':main()
