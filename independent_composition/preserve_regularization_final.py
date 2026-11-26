"""Verify terminal experiments and preserve source, weights and aggregates only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    root=Path('/content/cs585_regularization');art=root/'independent_composition_artifacts'
    control=art/'regularization_v145_final_backup'
    if a.launch:
        assert not control.exists();control.mkdir()
        with (control/'backup.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=root,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'control':str(control)}
        (control/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    status={'pid':os.getpid(),'status':'waiting_for_cube_final','started':time.time()}
    def save():(control/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    while True:
        final=json.loads((art/'regularization_v142_cube_launch/status.json').read_text())
        status['observed_final_pid']=final['pid'];status['observed_final_status']=final['status'];status['updated']=time.time();save()
        if final['status']=='complete':break
        if final['status']=='failed' or not Path('/proc',str(final['pid'])).exists():
            status['status']='failed_final_dependency';save();sys.exit(2)
        time.sleep(2)
    status['status']='verifying_and_archiving';save();files={};native_proof={};final_results={}
    def add(path,name=None):
        path=Path(path);name=name or str(path.relative_to(root))
        assert path.is_file() and not path.is_symlink()
        assert path.suffix in ['.py','.json','.toml','.whl','.pt']
        assert path.name not in ['records.json','planning.json']
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        files[name]=path
    for env in ['tworoom','cube']:
        protocol=art/('regularization_v142_'+env+'_protocol');freeze=json.loads((protocol/'freeze.json').read_text())
        parent=json.loads((art/('regularization_v142_'+env+'_launch/status.json')).read_text());assert parent['status']=='complete'
        for path,digest in freeze['source_files'].items():assert sha(path)==digest;add(path,'source/'+str(Path(path).relative_to(root)))
        for path,digest in freeze['models'].items():assert sha(root/path if not Path(path).is_absolute() else path)==digest
        manifest=json.loads((art/('regularization_v142_'+env+'_bank/manifest.json')).read_text())
        assert manifest['protocol_sha256']==sha(protocol/'freeze.json') and manifest['created']>freeze['created']
        aggregate={'protocol_sha256':manifest['protocol_sha256'],'created':manifest['created'],
                   'selection':manifest['selection'],'strata':{k:{field:v[field] for field in ['count','sha256','generation_seed']} for k,v in manifest['strata'].items()},
                   'raw_arrays_or_individual_certification_records_exported':False}
        target=protocol/'fresh_bank_aggregate.json';target.write_text(json.dumps(aggregate,indent=2)+'\n')
        subprocess.run([sys.executable,'-m','independent_composition.summarize_regularization_codes',
                        '--evaluations',str(art/('regularization_v142_'+env+'_eval')),'--out',str(protocol/'code_effects.json')],cwd=root,check=True)
        final_results[env]={}
        for method in freeze['final_methods']:
            final_results[env][method]={}
            for kind in ['supported_easy','supported_conflict','coupled_ood']:
                folder=art/('regularization_v142_'+env+'_eval')/method/kind
                done=json.loads((folder/'DONE.json').read_text());proof=json.loads((folder/'frozen_proof.json').read_text())
                cfg=json.loads((folder/'config.json').read_text());summary=json.loads((folder/'summary.json').read_text())
                assert done['episodes']==summary['episodes']==64 and done['all_weights_frozen']
                assert done['inference_environment_queries']==0 and done['new_neural_modules']==0 and done['inference_tripwires_enabled']
                assert proof['source_models_config_bank_verified'] and proof['freeze_sha256']==sha(protocol/'freeze.json')
                assert cfg['models']==freeze['method_models'][method]
                assert cfg['task_sha256']==manifest['strata'][kind]['sha256']
                key=env+'/'+method+'/'+kind
                native_proof[key]={'episodes':64,'joint_successes':round(summary['success']*64),
                    'models_source_config_bank_frozen_verified':True,'inference_simulator_queries':0,
                    'all_weights_frozen':True,'new_neural_modules':0,'existing_private_encoders_used':bool(done['cycle_weight'])}
                if env=='cube':
                    reset=json.loads((folder/'observation_replay.json').read_text())
                    assert reset['native_derived_contact_force_refreshed_at_reset'] and reset['extra_integration_steps']==0
                    native_proof[key]['native_mujoco_execution']=True;add(folder/'observation_replay.json')
                for name in ['summary.json','config.json','DONE.json','frozen_proof.json']:add(folder/name)
                final_results[env][method][kind]=round(summary['success']*64)
        for folder in [protocol,art/('regularization_v142_'+env+'_launch'),art/('regularization_v141_'+env+'_study'),art/('private_encoder_v140_'+env+'_protocol')]:
            for f in folder.glob('*.json'):add(f)
        for f in (art/('regularization_v141_'+env+'_eval')).glob('*/*/summary.json'):
            for name in ['summary.json','config.json','DONE.json']:add(f.parent/name)
        for skill in ['keepout','waypoint']:
            folder=art/('private_encoder_v140_'+env+'_'+skill)
            for name in ['best.pt','config.json','initial_diagnostics.json','DONE.json']:add(folder/name)
        add(art/('private_encoder_v140_'+env+'_data/summary.json'))
    replay=art/'regularization_v142_tworoom_native_replay'
    measured=json.loads((replay/'summary.json').read_text());assert len(measured['benchmarks'])==9
    assert sum(v['episodes'] for v in measured['benchmarks'].values())==576
    for key,row in measured['benchmarks'].items():
        assert row['all_positions_and_outcomes_match'] and row['maximum_position_error_pixels']==0 and row['premature_native_done']==0
        assert native_proof['tworoom/'+key]['joint_successes']==row['native_joint_successes']
    for name in ['summary.json','DONE.json']:add(replay/name)
    for folder in [art/'regularization_v143_runtime_checks',art/'regularization_v144_helper_checks']:
        assert (folder/'DONE.json').exists()
        for f in folder.glob('*.json'):add(f)
    runtime=art/'regularization_v143_runtime';add(runtime/'pyproject.toml')
    for f in (runtime/'skill_latent_regularization').iterdir():
        if f.is_file() and f.suffix in ['.py','.json']:add(f)
    for f in (art/'regularization_v143_wheels').glob('*.whl'):add(f)
    for name in ['regularized_latent_composition.py','regularization_models.py','collect_private_encoder_observations.py',
                 'audit_static_private_encoders.py','calibrate_static_private_encoder.py','evaluate_regularized_latents.py',
                 'freeze_regularized_composition.py','check_regularization_loss_paths.py','export_regularization_runtime.py',
                 'check_regularization_runtime.py','summarize_regularization_codes.py']:
        add(root/'independent_composition'/name,'source/independent_composition/'+name)
    for name in ['bootstrap_regularization_host.py','run_regularization_preparation.py','run_regularization_banks.py',
                 'run_regularization_grid.py','run_regularization_final.py','await_regularization_final.py',
                 'run_regularization_runtime_checks.py','release_regularized_example.py','check_regularized_user_commands.py']:
        add(Path('/content')/name,'source/control_scripts/'+name)
    report={'scope':'All frozen18finalcells; nativeCubeexecution and independent nativeTwoRoom replay. No inference simulation or new neural modules.',
            'results_of64':final_results,'checks':native_proof,'tworoom_native_replay_trajectories':576,'all_checks_passed':True}
    target=control/'final_native_proof.json';target.write_text(json.dumps(report,indent=2)+'\n');add(target)
    manifest={'created':time.time(),'files':{name:{'sha256':sha(path),'bytes':path.stat().st_size} for name,path in sorted(files.items())},
              'raw_observations_actions_snapshots_or_individual_episode_records_exported':False}
    mp=control/'MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=art/'regularization_v139_v145_final_evidence.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz') as t:
        for name,path in sorted(files.items()):t.add(path,arcname=name,recursive=False)
        t.add(mp,arcname='MANIFEST.json',recursive=False)
    receipt={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'verified_source_and_aggregate_files':len(files)}
    (control/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    status.update(status='complete',completed=time.time(),receipt=receipt);save();print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
