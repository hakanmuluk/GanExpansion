"""Verify the completed evidence archive and assemble the reviewable release."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args();archive=Path(a.archive);assert sha(archive)==a.sha256
    artifacts=Path('independent_composition_artifacts');e=artifacts/'regularization_v139_v145_final_evidence'
    assert not e.exists();e.mkdir()
    with tarfile.open(archive) as t:
        manifest=json.load(t.extractfile('MANIFEST.json'));members=t.getmembers()
        assert len(members)==len(manifest['files'])+1
        seen=set()
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk()
            assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.name not in seen
            assert Path(m.name).suffix in ['.py','.json','.toml','.whl','.pt']
            assert Path(m.name).name not in ['records.json','planning.json'];seen.add(m.name)
            data=t.extractfile(m).read()
            if m.name!='MANIFEST.json':
                item=manifest['files'][m.name];assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
            target=e/m.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    receipt={'archive':str(archive),'sha256':a.sha256,'verified_files':len(manifest['files']),
             'raw_arrays_or_individual_records_exported':False}
    (artifacts/'regularization_v145_backup_verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
    data=e/'independent_composition_artifacts'
    proof=json.loads((data/'regularization_v145_final_backup/final_native_proof.json').read_text())
    assert proof['all_checks_passed'] and len(proof['checks'])==18
    assert all(v['episodes']==64 and v['models_source_config_bank_frozen_verified'] and v['all_weights_frozen']
               and v['inference_simulator_queries']==0 and v['new_neural_modules']==0 for v in proof['checks'].values())
    assert proof['tworoom_native_replay_trajectories']==576
    release=artifacts/'z_regularized_composition_release';assert release.exists()
    shutil.copytree(data/'regularization_v143_runtime',release/'source/regularizer',dirs_exist_ok=True)
    for f in (data/'regularization_v143_wheels').glob('*.whl'):shutil.copy2(f,release/'wheels'/f.name)
    ev=release/'evidence';ev.mkdir(exist_ok=True)
    shutil.copy2(data/'regularization_v145_final_backup/final_native_proof.json',ev/'final_native_proof.json')
    cells={};calibration={};effects={}
    for env in ['tworoom','cube']:
        protocol=data/('regularization_v142_'+env+'_protocol')
        frozen=json.loads((protocol/'freeze.json').read_text())
        bank=json.loads((protocol/'fresh_bank_aggregate.json').read_text())
        assert bank['protocol_sha256']==sha(protocol/'freeze.json') and bank['created']>frozen['created']
        assert all(v['count']==64 for v in bank['strata'].values())
        for name in ['freeze.json','code_effects.json','fresh_bank_aggregate.json']:
            shutil.copy2(protocol/name,ev/(env+'_'+name))
        effects[env]=json.loads((protocol/'code_effects.json').read_text())
        for role,filename in [('base','base_policy.pt'),('keepout','keepout_adapter.pt'),('waypoint','waypoint_adapter.pt')]:
            assert sha(release/'weights'/env/filename)==frozen['role_model_sha256'][role]
        for role in ['keepout','waypoint']:
            assert sha(release/'weights'/env/(role+'_encoder.pt'))==frozen['cycle_encoder_sha256'][role]
        for stage in [141,142]:
            key='v'+str(stage)+'_'+env;cells[key]={}
            for f in sorted((data/('regularization_v'+str(stage)+'_'+env+'_eval')).glob('*/*/summary.json')):
                cells[key][str(f.parent.relative_to(f.parents[2]))]=json.loads(f.read_text())
            assert len(cells[key])==(24 if stage==141 else 9)
        calibration[env]={}
        for name in ['original_encoder_audit.json','calibrated_encoder_audit.json']:
            src=data/('private_encoder_v140_'+env+'_protocol')/name
            calibration[env][name]=json.loads(src.read_text());shutil.copy2(src,ev/(env+'_'+name))
        shutil.copy2(data/('regularization_v141_'+env+'_study/loss_path_proof.json'),ev/(env+'_loss_path_proof.json'))
    native=data/'regularization_v142_tworoom_native_replay/summary.json'
    replay=json.loads(native.read_text());assert len(replay['benchmarks'])==9
    assert all(v['all_positions_and_outcomes_match'] and v['maximum_position_error_pixels']==0 and v['premature_native_done']==0 for v in replay['benchmarks'].values())
    shutil.copy2(native,ev/'tworoom_native_replay.json')
    runtime=data/'regularization_v143_runtime_checks/parity_summary.json'
    parity=json.loads(runtime.read_text());assert len(parity)==12
    assert all(v['original_actions_and_optimized_codes_bitwise_equal'] and v['simulator_and_original_project_imports_blocked'] for v in parity.values())
    shutil.copy2(runtime,ev/'runtime_parity_summary.json')
    helper=data/'regularization_v144_helper_checks/helper_proof.json';hp=json.loads(helper.read_text())
    assert all(hp[env]['api_all_three_modes_bitwise_match_original_reference'] and len(hp[env]['cli_modes'])==7 for env in ['cube','tworoom'])
    shutil.copy2(helper,ev/'helper_proof.json')
    results={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Existing base/action adapters unchanged; private Gaussian prior and optional original-private-encoder cycle, same latent solver and output rules.',
        'final_joint_successes_of64':proof['results_of64'],
        'joint_success':'Goal AND keepout AND waypoint AND original strict physical/wall/early-goal validity checks, not goal-only success.',
        'objective':'ActionMSEfirst8 + lambda_base*base_amplitude_tether +lambda_z*.5*(uK^2+uW^2)+lambda_cycle*meanactorMAE(E(current_state_repeated,issued_proposal16,own_context_repeated,own_base),u)',
        'settings':{'default':'prior','private_prior_weight':.01,'optional_cycle_weight':.01,'cycle_horizon':16,'agreement_horizon':8,'predict':16,'execute':8,'adam_steps':20,'starts':4,
                    'base_tether':{'tworoom':1.,'cube':.0005},'executors':{'tworoom':'clip(AW+AK-AB,-1,1)','cube':'AW'}},
        'training':'No generator/base/action-adapter retraining. Only original private encoder architecture calibrated on separate single-skill observations and known codes passed to frozen generators; whole-context heldout splits and within-condition action-shuffle controls.',
        'all_development_and_final_cells':cells,'code_effects':effects,'encoder_calibration':calibration,
        'verification':{'final_cells':18,'native_cube_episodes':576,'independent_native_tworoom_replays':576,'runtime_parity_cells':12,'both_env_all_three_modes_and_cli_commands_verified':True,'no_inference_simulator_queries':True,'new_neural_architectures':0},
        'limitations':['Single frozen model pair per environment; small result differences do not establish statistical superiority.','Coupled/OOD remains weak and is not declared solved.','A Gaussian point-prior encourages small codes; it does not make optimized codes Gaussian-distributed or prove semantic correctness.','Cycle consistency encourages code recovery, not joint task satisfaction.','Same optimization framework, but Cube executes waypoint output and TwoRoom uses original residual execution.','Inputs are original state features10/47, not images.'],
        'research_archive':receipt,'public_repository_and_readme_changed':False}
    (release/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    table=['\nFrozen fresh-test joint successes (64 independently feasible cases per cell)\n',
           'Environment / method             easy        conflict      coupled/OOD']
    for env in ['tworoom','cube']:
        for method in ['baseline','prior','prior_cycle']:
            row=proof['results_of64'][env][method]
            table.append(f"{env+' / '+method:31} {row['supported_easy']:2}/64       {row['supported_conflict']:2}/64         {row['coupled_ood']:2}/64")
    table+=['','All18finalcells were frozen before fresh case generation. The original native',
            'TwoRoom replay matched all576trajectories exactly; Cube was run directly in',
            'native MuJoCo. Differences are modest; coupled/OOD failures remain explicit.',
            'RESULTS.json contains every development and final cell and calibration metrics.\n']
    running=release/'RUNNING.txt';text=running.read_text();marker='\nFrozen fresh-test joint successes'
    if marker in text:text=text[:text.index(marker)]
    text+='\n'.join(table)
    for old,new in [('All18finalcells','All 18 final cells'),('all576trajectories','all 576 trajectories')]:text=text.replace(old,new)
    running.write_text(text)
    for folder in ['cube','tworoom','regularizer']:
        package=release/'source'/folder/({'cube':'cube_action_composition','tworoom':'tworoom_latent_composition','regularizer':'skill_latent_regularization'}[folder])
        m=json.loads((package/'manifest.json').read_text())
        for name,digest in m['files'].items():assert sha(package/name)==digest
        if isinstance(m.get('configuration_sha256'),dict):
            for name,digest in m['configuration_sha256'].items():assert sha(package/name)==digest
    for cache in release.rglob('__pycache__'):shutil.rmtree(cache)
    files={}
    for f in sorted(release.rglob('*')):
        assert not f.is_symlink()
        if f.is_file() and f.name!='MANIFEST.json':
            assert f.suffix in ['.py','.json','.txt','.toml','.whl','.pt']
            files[str(f.relative_to(release))]={'bytes':f.stat().st_size,'sha256':sha(f)}
    (release/'MANIFEST.json').write_text(json.dumps({'files':files,'checkpoint_count':10,'raw_arrays_included':False},indent=2)+'\n')
    target=artifacts/'z_regularized_composition_release.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(release.rglob('*')):
            if f.is_file():z.write(f,arcname=str(Path(release.name)/f.relative_to(release)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and len(z.namelist())==len(files)+1
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            rel='/'.join(Path(name).parts[1:])
            if rel!='MANIFEST.json':assert hashlib.sha256(z.read(name)).hexdigest()==files[rel]['sha256']
    receipt={'archive':str(target.resolve()),'sha256':sha(target),'bytes':target.stat().st_size,'verified_members':len(files)+1,'checkpoints':10,
             'all_model_and_source_hashes_verified':True,'full_results_and_evidence_included':True,'raw_arrays_included':False}
    (artifacts/'z_regularized_release_v145_verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
