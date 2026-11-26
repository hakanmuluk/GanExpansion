"""Authoritative selected checkpoints after the complete, valid study finishes.

Use these original-file hashes to select compact exports; never select an older
invalid checkpoint merely because its canonical pathname is the same.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .export_completed import state_hash,SLOTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    manifest=json.loads(Path('ablation_study/queue_manifest.json').read_text())
    status=json.loads(Path('ablation_results/training_queue_v2/status.json').read_text())
    unfinished=[j['id'] for j in manifest['jobs'] if status['jobs'].get(j['id'],{}).get('state')!='complete']
    assert not unfinished,('Study is not complete',unfinished)
    binding_path=Path('ablation_results/aggregate/base_evaluation_checkpoint_binding.json')
    binding=json.loads(binding_path.read_text())
    assert binding['complete'] and binding['models_checked']==8
    bound={r['case']:r for r in binding['checks']}
    assert len(bound)==8 and all(r['passed'] for r in bound.values())
    cases=[]
    for env,seeds in [('tworoom',[42,123]),('cube',[42,43])]:
        for method in ['full','adversarial']:
            for seed in seeds:
                name=f'{env}_'+('latest_' if env=='cube' else '')+f'base_{method}_s{seed}'
                filename='snapshots/step_005000.pt' if env=='tworoom' else 'best.pt'
                cases.append({'case':name,'kind':'base_comparison','source':f'ablation_results/training/{name}/{filename}',
                    'selection':'Fixed final step5000' if env=='tworoom' else 'Original validation selection rule'})
    for env,seeds in [('tworoom',[123,2026]),('cube',[42,43])]:
        for skill in ['keepout','waypoint']:
            for variant in ['r1','r2','r4','r8','r16','r8_lora_only']:
                for seed in seeds:
                    name=f'{env}_{skill}_{variant}_s{seed}';folder='training' if env=='tworoom' else 'calibration'
                    cases.append({'case':name,'kind':'independent_skill','source':f'ablation_results/{folder}/{name}/best.pt',
                        'selection':'Original actor validation selection; Cube also uses original private-E calibration selection',
                        'required_original_base':'artifacts/final/base_policy.pt' if env=='tworoom' else 'cube/artifacts/final/base_policy.pt'})
    for row in cases:
        path=Path(row['source']);ck=torch.load(path,map_location='cpu',weights_only=False)
        row['original_source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        row['tensor_sha256']={key:state_hash(ck[key]) for key in SLOTS if key in ck}
        row['actor_selected_step']=ck['step']
        if row['kind']=='base_comparison':
            evidence=bound[row['case']]
            assert evidence['checkpoint']==row['source']
            assert evidence['checkpoint_sha256']==row['original_source_sha256']
            assert evidence['generator_tensor_sha256']==row['tensor_sha256'][evidence['generator_slot']]
            assert evidence['max_abs_error']<=evidence['absolute_tolerance']==1e-5
            env='tworoom' if row['case'].startswith('tworoom_') else 'cube'
            method='adversarial' if 'adversarial' in row['case'] else 'full'
            seed=row['case'].rsplit('_s',1)[1]
            evaluation=Path('ablation_results/evaluation')/f'original_gaussian_{env}_{method}_s{seed}'/'study_summary.json'
            assert hashlib.sha256(evaluation.read_bytes()).hexdigest()==evidence['evaluation_summary_sha256']
            assert json.loads(evaluation.read_text())['original_cli_configuration']['checkpoint']==row['source']
            assert evidence['episodes_checked']==(8192 if env=='tworoom' else 1200)
            row['base_evaluation_binding']=evidence
        if row['kind']=='base_comparison' and 'adversarial' in row['case']:
            cfg=ck.get('train_config',ck.get('args',{}))
            required=['rec_weight','latent_weight','kl_weight','style_weight']
            required.append('known_rec_weight' if row['case'].startswith('cube_') else 'prior_rec_weight')
            assert all(cfg.get(k)==0 for k in required),(row['case'],cfg)
            row['removed_objective_coefficients_verified_zero']=required
        if row['kind']=='independent_skill':
            assert ck['base_sha256']==hashlib.sha256(Path(row['required_original_base']).read_bytes()).hexdigest()
            row['ablation_config']=ck['ablation_config']
            evaluated=Path('ablation_results/evaluation')/('evaluate_'+row['case'])/'config.json'
            evaluation=json.loads(evaluated.read_text())
            assert evaluation['checkpoint_sha256']==row['original_source_sha256'],('Evaluated checkpoint differs',row['case'])
            if row['case'].startswith('cube_'):
                # The original calibrator retains the uncalibrated source as
                # best.pt when none of its updates improves validation RMSE.
                row['private_encoder_selected_step']=ck.get('calibration',{}).get('step',0)
                row['private_encoder_initial_fallback']='calibration' not in ck
                actor_path=Path('ablation_results/training')/row['case']/'best.pt'
                actor=torch.load(actor_path,map_location='cpu',weights_only=False)
                assert state_hash(ck['addon'])==state_hash(actor['addon']),('Calibration changed actor',row['case'])
    frozen=[]
    for name,path in [('tworoom_original','artifacts/final/base_policy.pt'),('cube_original','cube/artifacts/final/base_policy.pt'),
                      ('cube_latest_existing','ablation_inputs/cube_latest_base.pt')]:
        frozen.append({'case':name,'source':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'scope':'Unchanged original; preserve its exact file bytes because adapter parent validation uses its SHA256.'})
    result={'selected_models':cases,'unchanged_original_bases':frozen,'selected_count':len(cases),
        'base_evaluation_binding_sha256':hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        'scope':'Authoritative original-file and tensor hashes for models used in the final reported tests. Superseded Cube attempts must not be substituted by pathname.'}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'selected_models':len(cases),'required_original_bases':len(frozen),'out':str(out)}))


if __name__=='__main__':main()
