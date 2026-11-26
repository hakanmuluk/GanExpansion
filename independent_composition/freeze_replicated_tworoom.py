"""Freeze all three selected TwoRoom pairs before any fresh final scene draw."""
import argparse
import json
from pathlib import Path
import shutil
import time
from .legacy import digest_file
from .freeze_candidate import BOOTSTRAP


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    root=Path('independent_composition_artifacts');prototype=root/'frozen_candidates/tworoom_selected_v12'
    selected=json.loads((prototype/'selection.json').read_text())
    assert selected['inference']['support_weight']==.03
    def source_weights(seed):
        return [prototype/'models'/f'{skill}_{role}.pt' if seed==42 else
                root/f'tworoom_retrain_s{seed}_v12/models'/f'{role}s'/skill/'best.pt'
                for role in ['route','controller','support'] for skill in ['keepout','waypoint']]
    # Evaluation adds only a named domain-disabled ablation. Neural policy,
    # route/controller/domain forwards and all their inputs stay identical.
    for name in ['tworoom_path_policy.py','tworoom_paths.py','private_paths.py','path_support.py']:
        assert digest_file(Path(__file__).parent/name)==digest_file(prototype/'independent_composition'/name)
    validation=[]
    for seed in [42,43,44]:
        if seed!=42:
            directory=root/f'tworoom_retrain_s{seed}_v12';state=json.loads((directory/'status.json').read_text())
            assert all(j['status']=='complete' for j in state['jobs'])
        for kind in ['supported_easy','supported_conflict','coupled_ood']:
            if seed==42:directory=root/'tworoom_private_domain_validation_v12/results'/f'pointwise_executor1_domain0.03_{kind}'
            else:directory=root/f'tworoom_retrain_s{seed}_v12/results'/kind
            assert (directory/'DONE.json').exists()
            summary=json.loads((directory/'summary.json').read_text());config=json.loads((directory/'config.json').read_text())
            proof=json.loads((directory/'DONE.json').read_text())
            assert proof['models_unchanged'] and proof['inference_tripwires_enabled']
            assert proof['inference_environment_queries']==0 and proof['new_shared_learned_networks']==0
            validated=[entry['route']['sha256'] for entry in config['models']]
            validated += [entry['controller']['sha256'] for entry in config['models']]
            validated += [entry['sha256'] for entry in config['supports']]
            assert [digest_file(p) for p in source_weights(seed)]==validated,'Validated weights changed before freeze'
            assert summary['episodes']==112
            assert config['args']['support_weight']==.03 and config['args']['executor']==1
            assert config['args']['agreement']=='pointwise' and config['args']['seed']==241
            validation.append({'seed':seed,'kind':kind,'path':str(directory),
                'summary_sha256':digest_file(directory/'summary.json'),'task_sha256':config['task_sha256'],**summary})
    for kind,target in [('supported_easy',.9),('supported_conflict',.9),('coupled_ood',.8)]:
        rows=[r for r in validation if r['kind']==kind]
        assert len(rows)==3 and len({r['task_sha256'] for r in rows})==1
        assert sum(r['success'] for r in rows)/3>=target,(kind,rows)
    (out/'independent_composition').mkdir(parents=True);(out/'models').mkdir();sources={}
    for file in Path(__file__).parent.iterdir():
        if file.is_file() and file.suffix in ['.py','.json']:
            target=out/'independent_composition'/file.name;shutil.copy2(file,target);sources[file.name]=digest_file(target)
    (out/'run_frozen.py').write_text(BOOTSTRAP);pairs=[]
    for seed in [42,43,44]:
        copied=[];originals=[]
        for role in ['route','controller','support']:
            for skill in ['keepout','waypoint']:
                if seed==42:path=prototype/'models'/f'{skill}_{role}.pt'
                else:
                    directory=root/f'tworoom_retrain_s{seed}_v12/models'/f'{role}s'/skill
                    assert (directory/'DONE.json').exists();path=directory/'best.pt'
                target=out/'models'/f's{seed}_{skill}_{role}.pt';shutil.copy2(path,target)
                assert digest_file(target)==digest_file(path)
                copied.append(str(target));originals.append(str(path))
        pairs.append({'training_seed':seed,'ordered_weight_paths':copied,
            'ordered_weight_sha256':[digest_file(p) for p in copied],'original_paths':originals})
    protocol={'environment':'tworoom','created':time.time(),'source_sha256':sources,
        'bootstrap_sha256':digest_file(out/'run_frozen.py'),'model_pairs':pairs,
        'base_sha256':digest_file('artifacts/final/base_policy.pt'),'validation_before_freeze':validation,
        'inference_configuration':{'iterations':120,'starts':8,'executor':1,'execute':1,'prior':.0001,'lr':.12,
            'guard_inference':True,'agreement':'pointwise','cycle_mode':'none','cycle_weight':0.,'code_bound':3.,'support_weight':.03},
        'methods':['compose','agreement','fixed','keepout','waypoint','base'],
        'method_overrides':{'agreement':{'effective_support_weight':0.,'meaning':'Same frozen route models/controllers, matched code-optimization budget, learned domain terms disabled.'}},
        'evaluation_seed':7141,'fresh_scene_seed':1732500001,'scene_count':64,'minimum_scenes_per_stratum':64,
        'scope':'All three retraining pairs retained. Source, eighteen weights and complete inference recipe frozen before fresh final scene construction. No test-dependent model selection.',
        'evaluation_plan':'64 fresh scenes per stratum shared across pairs. Compose, domain-disabled agreement, fixed route mean and each individual skill for all pairs; identical original frozen base once. Scene-clustered uncertainty.',
        'scene_construction':'The original frozen base supplies a keepout-region construction coordinate, without outcome rejection. Geometry difficulty and independent native expert feasibility select scenes. No composition-model outcome filters.',
        'control_rate_note':'New path-following adapters use one observed control per call. The original base forward remains exact when disabled; original published execute8 benchmarks remain separate.',
        'privacy':'Raw scenes, witnesses, actions and trajectories stay on Colab. Back up only code, model weights and aggregate experiment metadata.'}
    dependencies=['tworoom_gan/env.py','tworoom_gan/models.py','tworoom_gan/evaluate.py','tworoom_gan/style.py',
        'tworoom_addons/data.py','tworoom_addons/models.py','tworoom_addons/geometry.py',
        'tworoom_composition/data.py','tworoom_composition/planner.py','tworoom_composition/geometry.py',
        'tworoom_composition/audit.py','tworoom_composition/evaluate.py','tworoom_composition/runtime.py']
    protocol['repository_dependencies']={name:digest_file(name) for name in dependencies}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'capsule':str(out),'pairs':3,'weights':18,'protocol_sha256':digest_file(out/'protocol.json')}))


if __name__=='__main__':main()
