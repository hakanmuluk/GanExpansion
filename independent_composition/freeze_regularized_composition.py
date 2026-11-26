"""Freeze nonzero private-z regularization and controls before fresh final cases."""
import argparse
import json
from pathlib import Path
import shutil
import time
from .regularization_models import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--study',required=True);p.add_argument('--out',required=True)
    p.add_argument('--prior-method',default='prior01');p.add_argument('--combined-method',default='prior_cycle01')
    p.add_argument('--scene-seed',type=int,required=True);a=p.parse_args()
    study=Path(a.study);out=Path(a.out);assert not out.exists()
    workers=list(study.glob('worker_*_status.json'));assert len(workers)==2
    validation={}
    for path in workers:
        status=json.loads(path.read_text());assert status['status']=='complete'
        for job in status['jobs']:
            assert job['status']=='complete';folder=Path(job['out'])
            proof=json.loads((folder/'DONE.json').read_text())
            assert proof['all_weights_frozen'] and proof['inference_environment_queries']==0 and proof['new_neural_modules']==0
            validation[job['id']]=json.loads((folder/'summary.json').read_text())
    for name in [a.prior_method,a.combined_method]:
        assert all(validation[name+'_'+kind]['success']>=.8 for kind in ['supported_easy','supported_conflict'])
    selected={'baseline':'baseline','prior':a.prior_method,'prior_cycle':a.combined_method}
    out.mkdir(parents=True)
    for target,source in selected.items():shutil.copy2(study/(source+'.json'),out/(target+'.json'))
    prior=json.loads((out/'prior.json').read_text());combined=json.loads((out/'prior_cycle.json').read_text())
    assert prior['private_prior_weight']>0 and not prior['cycle_weight']
    assert combined['private_prior_weight']>0 and combined['cycle_weight']>0 and combined['cycle_horizon']==16
    if a.env=='tworoom':
        model_paths={'base':'artifacts/final/base_policy.pt','keepout':'addon_artifacts/final/keepout.pt','waypoint':'addon_artifacts/final/waypoint.pt'}
        originals=['tworoom_gan/env.py','tworoom_gan/models.py','tworoom_gan/style.py','tworoom_gan/evaluate.py',
                   'tworoom_addons/models.py','tworoom_addons/geometry.py','tworoom_addons/evaluate.py',
                   'tworoom_composition/models.py','tworoom_composition/data.py','tworoom_composition/geometry.py',
                   'tworoom_composition/planner.py','tworoom_composition/audit.py','tworoom_composition/evaluate.py']
    else:
        model_paths={'base':'cube/artifacts/final/base_policy.pt','keepout':'cube_addon_artifacts/final/keepout_s42/adapter.pt','waypoint':'cube_addon_artifacts/final/waypoint_s42/adapter.pt'}
        originals=['cube/cube_gan/env.py','cube/cube_gan/models.py','cube/cube_gan/evaluate.py',
                   'cube_addons/models.py','cube_addons/geometry.py','cube_addons/evaluate.py','cube_addons/env.py','cube_addons/contacts.py',
                   'cube_composition/models.py','cube_composition/data.py','cube_composition/geometry.py','cube_composition/scene.py',
                   'cube_composition/rollout.py','cube_composition/planner.py']
    art=Path('independent_composition_artifacts').resolve()
    encoders={skill:str(art/('private_encoder_v140_'+a.env+'_'+skill)/'best.pt') for skill in ['keepout','waypoint']}
    original_models={path:digest_file(path) for path in model_paths.values()}
    encoder_models={path:digest_file(path) for path in encoders.values()}
    planning=['regularized_latent_composition.py','regularization_models.py','latent_action_composition.py',
              'unchanged_composition.py','inference_guard.py','evaluate_regularized_latents.py','evaluate_diffusion.py',
              'observation_replay.py','generate_final_tworoom.py','generate_final_cube.py','generate_cube_frozen_bank.py',
              'audit_tworoom_latent_native.py']
    source_files={str((Path(__file__).parent/name).resolve()):digest_file(Path(__file__).parent/name) for name in planning}
    source_files.update({str(Path(path).resolve()):digest_file(path) for path in originals})
    protocol={'environment':a.env,'created':time.time(),'scope':'Original base/adapters and original solver/executor; nonzero private Gaussian prior and optional existing-encoder cycle. Only same-architecture private encoders calibrated on separate own-skill observations. No inference simulator or joint-task score/new network.',
        'configuration_sha256':{name:digest_file(out/(name+'.json')) for name in selected},
        'selected_development_methods':selected,'development_results':validation,'final_methods':list(selected),
        'source_files':source_files,'original_sources':{path:digest_file(path) for path in originals},
        'models':{**original_models,**encoder_models},'method_models':{'baseline':original_models,'prior':original_models,'prior_cycle':{**original_models,**encoder_models}},
        'role_model_paths':model_paths,'role_model_sha256':{role:digest_file(path) for role,path in model_paths.items()},
        'encoder_paths':encoders,'cycle_encoder_sha256':{role:digest_file(path) for role,path in encoders.items()},
        'base_sha256':digest_file(model_paths['base']),'evaluation_seed':241,'execute':8,
        'scene_count':64,'minimum_scenes_per_stratum':64,'fresh_scene_seed':a.scene_seed,
        'scene_construction':'Original supported/context distributions, fresh seed family after freeze. Original geometry and independent native feasibility witnesses only; no evaluated composition outcome selects cases. Include coupled/OOD unchanged.',
        'interpretation':'Report all frozen methods on identical fresh cases. Encoder recovery and low regularizer values are not evidence of joint success. Keep hard-case failures and do not claim gains unless results establish them.'}
    (out/'freeze.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'freeze':str(out/'freeze.json'),'sha256':digest_file(out/'freeze.json')}),flush=True)


if __name__=='__main__':main()
