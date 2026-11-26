"""Freeze a selected original-model TwoRoom latent search before new final scenes."""
import argparse
import json
from pathlib import Path
import shutil
import time
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);assert not out.exists()
    root=Path('independent_composition_artifacts')
    dev=root/'tworoom_latent_v135_protocol'
    status=json.loads((dev/'status.json').read_text())
    assert status['status']=='complete' and all(j['status']=='complete' for j in status['jobs'])
    validation={}
    for job in status['jobs']:
        path=Path(job['out']);done=json.loads((path/'DONE.json').read_text())
        assert done['weights_frozen'] and done['inference_environment_queries']==0
        assert done['new_learned_modules']==0 and done['inference_tripwires_enabled']
        validation[job['id']]=json.loads((path/'summary.json').read_text())
    assert all(validation['latent_residual_'+kind]['success']>=.85
               for kind in ['supported_easy','supported_conflict'])
    out.mkdir(parents=True)
    for name in ['latent_residual','coarse_only','fixed_residual']:
        shutil.copy2(dev/(name+'.json'),out/(name+'.json'))
    planning=['evaluate_tworoom_latent_frozen.py','latent_action_composition.py',
              'amplitude_composition.py','unchanged_composition.py','inference_guard.py',
              'evaluate_diffusion.py','generate_final_tworoom.py','audit_tworoom_latent_native.py']
    originals=['tworoom_gan/models.py','tworoom_gan/env.py','tworoom_gan/evaluate.py','tworoom_gan/style.py',
               'tworoom_addons/models.py','tworoom_addons/evaluate.py','tworoom_addons/data.py','tworoom_addons/geometry.py',
               'tworoom_composition/models.py','tworoom_composition/data.py','tworoom_composition/geometry.py',
               'tworoom_composition/planner.py','tworoom_composition/audit.py','tworoom_composition/evaluate.py']
    models=['artifacts/final/base_policy.pt','addon_artifacts/final/keepout.pt','addon_artifacts/final/waypoint.pt']
    protocol={'environment':'tworoom','created':time.time(),
              'scope':'Original base and independent adapters, no retraining. Same Cube private+base-magnitude latent optimizer; preserve negative private-code branch, stronger base-input tether1. Original residual output executor, explicitly different from Cube waypoint-only execution. No new neural network, task score, simulator or encoder in inference.',
              'models':{path:digest_file(path) for path in models},'base_sha256':digest_file(models[0]),
              'source_sha256':{name:digest_file(Path(__file__).parent/name) for name in planning},
              'original_sources':{path:digest_file(path) for path in originals},
              'configuration_sha256':{name:digest_file(out/(name+'.json')) for name in ['latent_residual','coarse_only','fixed_residual']},
              'development_results':validation,'selected_before_final_scene_generation':True,
              'evaluation_seed':241,'execute':8,'scene_count':64,'minimum_scenes_per_stratum':64,
              'fresh_scene_seed':1894500001,
              'scene_construction':'Original supported/context distributions, fresh seed families. Reserve128 expert-feasible cases perstratum; retain first64 independently native-certified cases. Frozen base defines originalsingle-skill placement distribution. No composition outcome selects cases.',
              'final_methods':['latent_residual','coarse_only','fixed_residual'],
              'interpretation':'Measure joint success, all three strata and controls. Zero-Adam still performs coarse neural code search; fixed residual does no code search. Do not claim Adam outperforms these controls unless outcomes establish it.'}
    (out/'freeze.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'protocol':str(out/'freeze.json'),'sha256':digest_file(out/'freeze.json')}),flush=True)


if __name__=='__main__':main()
