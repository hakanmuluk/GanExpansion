"""Export the validated original TwoRoom networks with the Cube latent optimizer.

Reuses the previously verified pure-PyTorch original neural classes and loaders.
No weights, native observations, raw arrays or new neural classes are generated.
"""
import argparse
import ast
import json
from pathlib import Path
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser()
    for name in ['previous-runtime','planning-dir','freeze','out']:p.add_argument('--'+name,required=True)
    a=p.parse_args();previous=Path(a.previous_runtime)/'tworoom_action_composition'
    planning=Path(a.planning_dir);out=Path(a.out);assert not out.exists()
    frozen=json.loads(Path(a.freeze).read_text());old=json.loads((previous/'manifest.json').read_text())
    for name,checksum in old['files'].items():assert digest_file(previous/name)==checksum
    for name,checksum in old['source_provenance'].items():assert frozen['original_sources'][name]==checksum
    cube=(planning/'amplitude_composition.py').read_text()
    generic=(planning/'latent_action_composition.py').read_text()
    expected=cube.replace("assert c.merge in ['keepout', 'waypoint', 'blend']",
                          "assert c.merge in ['keepout', 'waypoint', 'blend', 'residual']")
    expected=expected.replace('Base route signs and amplitude inputs stay fixed.',
        'Base route signs and speed inputs stay fixed. This generic copy also permits\nthe existing residual action executor; the optimizer is otherwise unchanged.')
    assert generic==expected,'Optimizer must match Cube except for permitting the existing residual executor'
    names=['style.py','base_models.py','adapter_models.py','constants.py','base_loading.py',
           'adapter_loading.py','bank.py','unchanged_composition.py','inference_guard.py']
    contents={name:(previous/name).read_text() for name in names}
    for name in ['unchanged_composition.py','inference_guard.py']:
        assert contents[name]==(planning/name).read_text()
    for name in ['amplitude_composition.py','latent_action_composition.py']:
        assert digest_file(planning/name)==frozen['source_sha256'][name]
        contents[name]=(planning/name).read_text()
    contents['__init__.py']='''"""Frozen original TwoRoom base/adapters, latent agreement and residual execution.

Input: measured state[N,10], own contexts([N,12],[N,12]), base code[N,2].
Output:16normalized2D actions; execute8, reobserve; retain policy within episode.
No simulator, task-score network or encoder is used for planning.
"""
import hashlib
import json
from pathlib import Path
from .bank import FrozenBank
from .latent_action_composition import AmplitudeConfig,AmplitudeComposer
from .inference_guard import GuardedPolicy


def load_composer(base,keepout,waypoint,device='cpu',seed=241):
    root=Path(__file__).parent
    manifest=json.loads((root/'manifest.json').read_text())
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name,checksum in manifest['files'].items():assert digest(root/name)==checksum
    assert digest(root/'config.json')==manifest['configuration_sha256']
    for path,key in [(base,'base'),(keepout,'keepout'),(waypoint,'waypoint')]:
        assert digest(path)==manifest['checkpoint_sha256'][key]
    assert seed==manifest['evaluation_seed']
    bank=FrozenBank(keepout=keepout,waypoint=waypoint,base=base,device=device)
    config=AmplitudeConfig(**json.loads((root/'config.json').read_text()))
    assert config.merge=='residual' and config.alpha==1 and config.optimize
    assert config.keepout_sign==config.waypoint_sign==-1
    assert config.agreement_horizon==8 and config.steps==20 and config.amplitude_tether==1
    policy=AmplitudeComposer(bank,config,seed);policy.execute_steps=8
    return GuardedPolicy(policy)
'''
    package=out/'tworoom_latent_composition';package.mkdir(parents=True)
    for name,text in contents.items():ast.parse(text);(package/name).write_text(text)
    config=Path(a.freeze).parent/'latent_residual.json'
    assert digest_file(config)==frozen['configuration_sha256']['latent_residual']
    (package/'config.json').write_bytes(config.read_bytes())
    models=frozen['models'];manifest={'scope':__doc__,'parent_freeze_sha256':digest_file(a.freeze),
        'checkpoint_sha256':{'base':models['artifacts/final/base_policy.pt'],
                            'keepout':models['addon_artifacts/final/keepout.pt'],
                            'waypoint':models['addon_artifacts/final/waypoint.pt']},
        'evaluation_seed':241,'execute_steps':8,'new_neural_modules':0,
        'original_neural_class_loader_bank_ast_parity':True,
        'previous_verified_runtime_manifest_sha256':digest_file(previous/'manifest.json'),
        'copied_original_runtime_files':{name:digest_file(previous/name) for name in names},
        'same_cube_optimizer_except_residual_executor_allowed':True,
        'planning_provenance':{name:digest_file(planning/name) for name in
             ['amplitude_composition.py','latent_action_composition.py','unchanged_composition.py','inference_guard.py']},
        'files':{name:digest_file(package/name) for name in contents},
        'configuration_sha256':digest_file(package/'config.json')}
    (package/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'pyproject.toml').write_text('[build-system]\nrequires=["setuptools"]\nbuild-backend="setuptools.build_meta"\n\n[project]\nname="tworoom-latent-composition"\nversion="0.1.0"\nrequires-python=">=3.9"\ndependencies=["torch","numpy"]\n\n[tool.setuptools.package-data]\ntworoom_latent_composition=["*.json"]\n')
    print(json.dumps({'package':str(package),'original_networks_unchanged':True,'same_cube_optimizer':True}),flush=True)


if __name__=='__main__':main()
