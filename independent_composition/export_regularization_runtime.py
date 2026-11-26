"""Export regularization around the existing verified pure-PyTorch model banks.

The added package contains planning/loading code only. Original neural classes
come from the already verified Cube/TwoRoom inference wheels. No new network,
checkpoint, observation array or native simulator dependency is exported here.
"""
import argparse
import ast
import json
from pathlib import Path
from .regularization_models import digest_file


LOADER = '''"""Original action models plus private Gaussian/cycle latent regularization."""
import importlib
import json
from pathlib import Path
from .regularization_models import digest_file,load_cycle_encoders
from .regularized_latent_composition import RegularizedConfig,RegularizedComposer
from .inference_guard import GuardedPolicy


def load_composer(environment,base,keepout,waypoint,device='cpu',regularizer='selected',encoders=None,seed=241):
    root=Path(__file__).parent
    manifest=json.loads((root/'manifest.json').read_text())
    for name,checksum in manifest['files'].items():assert digest_file(root/name)==checksum, name
    settings=manifest['environments'][environment]
    if regularizer=='selected':regularizer=settings['default_regularizer']
    assert regularizer in settings['configurations'],regularizer
    assert seed==settings['evaluation_seed']
    config_name=settings['configurations'][regularizer]
    assert digest_file(root/config_name)==manifest['configuration_sha256'][config_name]
    expected=settings['role_model_sha256']
    for name,path in [('base',base),('keepout',keepout),('waypoint',waypoint)]:
        assert digest_file(path)==expected[name],('Unexpected checkpoint',name)
    package_name='cube_action_composition' if environment=='cube' else 'tworoom_latent_composition'
    old_package=importlib.import_module(package_name);old_root=Path(old_package.__file__).parent
    old=json.loads((old_root/'manifest.json').read_text())
    assert old['checkpoint_sha256']==expected
    for name,checksum in old['files'].items():assert digest_file(old_root/name)==checksum,name
    if environment=='cube':
        from cube_action_composition.cube_checkpoint_bank import CheckpointBank as Bank
    else:
        from tworoom_latent_composition.bank import FrozenBank as Bank
    bank=Bank(keepout=keepout,waypoint=waypoint,base=base,device=device)
    bank.original_model_sources=expected
    config=RegularizedConfig(**json.loads((root/config_name).read_text()))
    if config.cycle_weight:
        paths=encoders or [str(Path(keepout).parent/(skill+'_encoder.pt')) for skill in ['keepout','waypoint']]
        for skill,path in zip(['keepout','waypoint'],paths):
            assert digest_file(path)==settings['cycle_encoder_sha256'][skill],skill
        load_cycle_encoders(bank,environment,paths,config.cycle_horizon)
    else:assert not encoders
    policy=RegularizedComposer(bank,config,seed);policy.execute_steps=8
    return GuardedPolicy(policy)
'''


def main():
    p=argparse.ArgumentParser()
    for name in ['cube-freeze','tworoom-freeze','planning-dir','out']:p.add_argument('--'+name,required=True)
    a=p.parse_args();planning=Path(a.planning_dir);out=Path(a.out);assert not out.exists()
    sources=['regularized_latent_composition.py','latent_action_composition.py','unchanged_composition.py','inference_guard.py']
    contents={name:(planning/name).read_text() for name in sources}
    helpers=(planning/'regularization_models.py').read_text();tree=ast.parse(helpers)
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['digest_file','state_digest','load_cycle_encoders']]
    assert len(selected)==3
    contents['regularization_models.py']='import hashlib\nfrom pathlib import Path\nimport torch\n\n'+'\n\n'.join(ast.get_source_segment(helpers,n) for n in selected)+'\n'
    contents['__init__.py']=LOADER
    package=out/'skill_latent_regularization';package.mkdir(parents=True)
    environments={};configs={}
    for env,path in [('cube',a.cube_freeze),('tworoom',a.tworoom_freeze)]:
        frozen=json.loads(Path(path).read_text());assert frozen['environment']==env
        for name in sources:
            assert digest_file(planning/name)==frozen['source_files'][str((planning/name).resolve())]
        assert digest_file(planning/'regularization_models.py')==frozen['source_files'][str((planning/'regularization_models.py').resolve())]
        names={}
        for method in ['baseline','prior','prior_cycle']:
            src=Path(path).parent/(method+'.json');assert digest_file(src)==frozen['configuration_sha256'][method]
            name=env+'_'+method+'.json';(package/name).write_bytes(src.read_bytes());names[method]=name;configs[name]=digest_file(src)
        environments[env]={'role_model_sha256':frozen['role_model_sha256'],
            'cycle_encoder_sha256':frozen['cycle_encoder_sha256'],'evaluation_seed':241,
            'default_regularizer':'prior','configurations':names,'parent_freeze_sha256':digest_file(path),
            'executor':'original waypoint output' if env=='cube' else 'original residual A_W+A_K-A_B'}
    for name,text in contents.items():ast.parse(text);(package/name).write_text(text)
    manifest={'scope':__doc__,'environments':environments,'configuration_sha256':configs,
        'new_neural_modules':0,'existing_generators_unchanged':True,'private_cycle_encoder_architecture_unchanged':True,
        'helper_functions_AST_identical':True,'planning_source_sha256':{name:digest_file(planning/name) for name in sources},
        'files':{name:digest_file(package/name) for name in contents}}
    (package/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'pyproject.toml').write_text('[build-system]\nrequires=["setuptools"]\nbuild-backend="setuptools.build_meta"\n\n[project]\nname="skill-latent-regularization"\nversion="0.1.0"\nrequires-python=">=3.9"\ndependencies=["torch","numpy","cube-action-composition==0.1.0","tworoom-latent-composition==0.1.0"]\n\n[tool.setuptools.package-data]\nskill_latent_regularization=["*.json"]\n')
    print(json.dumps({'package':str(package),'new_neural_modules':0,'default':'private Gaussian prior','optional':'private Gaussian plus existing-encoder cycle'}),flush=True)


if __name__=='__main__':main()
