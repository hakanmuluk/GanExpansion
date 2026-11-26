"""Export unchanged original neural classes and planning code without MuJoCo.

Only import wiring and unused evaluation dependencies are removed. Class/function
AST parity is checked. This packages a frozen research candidate; export is not
evidence of task success. Weights and raw observations are not copied.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(source, name):
    tree = ast.parse(source)
    node = next(n for n in tree.body if getattr(n, 'name', None) == name)
    return ast.get_source_segment(source, node)


def class_asts(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.')
    p.add_argument('--freeze', required=True); p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(); repo = Path(a.repo); out = Path(a.out)
    assert not out.exists(); frozen = json.loads(Path(a.freeze).read_text())
    assert sha(a.config) == frozen['configuration_sha256']
    paths = ['cube/cube_gan/models.py', 'cube/cube_gan/evaluate.py', 'cube_addons/models.py',
             'cube_addons/evaluate.py', 'cube_addons/geometry.py', 'cube_composition/models.py']
    for name in paths:
        assert sha(repo / name) == frozen['original_dependencies'][name]
    planning = ['cube_checkpoint_bank.py', 'unchanged_composition.py',
                'amplitude_composition.py', 'inference_guard.py']
    for name in planning:
        assert sha(repo / 'independent_composition' / name) == frozen['source_sha256'][name]
    package = out / 'cube_action_composition'; package.mkdir(parents=True)
    sources = {name: (repo / name).read_text() for name in paths}
    contents = {'base_models.py': sources['cube/cube_gan/models.py']}
    adapter = sources['cube_addons/models.py']
    rewires = [('from cube.cube_gan.evaluate import load_models', 'from .base_loading import load_models'),
               ('from cube.cube_gan.models import Normalized, Discriminator', 'from .base_models import Normalized, Discriminator'),
               ('from .geometry import CONTEXT_DIM, MAX_STEPS', 'from .constants import CONTEXT_DIM, MAX_STEPS')]
    for old, new in rewires:
        assert adapter.count(old) == 1; adapter = adapter.replace(old, new)
    assert class_asts(adapter) == class_asts(sources['cube_addons/models.py'])
    contents['adapter_models.py'] = adapter
    constants = []
    for node in ast.parse(sources['cube_addons/geometry.py']).body:
        if isinstance(node, ast.Assign) and any(isinstance(v, ast.Name) and v.id in ['CONTEXT_DIM', 'MAX_STEPS'] for v in node.targets):
            constants.append(ast.get_source_segment(sources['cube_addons/geometry.py'], node))
    assert len(constants) == 2
    contents['constants.py'] = '\n'.join(constants) + '\n'
    contents['base_loading.py'] = ('import torch\nfrom .base_models import ModelConfig, Generator, TrajectoryEncoder\n\n' +
                                   extract(sources['cube/cube_gan/evaluate.py'], 'load_models') + '\n')
    contents['adapter_loading.py'] = ('import hashlib\nfrom pathlib import Path\nimport torch\nfrom .adapter_models import AdapterConfig, GatedGenerator, SkillEncoder\n\n' +
                                      extract(sources['cube_addons/evaluate.py'], 'load_adapter') + '\n')
    old_bank = next(n for n in ast.parse(sources['cube_composition/models.py']).body if isinstance(n, ast.ClassDef))
    names = ['reset_counts', 'count', 'base', 'assert_frozen']
    methods = [ast.get_source_segment(sources['cube_composition/models.py'], n) for n in old_bank.body
               if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    # ast.get_source_segment retains indentation after the first method line.
    # Normalize each selected method before inserting it under the new class.
    import textwrap
    def indent_method(method):
        lines = method.splitlines()
        body = textwrap.dedent('\n'.join(lines[1:]))
        return '    ' + lines[0] + '\n' + textwrap.indent(body, '        ')
    contents['bank_base.py'] = ('from torch import nn\nfrom .adapter_models import state_digest\n\nclass FrozenBank(nn.Module):\n' +
                               '\n\n'.join(indent_method(method) for method in methods) + '\n')
    bank = (repo / 'independent_composition/cube_checkpoint_bank.py').read_text()
    for old, new in [('from cube_composition.models import FrozenBank', 'from .bank_base import FrozenBank'),
                     ('from cube_addons.evaluate import load_adapter', 'from .adapter_loading import load_adapter'),
                     ('from cube_addons.models import state_digest', 'from .adapter_models import state_digest')]:
        assert bank.count(old) == 1; bank = bank.replace(old, new)
    assert class_asts(bank) == class_asts((repo / 'independent_composition/cube_checkpoint_bank.py').read_text())
    contents['cube_checkpoint_bank.py'] = bank
    contents['legacy.py'] = 'import hashlib\nfrom pathlib import Path\n\ndef digest_file(path):\n    return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n'
    for name in ['unchanged_composition.py', 'amplitude_composition.py', 'inference_guard.py']:
        contents[name] = (repo / 'independent_composition' / name).read_text()
    contents['__init__.py'] = '''"""Original Cube base and adapters; PyTorch-only composition runtime.

Supply measured state [N,47], own contexts ([N,20],[N,20]), base input [N,2].
sample returns original normalized actions [N,16,5]. Execute the first 8,
threshold gripper to +/-1 as in the original controller, then re-observe.
Load a new composer at each episode boundary. No environment is constructed.
"""
import json
from pathlib import Path
from .legacy import digest_file
from .cube_checkpoint_bank import CheckpointBank
from .amplitude_composition import AmplitudeConfig, AmplitudeComposer
from .inference_guard import GuardedPolicy


def load_composer(base, keepout, waypoint, device="cpu", seed=241):
    root = Path(__file__).parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        assert digest_file(root / name) == checksum, ("Exported source changed", name)
    assert digest_file(root / "config.json") == manifest["configuration_sha256"]
    expected = manifest["checkpoint_sha256"]
    for path, key in [(base, "base"), (keepout, "keepout"), (waypoint, "waypoint")]:
        assert digest_file(path) == expected[key], ("Unexpected checkpoint", key)
    assert seed == manifest["evaluation_seed"]
    bank = CheckpointBank(keepout, waypoint, base, device)
    config = AmplitudeConfig(**json.loads((root / "config.json").read_text()))
    assert config.merge == "waypoint" and config.agreement_horizon == 8
    composer = AmplitudeComposer(bank, config, seed)
    composer.execute_steps = 8
    return GuardedPolicy(composer)
'''
    for name, text in contents.items():
        ast.parse(text); (package / name).write_text(text)
    assert class_asts(contents['base_models.py']) == class_asts(sources['cube/cube_gan/models.py'])
    for name, origin in [('base_loading.py', 'cube/cube_gan/evaluate.py'), ('adapter_loading.py', 'cube_addons/evaluate.py')]:
        function = 'load_models' if name.startswith('base_') else 'load_adapter'
        assert ast.dump(ast.parse(extract(contents[name], function))) == ast.dump(ast.parse(extract(sources[origin], function)))
    original_methods = {n.name: ast.dump(n) for n in old_bank.body if isinstance(n, ast.FunctionDef) and n.name in names}
    new_bank = next(n for n in ast.parse(contents['bank_base.py']).body if isinstance(n, ast.ClassDef))
    assert original_methods == {n.name: ast.dump(n) for n in new_bank.body}
    (package / 'config.json').write_bytes(Path(a.config).read_bytes())
    manifest = {'scope': __doc__, 'parent_freeze_sha256': sha(a.freeze),
        'checkpoint_sha256': {'base': frozen['base_sha256'], 'keepout': frozen['model_pairs']['prefix'][0],
                              'waypoint': frozen['model_pairs']['prefix'][1]},
        'evaluation_seed': frozen['evaluation_seed'], 'execute_steps': frozen['execute'],
        'source_provenance': {name: sha(repo / name) for name in paths},
        'planning_provenance': {name: sha(repo / 'independent_composition' / name) for name in planning},
        'class_function_method_ast_parity': True, 'new_neural_modules': 0,
        'simulator_or_project_packages_required': False, 'task_success_not_proven_by_export': True,
        'files': {name: sha(package / name) for name in contents}, 'configuration_sha256': sha(package / 'config.json')}
    (package / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'pyproject.toml').write_text('[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "cube-action-composition"\nversion = "0.1.0"\nrequires-python = ">=3.9"\ndependencies = ["torch", "numpy"]\n\n[tool.setuptools.package-data]\ncube_action_composition = ["*.json"]\n')
    print(json.dumps({'package': str(package), 'ast_parity': True, 'weights_copied': False}), flush=True)


if __name__ == '__main__':
    main()
