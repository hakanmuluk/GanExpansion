"""Export the original TwoRoom neural classes and fixed residual composition.

Only import wiring and unused evaluation helpers are removed. Neural class,
loader-function, bank-method and planner source parity is checked explicitly.
No checkpoint or native observation arrays are copied by this exporter.
"""
import argparse
import ast
import json
from pathlib import Path
import textwrap
from .export_cube_runtime import sha, extract, class_asts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.'); p.add_argument('--planning-dir', required=True)
    p.add_argument('--freeze', required=True); p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(); repo = Path(a.repo); planning = Path(a.planning_dir)
    out = Path(a.out); assert not out.exists(); frozen = json.loads(Path(a.freeze).read_text())
    assert sha(a.config) == frozen['configuration_sha256']['composition']
    paths = ['tworoom_gan/models.py', 'tworoom_gan/evaluate.py', 'tworoom_gan/env.py', 'tworoom_gan/style.py',
             'tworoom_addons/models.py', 'tworoom_addons/evaluate.py', 'tworoom_addons/geometry.py',
             'tworoom_composition/models.py']
    for name in paths:
        values = [v for k, v in frozen['source_sha256'].items() if k.endswith('/' + name)]
        assert len(values) == 1 and sha(repo / name) == values[0], name
    for name in ['unchanged_composition.py', 'inference_guard.py']:
        values = [v for k, v in frozen['source_sha256'].items() if k.endswith('/independent_composition/' + name)]
        assert len(values) == 1 and sha(planning / name) == values[0]
    source = {name: (repo / name).read_text() for name in paths}
    contents = {'style.py': source['tworoom_gan/style.py']}
    base = source['tworoom_gan/models.py'].replace('from .env import OBS_DIM, HORIZON', 'from .constants import OBS_DIM, HORIZON')
    assert class_asts(base) == class_asts(source['tworoom_gan/models.py'])
    contents['base_models.py'] = base
    adapter = source['tworoom_addons/models.py']
    for old, new in [('from tworoom_gan.evaluate import load_checkpoint', 'from .base_loading import load_checkpoint'),
                     ('from tworoom_gan.models import Discriminator as BaseDiscriminator', 'from .base_models import Discriminator as BaseDiscriminator'),
                     ('from tworoom_gan.style import canonical_state', 'from .style import canonical_state'),
                     ('from .geometry import CONTEXT_DIM', 'from .constants import CONTEXT_DIM')]:
        assert adapter.count(old) == 1; adapter = adapter.replace(old, new)
    assert class_asts(adapter) == class_asts(source['tworoom_addons/models.py'])
    contents['adapter_models.py'] = adapter
    constants = []
    for name, wanted in [('tworoom_gan/env.py', ['OBS_DIM', 'HORIZON']), ('tworoom_addons/geometry.py', ['CONTEXT_DIM'])]:
        for node in ast.parse(source[name]).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
                constants.append(ast.get_source_segment(source[name], node))
    assert len(constants) == 3; contents['constants.py'] = '\n'.join(constants) + '\n'
    contents['base_loading.py'] = ('import torch\nfrom .base_models import ModelConfig, Generator, TrajectoryEncoder\n\n' +
                                   extract(source['tworoom_gan/evaluate.py'], 'load_checkpoint') + '\n')
    contents['adapter_loading.py'] = ('import hashlib\nfrom pathlib import Path\nimport torch\nfrom .adapter_models import AdapterConfig, GatedGenerator, SkillEncoder\n\n' +
                                      extract(source['tworoom_addons/evaluate.py'], 'load') + '\n')
    cls = next(n for n in ast.parse(source['tworoom_composition/models.py']).body if isinstance(n, ast.ClassDef))
    wanted = ['__init__', 'reset_counts', 'count', 'pair', 'base', 'assert_frozen']
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(methods) == len(wanted)
    def method_text(n):
        lines = ast.get_source_segment(source['tworoom_composition/models.py'], n).splitlines()
        return '    ' + lines[0] + '\n' + textwrap.indent(textwrap.dedent('\n'.join(lines[1:])), '        ')
    contents['bank.py'] = ('import hashlib\nfrom pathlib import Path\nimport torch\nfrom torch import nn\nfrom .adapter_loading import load\nfrom .adapter_models import state_digest\n\nclass FrozenBank(nn.Module):\n' +
                            '\n\n'.join(method_text(n) for n in methods) + '\n')
    exported = next(n for n in ast.parse(contents['bank.py']).body if isinstance(n, ast.ClassDef))
    assert {n.name: ast.dump(n) for n in methods} == {n.name: ast.dump(n) for n in exported.body}
    for name, origin, function in [('base_loading.py', 'tworoom_gan/evaluate.py', 'load_checkpoint'),
                                   ('adapter_loading.py', 'tworoom_addons/evaluate.py', 'load')]:
        assert ast.dump(ast.parse(extract(contents[name], function))) == ast.dump(ast.parse(extract(source[origin], function)))
    for name in ['unchanged_composition.py', 'inference_guard.py']:
        contents[name] = (planning / name).read_text()
    contents['__init__.py'] = '''"""Original TwoRoom base/private adapters and fixed residual composition.

Supply measured state [N,10], own contexts ([N,12],[N,12]) and base input [N,2].
sample returns normalized actions [N,16,2]; execute 8, then re-observe.
No simulator, route, phase handoff or learned composition network is loaded.
"""
import hashlib
import json
from pathlib import Path
from .bank import FrozenBank
from .unchanged_composition import CompositionConfig, OriginalComposer
from .inference_guard import GuardedPolicy


def load_composer(base, keepout, waypoint, device="cpu", seed=241):
    root=Path(__file__).parent
    manifest=json.loads((root/"manifest.json").read_text())
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name,checksum in manifest["files"].items():assert digest(root/name)==checksum
    assert digest(root/"config.json")==manifest["configuration_sha256"]
    for path,key in [(base,"base"),(keepout,"keepout"),(waypoint,"waypoint")]:
        assert digest(path)==manifest["checkpoint_sha256"][key]
    assert seed==manifest["evaluation_seed"]
    bank=FrozenBank(keepout=keepout,waypoint=waypoint,base=base,device=device)
    config=CompositionConfig(**json.loads((root/"config.json").read_text()))
    assert config.merge=="residual" and config.alpha==1 and not config.optimize
    assert config.k_code==config.w_code==-.8
    policy=OriginalComposer(bank,config,seed);policy.execute_steps=8
    return GuardedPolicy(policy)
'''
    package = out / 'tworoom_action_composition'; package.mkdir(parents=True)
    for name, text in contents.items():
        ast.parse(text); (package / name).write_text(text)
    (package / 'config.json').write_bytes(Path(a.config).read_bytes())
    manifest = {'scope': __doc__, 'parent_freeze_sha256': sha(a.freeze),
        'checkpoint_sha256': {'base': frozen['models']['artifacts/final/base_policy.pt'],
                              'keepout': frozen['models']['addon_artifacts/final/keepout.pt'],
                              'waypoint': frozen['models']['addon_artifacts/final/waypoint.pt']},
        'evaluation_seed': 241, 'execute_steps': 8, 'new_neural_modules': 0,
        'original_neural_class_loader_bank_ast_parity': True,
        'source_provenance': {name: sha(repo / name) for name in paths},
        'planning_provenance': {name: sha(planning / name) for name in ['unchanged_composition.py', 'inference_guard.py']},
        'files': {name: sha(package / name) for name in contents}, 'configuration_sha256': sha(package / 'config.json')}
    (package / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'pyproject.toml').write_text('[build-system]\nrequires=["setuptools"]\nbuild-backend="setuptools.build_meta"\n\n[project]\nname="tworoom-action-composition"\nversion="0.1.0"\nrequires-python=">=3.9"\ndependencies=["torch","numpy"]\n\n[tool.setuptools.package-data]\ntworoom_action_composition=["*.json"]\n')
    print(json.dumps({'package': str(package), 'ast_parity': True, 'weights_copied': False}), flush=True)


if __name__ == '__main__':
    main()
