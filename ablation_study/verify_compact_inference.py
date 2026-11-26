"""Check actual model loading and inference after optimizer/RNG compaction.

Synthetic CPU probes test export fidelity only, not task success or latent
semantics. Native benchmark results are neither rerun nor changed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from .evaluation import load_adapter
from .export_completed import SLOTS, state_hash


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def equal(a, b):
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b) and bool(torch.isfinite(a).all())


def load_base(path, environment, ema):
    if environment == 'tworoom':
        from tworoom_gan.evaluate import load_checkpoint
        return load_checkpoint(path, 'cpu', use_ema=ema)
    from cube.cube_gan.evaluate import load_models
    return load_models(path, 'cpu', ema=ema)


def probe(original, compact, row):
    environment = 'tworoom' if row['case'].startswith('tworoom_') else 'cube'
    is_skill = row['kind'] == 'independent_skill'
    n, horizon, obs_dim, action_dim = 8, 16, (10 if environment == 'tworoom' else 47), (2 if environment == 'tworoom' else 5)
    rng = torch.Generator().manual_seed(90210)
    state = torch.randn(n, obs_dim, generator=rng)
    z = torch.randn(n, 2, generator=rng)
    lengths = torch.tensor([16, 15, 12, 8, 16, 14, 10, 7])
    outcomes = {}
    if is_skill:
        a, ea, _ = load_adapter(original, row['required_original_base'], environment)
        b, eb, _ = load_adapter(compact, row['required_original_base'], environment)
        context = torch.rand(n, a.config.context_dim, generator=rng)
        u = torch.randn(n, a.config.new_z_dim, generator=rng)
        assert a.config == b.config
        assert getattr(a, 'execute_steps', 8) == getattr(b, 'execute_steps', 8)
        if environment == 'cube':
            state = a.base.obs_mean + state*a.base.obs_std
        else:
            state = 112 + state*30
        for gate in [0, 1]:
            outcomes['disabled' if gate == 0 else 'enabled'] = equal(
                a(state, z, u, context, gate=gate), b(state, z, u, context, gate=gate))
        outcomes['disabled_base_bypass'] = equal(b(state, z, u, context, gate=0), b.base(state, z))
        actions = a(state, z, u, context, gate=1)
        states = state[:, None].expand(-1, horizon, -1).clone()
        contexts = context[:, None].expand(-1, horizon, -1).clone()
        if environment == 'cube':
            contexts[:, :, 10] = torch.linspace(0, 1, horizon)
        outcomes['encoder'] = equal(ea(states, actions, contexts, z, lengths),
                                    eb(states, actions, contexts, z, lengths))
        if row['ablation_config'].get('lora_only'):
            outcomes['pure_private_disconnection'] = equal(actions, b(state, z, -u, context, gate=1))
            outcomes['pure_context_disconnection'] = equal(actions, b(state, z, u, 1-context, gate=1))
    else:
        for ema in [False, True]:
            a, ea, _ = load_base(original, environment, ema)
            b, eb, _ = load_base(compact, environment, ema)
            inputs = a.obs_mean + state*a.obs_std if environment == 'cube' else 112 + state*30
            actions = a(inputs, z)
            outcomes['generator_ema' if ema else 'generator_raw'] = equal(actions, b(inputs, z))
            states = inputs[:, None].expand(-1, horizon, -1).clone()
            outcomes['encoder_ema_inputs' if ema else 'encoder_raw_inputs'] = equal(
                ea(states, actions, lengths), eb(states, actions, lengths))
    assert all(outcomes.values()), (row['case'], outcomes)
    return outcomes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', required=True)
    p.add_argument('--exports', default='ablation_results/exports')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    inventory = json.loads(Path(args.inventory).read_text())
    assert inventory['selected_count'] == len(inventory['selected_models']) == 56
    available = {}
    for path in sorted(Path(args.exports).glob('completed_*/EXPORT_MANIFEST.json')):
        for row in json.loads(path.read_text())['models']:
            available[(row['original'], row['original_sha256'])] = (path.parent, row)
    torch.set_num_threads(1)
    results = []
    with torch.inference_mode():
        for model in inventory['selected_models']:
            original = Path(model['source'])
            assert sha(original) == model['original_source_sha256']
            root, entry = available[(model['source'], model['original_source_sha256'])]
            compact = root/entry['path']
            assert sha(compact) == entry['sha256']
            ck = torch.load(compact, map_location='cpu', weights_only=False)
            assert ck['export_provenance']['source_sha256'] == model['original_source_sha256']
            hashes = {key: state_hash(ck[key]) for key in SLOTS if key in ck}
            assert hashes == model['tensor_sha256'] == entry['tensor_sha256']
            checks = probe(original, compact, model)
            results.append({'case': model['case'], 'original_source_sha256': model['original_source_sha256'],
                            'compact_sha256': entry['sha256'], 'checks': checks})
            print(json.dumps({'checked': len(results), 'case': model['case'], 'passed': True}), flush=True)
    result = {'complete': True, 'selected_count': 56,
              'inventory_sha256': sha(args.inventory), 'cases': results,
              'scope': 'Actual loaders and bitwise action/encoder equality on fixed synthetic CPU probes. Export-fidelity check only; not an extra task benchmark or performance claim.',
              'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'device': 'cpu', 'threads': 1},
              'raw_episode_data_exported': False}
    out = Path(args.out)
    assert not out.exists(), 'Use a fresh output receipt'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'complete': True, 'models_checked': 56, 'out': str(out)}), flush=True)


if __name__ == '__main__':
    main()
