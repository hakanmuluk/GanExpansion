"""Paired native evaluation with common random starts within each world.

Physics, waypoint, goal and base style must be identical within a pair. Only
keepout differs. Matching solver seeds prevents random restarts from masquerading
as a conditional response. Native simulation is outside guarded policy inference.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .cube_checkpoint_bank import CheckpointBank
from .direction_composition import DirectionConfig, DirectionComposer
from .quantile_composition import QuantileConfig, QuantileComposer
from .unchanged_composition import CompositionConfig, OriginalComposer
from .evaluate_diffusion import evaluate_cube, summarize
from .inference_guard import GuardedPolicy
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['tasks', 'config', 'out', 'base', 'keepout', 'waypoint']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--metric', choices=['mse', 'direction', 'quantile'], default='mse')
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--execute', type=int, default=8)
    p.add_argument('--seed', type=int, default=241)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    assert not Path(args.tasks).name.startswith(('test', 'final'))
    torch.set_num_threads(2)
    with np.load(args.tasks) as data:
        n = min(args.count, len(data['paired_id']))
        assert n > 0 and n % 2 == 0
        pairs = data['paired_id'][:n].astype(int)
        assert np.array_equal(pairs[::2], pairs[1::2])
        assert len(np.unique(pairs)) == n // 2
        for key in ['initial_state', 'initial_cube', 'goal', 'base_z', 'waypoint']:
            assert np.array_equal(data[key][:n:2], data[key][1:n:2]), key
        assert all(not np.array_equal(data['keepout'][i], data['keepout'][i + 1]) for i in range(0, n, 2))
    out = Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    config_type, policy_type = {'mse': (CompositionConfig, OriginalComposer),
        'direction': (DirectionConfig, DirectionComposer),
        'quantile': (QuantileConfig, QuantileComposer)}[args.metric]
    cfg = config_type(**json.loads(Path(args.config).read_text()))
    bank = CheckpointBank(args.keepout, args.waypoint, args.base, args.device)
    names = ['evaluate_paired_composition.py', 'direction_composition.py', 'quantile_composition.py', 'unchanged_composition.py',
             'cube_checkpoint_bank.py', 'evaluate_diffusion.py', 'inference_guard.py']
    source = {name: digest_file(Path(__file__).parent / name) for name in names}
    (out / 'source').mkdir()
    for name in names:
        shutil.copy2(Path(__file__).parent / name, out / 'source' / name)
    (out / 'config.json').write_text(json.dumps({'args': vars(args), 'composition': asdict(cfg),
        'models': bank.files, 'source_sha256': source, 'task_sha256': digest_file(args.tasks),
        'scope': __doc__, 'same_seed_within_pair': True,
        'pair_policy_seeds': [args.seed + 9973 * int(x) for x in pairs]}, indent=2) + '\n')
    cursor = iter(pairs)

    def factory(args, unused, base, ignored_seed):
        seed = args.seed + 9973 * int(next(cursor))
        return GuardedPolicy(policy_type(bank, cfg, seed))

    started = time.time()
    rows = evaluate_cube(args, [], bank.k.base, out, factory)
    assert len(rows) == n
    metrics = summarize(rows)
    metrics['total_seconds'] = time.time() - started
    metrics['paired_worlds'] = n // 2
    metrics['both_variants_successful_worlds'] = sum(rows[i]['success'] and rows[i + 1]['success'] for i in range(0, n, 2))
    metrics['inner_successes'] = sum(r['success'] for r in rows[::2])
    metrics['outer_successes'] = sum(r['success'] for r in rows[1::2])
    bank.assert_frozen()
    assert all(digest_file(path) == checksum for path, checksum in bank.files.items())
    (out / 'summary.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'episodes': n, 'weights_unchanged': True,
        'inference_environment_queries': 0, 'inference_tripwires_enabled': True,
        'new_learned_modules': 0, 'common_random_starts_verified': True}) + '\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
