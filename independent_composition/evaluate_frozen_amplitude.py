"""Evaluate frozen original-generator composition on fresh certified Cube banks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .cube_checkpoint_bank import CheckpointBank
from .gate_ablation_bank import GateAblationBank
from .unchanged_composition import CompositionConfig, OriginalComposer
from .quantile_composition import QuantileConfig, QuantileComposer
from .speed_composition import SpeedConfig, SpeedComposer
from .mode_composition import ModeConfig, ModeComposer
from .shared_code_composition import SharedCodeComposer
from .evaluate_diffusion import evaluate_cube, summarize
from .inference_guard import GuardedPolicy
from .legacy import digest_file
from .amplitude_composition import AmplitudeConfig, AmplitudeComposer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['tasks', 'config', 'out', 'base', 'keepout', 'waypoint']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--backend', choices=['amplitude'], required=True)
    p.add_argument('--paired', action='store_true')
    p.add_argument('--paired-variable', choices=['keepout', 'waypoint'], default='keepout')
    p.add_argument('--keepout-gate', type=int, choices=[0, 1], default=1)
    p.add_argument('--waypoint-gate', type=int, choices=[0, 1], default=1)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--execute', type=int, default=8)
    p.add_argument('--seed', type=int, default=241)
    p.add_argument('--device', default='cuda')
    p.add_argument("--frozen-protocol", required=True)
    p.add_argument("--bank-manifest", required=True)
    args = p.parse_args()
    from .verify_frozen_amplitude import verify
    frozen_proof = verify(args)
    assert 1 <= args.execute <= 16
    torch.set_num_threads(2)
    with np.load(args.tasks) as data:
        n = min(args.count, len(data['seed']))
        pairs = data['paired_id'][:n].astype(int) if args.paired else np.arange(n)
        if args.paired:
            assert n % 2 == 0 and np.array_equal(pairs[::2], pairs[1::2])
            assert len(np.unique(pairs)) == n // 2
            fixed_region = 'waypoint' if args.paired_variable == 'keepout' else 'keepout'
            for key in ['initial_state', 'initial_cube', 'goal', 'base_z', fixed_region]:
                assert np.array_equal(data[key][:n:2], data[key][1:n:2]), key
    out = Path(args.out)
    assert n > 0 and not out.exists()
    out.mkdir(parents=True)
    config_type, policy_type = AmplitudeConfig, AmplitudeComposer
    cfg = config_type(**json.loads(Path(args.config).read_text()))
    assert args.keepout_gate or args.waypoint_gate, 'Use the fixed base control for both switches off'
    if not (args.keepout_gate and args.waypoint_gate):
        assert cfg.history == 0, 'These switch ablations use current-output agreement only'
    bank = GateAblationBank(args.keepout, args.waypoint, args.base, args.device,
        keepout_gate=args.keepout_gate, waypoint_gate=args.waypoint_gate)
    names = ['evaluate_amplitude.py', 'amplitude_composition.py', 'evaluate_existing_inputs.py', 'unchanged_composition.py', 'quantile_composition.py',
             'speed_composition.py', 'mode_composition.py', 'shared_code_composition.py', 'cube_checkpoint_bank.py', 'gate_ablation_bank.py', 'evaluate_diffusion.py', 'inference_guard.py', 'observation_replay.py']
    source = {name: digest_file(Path(__file__).parent / name) for name in names}
    names += ['evaluate_frozen_amplitude.py', 'verify_frozen_amplitude.py']
    (out / 'source').mkdir()
    for name in names:
        shutil.copy2(Path(__file__).parent / name, out / 'source' / name)
    source = {name: digest_file(Path(__file__).parent / name) for name in names}
    (out / 'frozen_proof.json').write_text(json.dumps(frozen_proof, indent=2) + '\n')
    (out / 'config.json').write_text(json.dumps({'args': vars(args), 'composition': asdict(cfg),
        'models': bank.files, 'source_sha256': source, 'task_sha256': digest_file(args.tasks),
        'scope': 'Original network classes and forward interfaces, all weights frozen. Existing inputs optimized only; no simulator, geometric task score, encoder or new learned module during inference.',
        'same_seed_within_pair': args.paired,
        'original_binary_gates': [args.keepout_gate, args.waypoint_gate],
        'one_common_semantic_code_for_two_independent_adapters': args.backend == 'shared',
        'reset_observation_protocol': 'Native derived contact force refreshed; full paired initial-observation equality asserted',
        'base_speed_inputs_may_change': args.backend == 'speed' or (args.backend == 'mode' and cfg.optimize_speed_inputs),
        'base_route_sign_may_change': False, 'base_amplitude_inputs_optimized': True}, indent=2) + '\n')
    cursor = iter(pairs)

    def factory(args, unused, base, ignored_seed):
        return GuardedPolicy(policy_type(bank, cfg, args.seed + 9973 * int(next(cursor))))

    started = time.time()
    rows = evaluate_cube(args, [], bank.k.base, out, factory)
    metrics = summarize(rows)
    metrics['total_seconds'] = time.time() - started
    if args.paired:
        metrics.update(paired_worlds=n // 2,
            both_variants_successful_worlds=sum(rows[i]['success'] and rows[i + 1]['success'] for i in range(0, n, 2)),
            inner_successes=sum(r['success'] for r in rows[::2]),
            outer_successes=sum(r['success'] for r in rows[1::2]))
    bank.assert_frozen()
    assert all(digest_file(path) == checksum for path, checksum in bank.files.items())
    (out / 'summary.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'episodes': len(rows), 'weights_unchanged': True,
        'inference_environment_queries': 0, 'inference_tripwires_enabled': True,
        'new_learned_modules': 0}) + '\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
