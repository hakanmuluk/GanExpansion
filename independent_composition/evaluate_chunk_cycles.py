"""Evaluate original-encoder cycles with frozen original Cube generators."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .chunk_cycle_composition import ChunkCycleConfig, ChunkCycleBank, ChunkCycleComposer
from .chunk_cycle_modes import ChunkCycleModeConfig, ChunkCycleModeComposer
from .evaluate_diffusion import evaluate_cube, summarize
from .inference_guard import GuardedPolicy
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['tasks', 'config', 'out', 'base', 'keepout', 'waypoint',
                'keepout-encoder', 'waypoint-encoder']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--execute', type=int, default=8)
    p.add_argument('--seed', type=int, default=241)
    p.add_argument('--device', default='cuda')
    p.add_argument('--backend', choices=['private', 'mode'], default='private')
    p.add_argument('--paired', action='store_true')
    p.add_argument('--paired-variable', choices=['keepout', 'waypoint'], default='keepout')
    p.add_argument('--keepout-gate', type=int, choices=[0, 1], default=1)
    p.add_argument('--waypoint-gate', type=int, choices=[0, 1], default=1)
    args = p.parse_args()
    assert not Path(args.tasks).name.startswith(('test', 'final'))
    assert 1 <= args.execute <= 16
    torch.set_num_threads(2)
    with np.load(args.tasks) as f:
        n = min(args.count, len(f['seed']))
        pairs = f['paired_id'][:n].astype(int) if args.paired else np.arange(n)
        if args.paired:
            assert n % 2 == 0 and np.array_equal(pairs[::2], pairs[1::2])
            fixed = 'waypoint' if args.paired_variable == 'keepout' else 'keepout'
            for key in ['initial_state', 'initial_cube', 'goal', 'base_z', fixed]:
                assert np.array_equal(f[key][:n:2], f[key][1:n:2]), key
    out = Path(args.out)
    assert n > 0 and not out.exists()
    out.mkdir(parents=True)
    config_type, policy_type = ((ChunkCycleConfig, ChunkCycleComposer) if args.backend == 'private'
        else (ChunkCycleModeConfig, ChunkCycleModeComposer))
    cfg = config_type(**json.loads(Path(args.config).read_text()))
    bank = ChunkCycleBank(args.keepout, args.waypoint, args.base,
        args.keepout_encoder, args.waypoint_encoder, args.device,
        keepout_gate=args.keepout_gate, waypoint_gate=args.waypoint_gate)
    sources = ['evaluate_chunk_cycles.py', 'chunk_cycle_composition.py', 'chunk_cycle_modes.py',
        'calibrate_original_chunk_encoder.py', 'quantile_composition.py',
        'unchanged_composition.py', 'gate_ablation_bank.py', 'cube_checkpoint_bank.py',
        'evaluate_diffusion.py', 'inference_guard.py', 'speed_composition.py', 'mode_composition.py', 'observation_replay.py']
    (out/'source').mkdir()
    for name in sources:
        shutil.copy2(Path(__file__).parent/name, out/'source'/name)
    (out/'config.json').write_text(json.dumps({'args': vars(args), 'composition': asdict(cfg),
        'models': bank.files, 'task_sha256': digest_file(args.tasks),
        'source_sha256': {name: digest_file(Path(__file__).parent/name) for name in sources},
        'scope': __doc__, 'generator_weights_frozen': True,
        'encoders': 'Existing two private SkillEncoder instances, recalibrated independently; no extra learned network',
        'state_interface': 'Identical CURRENT observation repeated for each action token, explicitly trained this way',
        'original_base_inputs_fixed': args.backend == 'private',
        'existing_base_speed_inputs_optimized': getattr(cfg, 'optimize_speed_inputs', False),
        'reset_observation_protocol': 'Native derived contact force refreshed; paired initial observations asserted equal',
        'original_binary_gates': [args.keepout_gate, args.waypoint_gate]}, indent=2)+'\n')
    cursor = iter(pairs)

    def factory(a, unused, base, ignored_seed):
        return GuardedPolicy(policy_type(bank, cfg, a.seed+9973*int(next(cursor))))

    started = time.time()
    rows = evaluate_cube(args, [], bank.k.base, out, factory)
    metrics = summarize(rows)
    metrics['total_seconds'] = time.time()-started
    if args.paired:
        metrics.update(paired_worlds=n//2,
            both_variants_successful_worlds=sum(rows[i]['success'] and rows[i+1]['success'] for i in range(0,n,2)),
            inner_successes=sum(row['success'] for row in rows[::2]),
            outer_successes=sum(row['success'] for row in rows[1::2]))
    bank.assert_frozen()
    assert all(digest_file(path)==sha for path,sha in bank.files.items())
    (out/'summary.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes': len(rows),
        'weights_unchanged': True, 'inference_environment_queries': 0,
        'inference_tripwires_enabled': True, 'new_learned_modules': 0,
        'retrained_existing_private_encoders_used': True})+'\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
