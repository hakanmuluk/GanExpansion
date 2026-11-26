"""Apply the unchanged Cube latent-agreement optimizer to original TwoRoom actors.

Only the original private scalars and base first-coordinate magnitudes are
optimized. Simulation/constraint predicates are evaluation-only. This driver
does not add a neural module, task score, encoder, or phase-dependent handoff.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import torch
from .latent_action_composition import AmplitudeConfig, AmplitudeComposer
from .evaluate_diffusion import evaluate_tworoom, summarize
from .inference_guard import GuardedPolicy
from .legacy import digest_file
from .unchanged_composition import CompositionConfig, OriginalComposer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['tasks', 'config', 'out']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--method', choices=['amplitude', 'fixed'], default='amplitude')
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--execute', type=int, default=8)
    p.add_argument('--seed', type=int, default=241)
    p.add_argument('--device', default='cuda')
    p.add_argument('--base', default='artifacts/final/base_policy.pt')
    p.add_argument('--keepout', default='addon_artifacts/final/keepout.pt')
    p.add_argument('--waypoint', default='addon_artifacts/final/waypoint.pt')
    p.add_argument('--frozen-protocol')
    p.add_argument('--bank-manifest')
    args = p.parse_args(); torch.set_num_threads(2)
    assert 1 <= args.execute <= 16 and args.count > 0
    if args.frozen_protocol:
        frozen = json.loads(Path(args.frozen_protocol).read_text())
        assert frozen['configuration_sha256'][args.method] == digest_file(args.config)
        assert frozen['execute'] == args.execute and frozen['evaluation_seed'] == args.seed
        assert frozen['scene_count'] == args.count
        for name, checksum in frozen['source_sha256'].items():
            assert digest_file(Path(__file__).parent / name) == checksum, name
        for path, checksum in frozen['models'].items():
            assert digest_file(path) == checksum, path
        assert args.bank_manifest
        manifest = json.loads(Path(args.bank_manifest).read_text())
        assert manifest['protocol_sha256'] == digest_file(args.frozen_protocol)
        assert any(row['path'] == args.tasks and row['sha256'] == digest_file(args.tasks)
                   for row in manifest['strata'].values())
    else:
        assert not Path(args.tasks).name.startswith(('test', 'final')), 'Development only without a freeze'
    from tworoom_composition.models import FrozenBank
    bank = FrozenBank(keepout=args.keepout, waypoint=args.waypoint, base=args.base, device=args.device)
    if args.method == 'amplitude':
        config = AmplitudeConfig(**json.loads(Path(args.config).read_text()))
        cls = AmplitudeComposer
        assert config.axes == 'all' and config.agreement_horizon == args.execute
    else:
        config = CompositionConfig(**json.loads(Path(args.config).read_text()))
        cls = OriginalComposer
        assert not config.optimize
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    sources = ['evaluate_tworoom_latent.py', 'amplitude_composition.py', 'latent_action_composition.py',
               'unchanged_composition.py', 'inference_guard.py', 'evaluate_diffusion.py']
    source_sha = {name: digest_file(Path(__file__).parent / name) for name in sources}
    (out / 'source').mkdir()
    for name in sources:
        shutil.copy2(Path(__file__).parent / name, out / 'source' / name)
    (out / 'config.json').write_text(json.dumps({
        'args': vars(args), 'composition': asdict(config), 'models': bank.files,
        'source_sha256': source_sha, 'task_sha256': digest_file(args.tasks),
        'scope': __doc__, 'optimizer_algorithm_identical_to_cube': cls is AmplitudeComposer,
        'same_original_input_dimensions': {'state': 10, 'own_context': 12, 'base_z': 2, 'private_z_each': 1},
        'prediction_horizon': 16, 'execute_horizon': args.execute,
    }, indent=2) + '\n')
    policy = GuardedPolicy(cls(bank, config, args.seed))
    def factory(*unused):
        return policy
    started = time.time()
    rows = evaluate_tworoom(args, [], bank.k.base, out, factory)
    summary = summarize(rows); summary['total_seconds'] = time.time() - started
    bank.assert_frozen()
    assert all(digest_file(path) == checksum for path, checksum in bank.files.items())
    (out / 'records.json').write_text(json.dumps(rows, indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({
        'episodes': len(rows), 'weights_frozen': True, 'inference_environment_queries': 0,
        'inference_tripwires_enabled': True, 'encoders_used_for_planning': False,
        'new_learned_modules': 0, 'optimizer_algorithm_identical_to_cube': cls is AmplitudeComposer,
        'model_files_sha256': bank.files,
    }, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
