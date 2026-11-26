"""Native evaluation outside guarded direction-agreement code inference."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import torch
from .cube_checkpoint_bank import CheckpointBank
from .direction_composition import DirectionConfig, DirectionComposer
from .evaluate_diffusion import evaluate_cube, summarize
from .inference_guard import GuardedPolicy
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['tasks', 'config', 'out', 'base', 'keepout', 'waypoint']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--execute', type=int, default=8)
    p.add_argument('--seed', type=int, default=241)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    assert not Path(args.tasks).name.startswith(('test', 'final'))
    assert 1 <= args.execute <= 16
    torch.set_num_threads(2)
    out = Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    cfg = DirectionConfig(**json.loads(Path(args.config).read_text()))
    bank = CheckpointBank(args.keepout, args.waypoint, args.base, args.device)
    names = ['direction_composition.py', 'evaluate_directional_composition.py', 'unchanged_composition.py',
             'cube_checkpoint_bank.py', 'evaluate_diffusion.py', 'inference_guard.py']
    source = {name: digest_file(Path(__file__).parent / name) for name in names}
    (out / 'source').mkdir()
    for name in names:
        shutil.copy2(Path(__file__).parent / name, out / 'source' / name)
    (out / 'config.json').write_text(json.dumps({'args': vars(args), 'composition': asdict(cfg),
        'models': bank.files, 'source_sha256': source, 'task_sha256': digest_file(args.tasks),
        'scope': 'Existing original network architectures/forwards and bounded Z search. Only agreement metric differs. No new learned module, simulator, geometry score or inverse encoder during inference.'}, indent=2) + '\n')

    def factory(args, unused, base, seed):
        return GuardedPolicy(DirectionComposer(bank, cfg, seed))

    started = time.time()
    rows = evaluate_cube(args, [], bank.k.base, out, factory)
    metrics = summarize(rows)
    metrics['total_seconds'] = time.time() - started
    bank.assert_frozen()
    assert all(digest_file(path) == checksum for path, checksum in bank.files.items())
    (out / 'summary.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'episodes': len(rows), 'weights_unchanged': True,
        'inference_environment_queries': 0, 'inference_tripwires_enabled': True,
        'new_learned_modules': 0}) + '\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
