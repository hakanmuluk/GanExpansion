"""Fit retained-plus-coverage data with the verbatim original adapter trainer.

Defaults preserve the template settings. Explicit existing output-rank and
feedback-weight overrides are recorded. Neural classes, forward interfaces,
private Gaussian dimensions and loss definitions stay unchanged. Checkpoint
selection uses own-skill validation. No joint examples or outcomes train models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'source', 'out', 'data', 'template-stage']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--skill', choices=['keepout', 'waypoint'], required=True)
    p.add_argument('--raw', help='If given, wait for collection and pack fresh data first')
    p.add_argument('--raw-pid', type=int)
    p.add_argument('--val-seed', type=int, default=2201001)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--feedback', help='Existing original feedback-format own-skill data')
    p.add_argument('--feedback-pid', type=int)
    p.add_argument('--feedback-weight', type=float,
        help='Optional weight for the existing original feedback reconstruction term')
    p.add_argument('--output-rank', type=int,
        help='Optional existing output-modulation rank; original trainer supports zero-preserving expansion')
    p.add_argument('--complete-observation-replay', action='store_true',
        help='Refresh derived contact observations at native validation episode resets')
    p.add_argument('--lr', type=float)
    p.add_argument('--eval-every', type=int, default=1500)
    args = p.parse_args()
    root, source, out, data, template = [Path(v).resolve() for v in
        [args.root, args.source, args.out, args.data, args.template_stage]]
    assert not out.exists()
    out.mkdir(parents=True)
    (out / 'logs').mkdir()
    description = json.loads((template / 'status.json').read_text())
    command = next(row['command'][:] for row in description['stages']
                   if row['name'] == 'train_' + args.skill)
    base = Path(command[command.index('--base') + 1])
    initialize = template / ('train_' + args.skill) / 'best.pt'
    state = {'pid': os.getpid(), 'args': vars(args), 'scope': __doc__, 'stages': [],
        'base_sha256': sha(base), 'initial_adapter_sha256': sha(initialize),
        'original_trainer_sha256': sha(source / 'cube_addons/train.py')}
    env = {**os.environ, 'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONPATH': str(source) + ':' + str(source / 'cube')}

    def save():
        temporary = out / 'status.tmp.json'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(out / 'status.json')

    def run(name, cmd):
        row = {'name': name, 'command': cmd, 'status': 'running', 'started_unix': time.time()}
        with (out / 'logs' / (name + '.log')).open('w') as log:
            process = subprocess.Popen(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            state['stages'].append(row)
            save()
            code = process.wait()
        row.update(status='complete' if code == 0 else 'failed', returncode=code,
                   finished_unix=time.time())
        save()
        if code:
            raise RuntimeError('Failed: ' + name + '; inspect saved log')

    save()
    if args.feedback:
        feedback = Path(args.feedback).resolve()
        state['waiting_for_feedback'] = str(feedback)
        save()
        while not (feedback.parent / 'DONE.json').exists():
            if not args.feedback_pid:
                raise ValueError('Pending feedback requires its live collector PID')
            process_file = Path('/proc/' + str(args.feedback_pid) + '/cmdline')
            if not process_file.exists() or not process_file.read_bytes():
                raise RuntimeError('Feedback collector stopped before DONE; inspect log')
            time.sleep(10)
        with np.load(feedback) as values:
            count = len(values['states'])
            for key, shape in [('states', (count, 47)), ('contexts', (count, 20)),
                               ('actions', (count, 16, 5)), ('mask', (count, 16)),
                               ('base_z', (count, 2)), ('new_z', (count, 1))]:
                assert values[key].shape == shape and np.isfinite(values[key]).all(), key
        state.pop('waiting_for_feedback')
        state['feedback'] = {'path': str(feedback), 'chunks': count,
                             'sha256': sha(feedback), 'existing_loss': 'Original feedback reconstruction'}
        save()
    if args.raw:
        assert args.raw_pid and not data.exists()
        raw = Path(args.raw).resolve()
        state['waiting_for_raw'] = str(raw)
        save()
        while True:
            config = json.loads((raw / 'config.json').read_text()) if (raw / 'config.json').exists() else {}
            if (raw / 'DONE.json').exists() and 'coverage_source_sha256' in config:
                break
            process_file = Path('/proc/' + str(args.raw_pid) + '/cmdline')
            if not process_file.exists() or not process_file.read_bytes():
                raise RuntimeError('Collection stopped before final provenance; inspect its log')
            time.sleep(10)
        state.pop('waiting_for_raw')
        run('pack_retained_plus_coverage', [sys.executable, '-m', 'independent_composition.pack_cube_coverage',
            '--previous', str(root / 'independent_composition_artifacts/cube_diverse_v34_stageA/data'),
            '--raw', str(raw), '--out', str(data), '--skill', args.skill, '--val-seed', str(args.val_seed)])
    assert (data / 'DONE.json').exists()
    from cube_addons.data import EpisodeDataset
    from cube_addons.evaluate import load_adapter
    torch.set_num_threads(1)
    torch.manual_seed(119)
    ds = EpisodeDataset(data / args.skill / 'train.npz', 'cpu', alignment=8)
    b = ds.sample(4, 2)
    model, encoder, meta = load_adapter(initialize, base, 'cpu')
    model.requires_grad_(False)
    bz = b['base_z'][:, None].expand(-1, 2, -1)
    z = b['new_z'][:, None].expand(-1, 2, -1)
    with torch.no_grad():
        pred = model(b['states'], bz, z, b['contexts'])
        assert pred.shape == (4, 2, 16, 5) and torch.isfinite(pred).all()
        assert torch.equal(model(b['states'], bz, z, b['contexts'], gate=0), model.base(b['states'], bz))
        mu, lv = encoder(b['episode_states'], b['episode_actions'], b['episode_contexts'], b['base_z'], b['lengths'])
        assert torch.isfinite(mu).all() and torch.isfinite(lv).all()
    state['data_forward_check'] = {'training_episodes': ds.size, 'original_chunk_shape': [16, 5],
        'disabled_adapter_equals_base': True, 'encoder_finite': True,
        'train_sha256': sha(data / args.skill / 'train.npz')}
    save()
    del ds, b, model, encoder, meta
    evaluation_module = ('independent_composition.evaluate_cube_single_replay'
                         if args.complete_observation_replay else 'cube_addons.evaluate')
    run('initial_adapter_control', [sys.executable, '-m', evaluation_module,
        '--checkpoint', str(initialize), '--base', str(base), '--contexts',
        str(data / args.skill / 'val_contexts.npz'), '--skill', args.skill, '--out',
        str(out / 'initial_adapter_control'), '--count', '32', '--codes', '4',
        '--seed', '867', '--workers', '3'])
    updates = {'--run': str(out / ('train_' + args.skill)), '--data': str(data),
               '--initialize': str(initialize), '--steps': str(args.steps),
               '--eval-contexts': '32', '--eval-workers': '3', '--eval-every': str(args.eval_every)}
    if args.lr is not None:
        updates['--lr'] = str(args.lr)
    for flag, value in updates.items():
        command[command.index(flag) + 1] = value
    if args.feedback:
        if '--feedback' in command:
            command[command.index('--feedback') + 1] = str(feedback)
        else:
            command.extend(['--feedback', str(feedback)])
    if args.feedback_weight is not None:
        assert args.feedback and args.feedback_weight > 0
        if '--feedback-weight' in command:
            command[command.index('--feedback-weight') + 1] = str(args.feedback_weight)
        else:
            command.extend(['--feedback-weight', str(args.feedback_weight)])
    if args.output_rank is not None:
        assert args.output_rank > 0
        command[command.index('--output-rank') + 1] = str(args.output_rank)
    if args.complete_observation_replay:
        assert command[1] == '-m'
        command[2] = 'independent_composition.train_cube_original_replay'
    run('train_' + args.skill, command)
    assert sha(base) == state['base_sha256'] and sha(initialize) == state['initial_adapter_sha256']
    trained = out / ('train_' + args.skill) / 'best.pt'
    ck = torch.load(trained, map_location='cpu', weights_only=False)
    assert ck['frozen_digest_before'] == ck['frozen_digest_after']
    (out / 'DONE.json').write_text(json.dumps({'skill': args.skill, 'new_adapter': str(trained),
        'sha256': sha(trained), 'selected_step': ck['step'], 'own_success': ck['validation']['success'],
        'base_unchanged': True,
        'architecture_and_loss_settings_unchanged': args.feedback_weight is None and args.output_rank is None,
        'architecture_unchanged': args.output_rank is None,
        'original_neural_classes_and_input_dimensions_unchanged': True,
        'loss_definitions_unchanged': True,
        'feedback_weight_override': args.feedback_weight,
        'output_modulation_rank_override': args.output_rank,
        'complete_observation_replay_in_validation': args.complete_observation_replay,
        'composition_not_yet_verified': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()
