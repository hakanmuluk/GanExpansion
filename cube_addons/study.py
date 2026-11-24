"""Recorded, restartable final evaluation after validation-only model selection.

This runner never trains a generator or selects using test outcomes. Encoder
calibration uses training trajectories and validation codes only.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path('cube_addon_artifacts')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def step(module, arguments, marker, name):
    marker = Path(marker)
    if marker.is_file():
        print('CACHED', name, str(marker), flush=True)
        return
    command = [sys.executable, '-m', 'cube_addons.' + module, *map(str, arguments)]
    logs = ROOT / 'study_logs'; logs.mkdir(exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('START', name, command, flush=True)
    with (logs / (name + '.log')).open('a') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    receipt = {'name': name, 'argv': command, 'started_utc': started,
               'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'returncode': result.returncode, 'marker': str(marker),
               'marker_sha256': digest(marker) if marker.is_file() else None}
    with (logs / 'steps.jsonl').open('a') as log:
        log.write(json.dumps(receipt) + '\n')
    if result.returncode or not marker.is_file():
        raise RuntimeError(f'{name} failed; see {logs / (name + ".log")}')
    print('DONE', name, flush=True)


def evaluate(skill, name, checkpoint=None, count=128, codes=8, resample=False):
    out = ROOT / ('temporal' if resample else 'test') / name
    arguments = ['--skill', skill, '--contexts', ROOT / 'data_force' / skill / 'test_contexts.npz',
                 '--count', count, '--codes', codes, '--seed', 20261122,
                 '--workers', 6, '--force-contacts', '--out', out]
    if checkpoint: arguments += ['--checkpoint', checkpoint]
    if resample: arguments += ['--resample']
    step('evaluate', arguments, out / 'summary.json', ('resample_' if resample else 'test_') + name)


def finalize(skill, seed, source, physical_count=64, export_name=None):
    import torch
    name = export_name or f'{skill}_s{seed}'
    physical = ROOT / 'physical' / name
    step('feedback', ['--skill', skill, '--checkpoint', source, '--contexts', ROOT / 'data_force' / skill / 'train_contexts.npz',
                     '--count', physical_count, '--codes', 8, '--seed', 1731, '--workers', 6,
                     '--out', physical, '--rollouts-only'], physical / 'summary.json', 'physical_' + name)
    cal = ROOT / 'calibration' / name
    step('calibrate', ['--skill', skill, '--checkpoint', source, '--physical-train', physical,
                      '--val-contexts', ROOT / 'data_force' / skill / 'val_contexts.npz',
                      '--val-count', 64, '--val-codes', 4, '--workers', 6, '--steps', 1500,
                      '--seed', seed, '--out', cal], cal / 'DONE.json', 'calibrate_' + name)
    selected = cal / 'best.pt'; final = ROOT / 'final' / name; final.mkdir(parents=True, exist_ok=True)
    original = torch.load(source, map_location='cpu', weights_only=False)
    calibrated = torch.load(selected, map_location='cpu', weights_only=False)
    assert set(original['addon']) == set(calibrated['addon'])
    assert all(torch.equal(v, calibrated['addon'][k]) for k, v in original['addon'].items())
    assert original['base_sha256'] == calibrated['base_sha256']
    target = final / 'adapter.pt'
    if target.is_file():
        assert digest(target) == digest(selected), 'Final selection changed after export'
    else:
        shutil.copy2(selected, target)
    (final / 'selection.json').write_text(json.dumps({
        'skill': skill, 'seed': seed, 'generator_source': str(source), 'generator_source_sha256': digest(source),
        'generator_training_step': original['step'], 'generator_validation': original['validation'],
        'encoder_source': str(selected), 'adapter_sha256': digest(target),
        'encoder_validation': calibrated.get('calibration', {}).get('validation'),
        'selection': 'Generator selected before test using validation success and diversity; encoder selected using validation physical-code RMSE.',
        'generator_identical_before_after_calibration': True,
        'base_sha256': original['base_sha256']}, indent=2) + '\n')
    evaluate(skill, name, target)
    return target


def main():
    p = argparse.ArgumentParser(); p.add_argument('--skill', choices=['keepout', 'waypoint'], required=True)
    p.add_argument('--runs', nargs=3); p.add_argument('--baselines', action='store_true')
    p.add_argument('--baseline-runs',nargs=2,help='No-code and full-fine-tuning run names, in that order')
    args = p.parse_args()
    names = args.runs or (['keepout_v2_s42', 'keepout_s43', 'keepout_s44'] if args.skill == 'keepout' else [])
    if len(names) != 3: raise ValueError('Supply the three independently trained run names')
    for seed, run in zip([42, 43, 44], names):
        finalize(args.skill, seed, ROOT / 'runs' / run / 'best.pt')
    selected = ROOT / 'final' / f'{args.skill}_s42' / 'adapter.pt'
    evaluate(args.skill, f'{args.skill}_s42', selected, count=64, resample=True)
    style = ROOT / 'style' / f'{args.skill}_s42'
    step('audit_style', ['--checkpoint', selected, '--contexts', ROOT / 'data_force' / args.skill / 'test_contexts.npz',
                         '--skill', args.skill, '--count', 16, '--workers', 6, '--out', style], style / 'summary.json', 'style_' + args.skill)
    if args.baselines:
        evaluate(args.skill, args.skill + '_original')
        baseline_names=args.baseline_runs or [args.skill+'_noz_single',args.skill+'_fullft']
        for suffix,run_name in zip(['noz_single', 'fullft'],baseline_names):
            run = ROOT / 'runs' / run_name
            if suffix=='fullft':
                finalize(args.skill,42,run/'best.pt',export_name=args.skill+'_fullft')
            else:
                evaluate(args.skill, args.skill + '_' + suffix, run / 'best.pt')


if __name__ == '__main__': main()
