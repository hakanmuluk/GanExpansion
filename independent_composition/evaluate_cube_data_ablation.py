"""Matched broader-data validation using only original model forwards.

Every own-skill/control and predeclared composition configuration is retained.
Native execution is used for measurement, outside guarded policy inference.
No episode selection, geometry scoring or simulator call occurs in inference.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .unchanged_composition import CompositionConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ['root', 'source', 'out', 'stage-a']:
        p.add_argument('--' + flag, required=True)
    p.add_argument('--include-original', action='store_true')
    p.add_argument('--checkpoint-dir', help='Default: Stage A; use Stage B for its trained adapters')
    p.add_argument('--base-checkpoint', help='Default: original frozen base')
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    root, source, out, stage_a = [Path(v).resolve() for v in
        [args.root, args.source, args.out, args.stage_a]]
    assert not out.exists()
    out.mkdir(parents=True)
    (out / 'logs').mkdir()
    original_base = root / 'cube/artifacts/final/base_policy.pt'
    base = Path(args.base_checkpoint).resolve() if args.base_checkpoint else original_base
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else stage_a
    if args.include_original and base != original_base:
        raise ValueError('Original adapters require their original base; compare separate matched runs')
    env = {**os.environ, 'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONPATH': str(source) + ':' + str(source / 'cube')}
    families = {'diverse': {s: checkpoint_dir / ('train_' + s) / 'best.pt' for s in ['keepout', 'waypoint']}}
    if args.include_original:
        families['original'] = {s: root / f'cube_addon_artifacts/final/{s}_s42/adapter.pt'
                                for s in ['keepout', 'waypoint']}
    state = {'pid': os.getpid(), 'args': vars(args), 'scope': __doc__, 'jobs': []}

    def save():
        temp = out / 'status.tmp.json'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(out / 'status.json')

    def run_jobs(jobs, parallel):
        active = []
        pending = list(jobs)
        while pending or active:
            while pending and len(active) < parallel:
                row = pending.pop(0)
                log = (out / 'logs' / (row['name'] + '.log')).open('w')
                process = subprocess.Popen(row['command'], cwd=source, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                row.update(pid=process.pid, status='running', started_unix=time.time())
                active.append((process, log, row))
                save()
            for process, log, row in active[:]:
                code = process.poll()
                if code is None:
                    continue
                log.close()
                row.update(returncode=code, status='complete' if code == 0 else 'failed')
                active.remove((process, log, row))
                save()
            if active:
                time.sleep(2)
        failures = [r['name'] for r in jobs if r['status'] != 'complete']
        if failures:
            raise RuntimeError('Failed jobs: ' + ', '.join(failures))

    own_jobs = []
    for family, checkpoints in families.items():
        for skill, checkpoint in checkpoints.items():
            name = family + '_own_' + skill
            command = [sys.executable, '-m', 'cube_addons.evaluate', '--checkpoint', str(checkpoint),
                '--base', str(base), '--contexts', str(stage_a / 'data' / skill / 'val_contexts.npz'),
                '--skill', skill, '--out', str(out / name), '--count', '16', '--codes', '4',
                '--seed', '867', '--workers', '2']
            own_jobs.append({'name': name, 'command': command, 'summary': str(out / name / 'summary.json')})
    state['jobs'].extend(own_jobs)
    # Predeclared output-algebra controls and short/full action agreement.
    # Near-zero codes are included because the new training family assigns
    # them outward detours. Both old/new weights use exactly the same codes.
    configs = {
        'keepout': CompositionConfig(merge='keepout', k_code=.05, w_code=.05),
        'waypoint': CompositionConfig(merge='waypoint', k_code=.05, w_code=.05),
        'fixed_z_blend': CompositionConfig(merge='blend', axes='z', alpha=.5),
        'fixed_xyz_blend': CompositionConfig(merge='blend', axes='xyz', alpha=.5, k_code=.05, w_code=.05),
        'agreement_h2': CompositionConfig(merge='blend', axes='xyz', alpha=.5, optimize=True,
            agreement_horizon=2, waypoint_sign=0, trust=0., minimum_magnitude=.01, maximum_magnitude=2.5),
        'agreement_h16': CompositionConfig(merge='blend', axes='xyz', alpha=.5, optimize=True,
            agreement_horizon=16, waypoint_sign=0, trust=0., minimum_magnitude=.01, maximum_magnitude=2.5),
    }
    composition_jobs = []
    for name, cfg in configs.items():
        config_path = out / (name + '.json')
        config_path.write_text(json.dumps(asdict(cfg), indent=2) + '\n')
        for family, checkpoints in families.items():
            for group in ['supported_easy', 'supported_conflict']:
                job_name = family + '_' + name + '_' + group
                command = [sys.executable, '-m', 'independent_composition.evaluate_unchanged_composition',
                    '--env', 'cube', '--tasks', str(root / 'cube_composition_artifacts/tasks_v2' / ('val_' + group + '.npz')),
                    '--out', str(out / job_name), '--config', str(config_path), '--count', str(args.count),
                    '--execute', '8', '--keepout-checkpoint', str(checkpoints['keepout']),
                    '--waypoint-checkpoint', str(checkpoints['waypoint']), '--base-checkpoint', str(base)]
                composition_jobs.append({'name': job_name, 'command': command,
                                         'summary': str(out / job_name / 'summary.json')})
    state['jobs'].extend(composition_jobs)
    save()  # Save every configuration before any evaluation.
    run_jobs(own_jobs, args.workers)
    run_jobs(composition_jobs, args.workers)
    report = {'scope': 'Development validation only; no final-test or general composition claim.',
              'own_validation': {r['name']: json.loads(Path(r['summary']).read_text()) for r in own_jobs},
              'joint_validation': {r['name']: json.loads(Path(r['summary']).read_text()) for r in composition_jobs}}
    (out / 'aggregate.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'jobs': len(state['jobs']), 'complete': True}) + '\n')


if __name__ == '__main__':
    main()
