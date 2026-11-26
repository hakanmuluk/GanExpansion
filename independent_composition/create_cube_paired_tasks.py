"""Counterfactual Cube evaluation cases; no training data or policy is produced.

Keep physics, goal, base code and waypoint fixed while moving keepout to
opposite sides of the waypoint. Their feasible waypoint contact caps are
disjoint. Fixed own-keepout experts provide OFFLINE feasibility witnesses only;
their paths/codes are never supplied to a policy. Every constructed case is
retained, regardless of witness outcome or any learned policy's performance.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from cube_addons.geometry import axis_frame, CUBE_BOUND_RADIUS
from .audit_cube_data_support import initialize, execute
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--offset', type=int, default=40)
    p.add_argument('--count', type=int, default=8, help='Worlds; two counterfactual cases each')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--vary', choices=['keepout', 'waypoint'], default='keepout')
    args = p.parse_args()
    out = Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    with np.load(args.bank) as data:
        assert args.offset + args.count <= len(data['seed'])
        originals = [{k: data[k][i] for k in data.files}
                     for i in range(args.offset, args.offset + args.count)]
    cases, jobs, certificates = [], [], []
    for pair_id, original in enumerate(originals):
        _, _, _, perp, _ = axis_frame(original['initial_cube'], original['goal'], original['initial_cube'])
        side = np.r_[perp, 0.] * (1. if original['base_z'][0] >= 0 else -1.)
        radius, distance = .026, .053
        inflated = radius + CUBE_BOUND_RADIUS
        wp_radius = float(original['waypoint'][3])
        gap = inflated ** 2 - distance ** 2 - wp_radius ** 2
        assert gap > 0 and distance + wp_radius > inflated
        certificates.append({'pair_id': pair_id, 'inflated_keepout_radius_m': inflated,
            'center_distance_m': distance, 'waypoint_radius_m': wp_radius,
            'disjoint_caps_squared_gap_m2': gap})
        for name, sign, code in [('inner', -1., 0.), ('outer', 1., 3.)]:
            case = {k: np.asarray(v).copy() for k, v in original.items()}
            if args.vary == 'keepout':
                case['keepout'] = np.r_[case['waypoint'][:3] + sign * distance * side, radius]
            else:
                case['keepout'] = np.r_[original['keepout'][:3], radius]
                case['waypoint'] = np.r_[case['keepout'][:3] + sign * distance * side, wp_radius]
                code = 3. if sign < 0 else 0.
            case['paired_id'] = np.int64(pair_id)
            case['paired_variant'] = np.array(name)
            case['paired_variable'] = np.array(args.vary)
            cases.append(case)
            jobs.append((case, 'keepout', 'diverse', code, len(cases) - 1))
    path = out / 'val_paired_keepout.npz'
    np.savez_compressed(path, **{k: np.stack([c[k] for c in cases]) for k in cases[0]})
    frozen = {'scope': __doc__, 'args': vars(args), 'cases': len(cases),
        'bank_sha256': digest_file(args.bank), 'cases_sha256': digest_file(path),
        'source_sha256': digest_file(__file__), 'geometric_certificates': certificates,
        'paired_variable': args.vary,
        'learned_policy_outcomes_used_for_selection': False,
        'witnesses_used_for_training_or_inference': False}
    (out / 'manifest.json').write_text(json.dumps(frozen, indent=2) + '\n')
    records = []
    with mp.get_context('spawn').Pool(args.workers, initializer=initialize) as pool:
        for row in pool.imap_unordered(execute, jobs, chunksize=1):
            records.append(row)
            (out / 'witness_progress.json').write_text(json.dumps({'finished': len(records), 'total': len(jobs)}) + '\n')
    records.sort(key=lambda r: r['context'])
    report = {'scope': 'OFFLINE feasibility diagnostic only; all cases retained.',
        'episodes': len(records), 'joint_witness_successes': sum(r['joint_success'] for r in records),
        'own_witness_successes': sum(r['own_success'] for r in records), 'records': records}
    (out / 'witness_aggregate.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'paired_cases': len(cases), 'all_cases_retained': True}) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))


if __name__ == '__main__':
    main()
