"""Offline, auditable tables from completed physical evaluations; no model queries."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze import bounded_mean_interval, auc


def collect(roots):
    groups = defaultdict(list)
    for root in roots:
        for done in sorted(Path(root).rglob('DONE.json')):
            folder = done.parent
            if not (folder / 'records.json').exists() or not (folder / 'config.json').exists():
                continue
            config = json.loads((folder / 'config.json').read_text())
            rows = json.loads((folder / 'records.json').read_text())
            plans = json.loads((folder / 'planning.json').read_text())
            assert len(rows) == len(plans)
            for row, episode in zip(rows, plans):
                groups[(row['kind'], folder.name)].append((row, episode, config, folder))
    return groups


def summarize(groups):
    result = []
    metrics = ['success', 'released_success', 'keepout_satisfied', 'waypoint_satisfied',
               'grasped', 'dropped', 'grasp_losses', 'robot_table_contact_in_transport',
               'ee_region_violation_diagnostic', 'supported_transport_path_fraction',
               'stationary', 'latent_total_variation', 'action_disagreement_initial',
               'action_disagreement_mean', 'steps']
    for (kind, method), items in sorted(groups.items()):
        by_context = defaultdict(list)
        by_run = defaultdict(list)
        for row, _, config, folder in items:
            by_context[row['seed']].append(row['success'])
            by_run[(config['args']['model_seed'], config['args']['seed'], str(folder))].append(row['success'])
        context_rates = [np.mean(x) for x in by_context.values()]
        a = {'kind': kind, 'method': method, 'episodes': len(items), 'unique_contexts': len(by_context),
             'model_seeds': sorted({c['args']['model_seed'] for _, _, c, _ in items}),
             'search_seeds': sorted({c['args']['seed'] for _, _, c, _ in items}),
             'success_count': int(sum(r['success'] for r, _, _, _ in items)),
             'success_context_bounded_95ci': bounded_mean_interval(float(np.mean(context_rates)), len(context_rates)),
             'per_run_success': [{'model_seed': k[0], 'search_seed': k[1], 'episodes': len(v), 'success': float(np.mean(v))}
                                 for k, v in by_run.items()]}
        a['success_run_sd'] = float(np.std([x['success'] for x in a['per_run_success']], ddof=1)) if len(by_run) > 1 else None
        for key in metrics:
            values = [r[key] for r, _, _, _ in items if r.get(key) is not None]
            a[key] = float(np.mean(values)) if values else None
        for skill in ['keepout', 'waypoint']:
            a['own_' + skill + '_released'] = float(np.mean([r['released_success'] and r[skill + '_satisfied'] for r, _, _, _ in items]))
        plans = [p for _, episode, _, _ in items for p in episode]
        active = [p for p in plans if p.get('iterations', 0) > 0]
        a['planning_calls'] = len(plans)
        a['active_optimization_calls'] = len(active)
        a['closed_loop_plan_seconds_p50'] = float(np.median([p['seconds'] for p in plans]))
        a['closed_loop_plan_seconds_p95'] = float(np.quantile([p['seconds'] for p in plans], .95))
        a['closed_loop_plan_exceeds_400ms'] = float(np.mean([p['seconds'] > .4 for p in plans]))
        a['mean_total_planning_seconds_per_episode'] = float(sum(p['seconds'] for p in plans) / len(items))
        for key in ['native_controls', 'native_forks', 'backward_steps', 'encoder_calls', 'encoder_seconds', 'iterations']:
            a[key + '_per_plan'] = float(np.mean([p.get(key, 0) for p in plans]))
            a[key + '_per_active_plan'] = float(np.mean([p.get(key, 0) for p in active])) if active else None
        a['family_call_counters_per_plan'] = {}
        for family in sorted({k for p in plans for k in p.get('calls', {})}):
            a['family_call_counters_per_plan'][family] = {
                key: float(np.mean([p.get('calls', {}).get(family, {}).get(key, 0) for p in plans]))
                for key in ['batched_calls', 'candidate_equivalents']}
        prefixes = [p['actual_prefix_max_error_m'] for p in plans if p.get('actual_prefix_max_error_m') is not None]
        a['max_native_forecast_prefix_error_m'] = max(prefixes, default=None)
        x = [(r['action_disagreement_initial'], r['success']) for r, _, _, _ in items if r.get('action_disagreement_initial') is not None]
        a['low_initial_disagreement_success_auc'] = auc(-np.array([v[0] for v in x]), [v[1] for v in x]) if x else None
        result.append(a)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = summarize(collect(args.root))
    (out / 'metrics.json').write_text(json.dumps({'roots': args.root, 'groups': records,
        'scope': 'Completed native evaluations only. Context intervals condition on tested model/search seeds. '
                 'closed_loop latency includes the actual experiment load. common_projection counters are non-neural operations. '
                 'Repeated executions of the same policy/context are not additional independent contexts.'}, indent=2) + '\n')
    scalar_keys = [k for k, v in records[0].items() if not isinstance(v, (list, dict))]
    with (out / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({'groups': len(records), 'physical_execution_records': sum(x['episodes'] for x in records)}))


if __name__ == '__main__':
    main()
