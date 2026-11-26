"""Delivery gate: verify the finite research protocol and all local raw evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='cube_composition_artifacts/completion_audit.json')
    args = parser.parse_args()
    root = Path('cube_composition_artifacts')
    expected = {'main_core': (93, 2976), 'main_extensions': (126, 3456),
                'prior_initialization_test': (36, 576), 'robustness_test': (24, 192),
                'infeasible_controls': (6, 96), 'combined_agreement_validation': (36, 288)}
    checks = []
    digest_cache = {}

    def sha(path):
        if str(path).startswith('/content/cs585_cube_composition/'):
            path = Path(str(path).removeprefix('/content/cs585_cube_composition/'))
        key = str(path)
        if key not in digest_cache:
            digest_cache[key] = digest(path)
        return digest_cache[key]

    for tree, (runs, episodes) in expected.items():
        folders = sorted(p.parent for p in (root / tree).rglob('records.json')
                         if (p.parent / 'config.json').exists())
        count = reused = controls = 0
        errors = []
        for folder in folders:
            if not (folder / 'DONE.json').exists():
                errors.append(str(folder) + ': not DONE')
                continue
            config = json.loads((folder / 'config.json').read_text())
            rows = json.loads((folder / 'records.json').read_text())
            plans = json.loads((folder / 'planning.json').read_text())
            if len(rows) != len(plans):
                errors.append(str(folder) + ': record/plan mismatch')
            count += len(rows)
            if config.get('reused_existing_episodes'):
                reused += len(rows)
            for name, expected_sha in config['frozen_files'].items():
                if sha(Path(name)) != expected_sha:
                    errors.append(name + ': frozen file mismatch')
            if sha(Path(config['args']['tasks'])) != config['task_sha256']:
                errors.append(str(folder) + ': task hash mismatch')
            for row in rows:
                source = folder / f"rollout_{row['context']:04d}.npz"
                if not source.exists():
                    errors.append(str(source) + ': missing physical trace')
                    continue
                with np.load(source) as data:
                    n = len(data['actions'])
                    controls += n
                    if len(data['poses']) != n + 1 or len(data['codes']) != n:
                        errors.append(str(source) + ': inconsistent trace lengths')
                    for field in ['actions', 'poses', 'codes', 'forces']:
                        if not np.isfinite(data[field]).all():
                            errors.append(str(source) + ': nonfinite ' + field)
        checks.append({'item': tree, 'expected_runs': runs, 'found_runs': len(folders),
                       'expected_records': episodes, 'found_records': count,
                       'reused_records': reused, 'new_native_executions': count - reused,
                       'recorded_control_steps': controls, 'errors': errors,
                       'pass': len(folders) == runs and count == episodes and not errors})
    geometry_path = root / 'geometry_audit_full/summary.json'
    geometry = json.loads(geometry_path.read_text()) if geometry_path.exists() else {}
    geometry_records_path = root / 'geometry_audit_full/records.json'
    geometry_records = json.loads(geometry_records_path.read_text()) if geometry_records_path.exists() else []
    geometry_errors = []
    for row in geometry_records:
        if sha(Path(row['source'])) != row['source_sha256']:
            geometry_errors.append(row['source'])
    checks.append({'item': 'full_native_geometry_replay', 'episodes': geometry.get('episodes'),
                   'local_raw_records': len(geometry_records), 'source_hash_errors': geometry_errors,
                   'pass': (root / 'geometry_audit_full/DONE.json').exists()
                           and geometry.get('episodes') == 7392
                           and len(geometry_records) == 7392 and not geometry_errors
                           and geometry.get('max_replay_pose_error_m', 1) < 2e-6})
    bench_path = root / 'benchmarks/DONE.json'
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    benchmark_checks = []
    for kind in ['supported_easy', 'supported_conflict', 'coupled_ood']:
        path = root / 'benchmarks' / (kind + '.json')
        record = json.loads(path.read_text()) if path.exists() else {}
        entries = record.get('records', [])
        benchmark_checks.append(bool(record) and len(entries) == bench.get('unique_methods', 0) * 3
                                and all(len(v['calls']) == 11 and all(np.isfinite(c['seconds']) and c['seconds'] > 0
                                                                    for c in v['calls']) for v in entries)
                                and record['source_replay_max_error_m'] < 2e-6
                                and sha(Path(record['args']['trajectory'])) == record['source_trajectory_sha256'])
    checks.append({'item': 'dedicated_timing', 'all_local_phase_measurements': benchmark_checks,
                   'pass': bench.get('complete', False) and bench.get('examples') == 3 and all(benchmark_checks)})
    video_checks = []
    for name in ['heldout_improvement', 'heldout_composition_failure', 'heldout_ood_failure']:
        folder = root / 'visuals' / name
        audit_path = folder / 'native_robot.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        passed = (audit.get('mp4') is True and (folder / 'native_robot.mp4').exists()
                  and (folder / 'paths.png').exists() and (folder / 'selection.json').exists())
        for source in audit.get('sources', []):
            passed &= source['max_replay_pose_error_m'] < 2e-6 and source['render_integration_state_change'] == 0
            passed &= sha(Path(source['source'])) == source['source_sha256']
        video_checks.append({'video': name, 'pass': bool(passed)})
    checks.append({'item': 'heldout_native_videos', 'videos': video_checks,
                   'pass': all(x['pass'] for x in video_checks)})
    visual_qa_path = root / 'visuals/visual_qa.json'
    visual_qa = json.loads(visual_qa_path.read_text()) if visual_qa_path.exists() else {}
    checks.append({'item': 'native_video_and_scientific_figure_visual_QA', 'pass': visual_qa.get('verified', False)})
    tests = ET.parse(root / 'tests_v6.xml').getroot().find('testsuite')
    checks.append({'item': 'native_implementation_tests', 'pass': tests.attrib['tests'] == '10'
                   and tests.attrib['failures'] == '0' and tests.attrib['errors'] == '0'})
    preserved = json.loads((root / 'preservation_final.json').read_text())
    checks.append({'item': 'five_previous_deliveries_and_frozen_dependencies', 'pass': preserved['all_match']})
    for path in ['tasks_v2/audit.json', 'executor_parity.json', 'analyses/original_base_disjointness.json',
                 'analyses/source_freeze_audit.json', 'analyses/behavior_full/episodes.json',
                 'analyses/behavior_main/episodes.json', 'analyses/evidence_main/metrics.json',
                 'analyses/evidence_ablation/metrics.json']:
        checks.append({'item': path, 'pass': (root / path).exists()})
    result = {'complete': all(x['pass'] for x in checks), 'checks': checks,
              'formal_study_new_physical_episodes': sum(x.get('new_native_executions', 0) for x in checks),
              'scope': 'Checks local complete raw data and required final diagnostics. '
                       'Exploratory development and expert certificates are preserved separately.'}
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    if result['complete']:
        requirements = [
            ('Inspect/reuse available frozen models', ['dependency_copy.json', 'executor_parity.json'],
             'Verified keep-out/waypoint seeds 42/43/44; no assumed approach checkpoint. Frozen parameters/buffers checked after every episode.'),
            ('Construct nontrivial joint tasks', ['tasks_v2/manifest.json', 'tasks_v2/audit.json', 'analyses/infeasible_inscribed_ball_certificate.json'],
             'Unseen certified combinations, held-out splits, independent native witnesses and separate actual-shape infeasible controls.'),
            ('Implement latent action agreement', ['protocols/core_plan.json', 'analyses/evidence_main/metrics.json'],
             'Four-sign-sector gradient search, multiple starts and no neural updates.'),
            ('Investigate task/trajectory alternatives', ['protocols/extensions/plan.json', 'analyses/evidence_combined/metrics.json'],
             'Exact native task costs, pose agreement, combined task/action validation and explicit common-action extensions; mechanism distinctions retained.'),
            ('Compare composed execution rules', ['main_core', 'main_extensions'],
             'Execute K/W, verified family selection, simultaneous adapters, full averaging, common actions and geometric correction.'),
            ('Study closed-loop consistency', ['analyses/evidence_validation/metrics.json', 'analyses/behavior_main/phases.json'],
             'Fixed/cold/warm/tail, causal encoder prefixes, code and observed physical-route changes.'),
            ('Measure faster methods', ['benchmarks', 'analyses/final/quantitative_tables.md'],
             'Random/CEM/hybrid, matched pair objectives, real forward/backward/native counts and dedicated phase-state latency.'),
            ('Investigate compatibility limitations', ['capability_dev', 'geometry_audit_full'],
             '960 finite fixed-code futures, minimum action/pose disagreement and independent feasible witnesses. No global non-intersection theorem claimed.'),
            ('Run controlled baselines/ablations', ['protocols', 'analyses/evidence_ablation/metrics.json', 'robustness_test'],
             'Three independently trained actor pairs, additional fixed-model RNGs, direct action optimization, original policies and expert feasibility references.'),
            ('Evaluate actual physical manipulation', ['analyses/evidence_main/metrics.json', 'analyses/behavior_main', 'visuals'],
             'Joint/goal/K/W success, force support, release/drop/contact, priors, stability, native videos and failure cases.'),
            ('Execute incremental GPU/native workflow', ['environment.json', 'protocols', 'tests_v6.xml'],
             'Actual L4 experiments, native simulation and physical replay; development corrections and test-selection chronology preserved.'),
            ('Deliver implementation and research findings', ['../CUBE_COMPOSITION_README.md', '../CUBE_COMPOSITION_RESULTS.md', 'delivery_index.json'],
             'Reproducible code/banks/configurations/raw outcomes/plots/videos and candid results. No optional joint network or unavailable third skill introduced.'),
        ]
        (root / 'requirements_audit.json').write_text(json.dumps({
            'complete': True,
            'items': [{'request_section': i, 'requirement': title, 'status': 'complete',
                       'evidence': evidence, 'scope': scope}
                      for i, (title, evidence, scope) in enumerate(requirements, 1)],
            'validation': str(args.out),
            'formal_new_native_episodes': result['formal_study_new_physical_episodes'],
            'archive_integrity': 'Checked separately by delivery --verify and --verify-archive receipts.'}, indent=2) + '\n')
    print(json.dumps({'complete': result['complete'], 'failed': [x['item'] for x in checks if not x['pass']]}))
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
