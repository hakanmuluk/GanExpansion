"""Independent original-environment replay of every frozen TwoRoom trajectory.

Benchmark verification only. All simulator operations are outside the policy.
Raw positions/actions and per-episode records remain on the evaluation host.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from tworoom_gan.env import make_reference, verify_reference
from tworoom_composition.audit import replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); root = Path(a.root); out = Path(a.out); assert not out.exists()
    out.mkdir(); started = time.time()
    protocol = json.loads((root / 'tworoom_frozen_v129_protocol/freeze.json').read_text())
    for name, checksum in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == checksum
    report = {'scope': __doc__, 'reference_dynamics': verify_reference(samples=3000, seed=17),
              'benchmarks': {}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    env = make_reference()
    try:
        for folder in sorted((root / 'tworoom_frozen_v129_eval').glob('*/*')):
            assert (folder / 'DONE.json').exists() and (folder / 'frozen_proof.json').exists()
            result = replay(folder, env)
            saved = json.loads((folder / 'records.json').read_text())
            assert len(result['rows']) == len(saved) == 64
            for row, actual in zip(saved, result['rows']):
                measured = actual['independent']
                assert row['keepout_satisfied'] == (not measured['keepout_violation'])
                assert row['wall_violation'] == measured['strict_wall_violation']
                assert row['waypoint_satisfied'] == (measured['visited'] and not measured['early_goal_entry'])
            report['benchmarks'][str(folder.relative_to(root / 'tworoom_frozen_v129_eval'))] = {
                'episodes': len(saved), 'all_native_positions_and_outcomes_match': result['all_match'],
                'maximum_position_error_pixels': max(v['max_position_error'] for v in result['rows']),
                'native_joint_successes': sum(v['independent']['success'] for v in result['rows']),
                'native_goal_successes': sum(v['independent']['goal_success'] for v in result['rows']),
                'premature_native_done': sum(v['premature_native_done'] for v in result['rows'])}
            (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'benchmark': str(folder.relative_to(folder.parents[1])),
                              **report['benchmarks'][str(folder.relative_to(root / 'tworoom_frozen_v129_eval'))]}), flush=True)
    finally:
        env.close()
    assert len(report['benchmarks']) == 12
    report['seconds'] = time.time() - started
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'episodes': 768, 'all12_native_replays_match': True,
        'raw_arrays_remain_remote': True}) + '\n')


if __name__ == '__main__':
    main()
