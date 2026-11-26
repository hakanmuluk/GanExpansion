"""Wait for Stage B training, then run the predeclared original-forward evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ['root', 'source', 'stage-a', 'stage-b', 'out']:
        p.add_argument('--' + flag, required=True)
    args = p.parse_args()
    root, source, stage_a, stage_b, out = [Path(v).resolve() for v in
        [args.root, args.source, args.stage_a, args.stage_b, args.out]]
    assert not out.exists()
    receipt = out.with_name(out.name + '_queue.json')
    state = {'pid': os.getpid(), 'args': vars(args), 'status': 'waiting_for_stage_B'}

    def save():
        temp = receipt.with_suffix('.tmp.json')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(receipt)

    save()
    while not (stage_b / 'DONE.json').exists():
        current = json.loads((stage_b / 'status.json').read_text())
        if any(r['status'] == 'failed' for r in current['stages']):
            state['status'] = 'upstream_failed'
            save()
            raise RuntimeError('Stage B failed; inspect saved logs before retrying')
        if not Path('/proc/' + str(current['pid']) + '/cmdline').exists():
            raise RuntimeError('Stage B exited without DONE')
        time.sleep(10)
    command = [sys.executable, '-m', 'independent_composition.evaluate_cube_data_ablation',
        '--root', str(root), '--source', str(source), '--stage-a', str(stage_a),
        '--checkpoint-dir', str(stage_b), '--base-checkpoint', str(stage_b / 'base_policy.pt'),
        '--out', str(out), '--count', '8', '--workers', '2']
    with out.with_suffix('.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        state.update(status='evaluating', evaluation_pid=proc.pid, command=command)
        save()
        code = proc.wait()
    state.update(status='complete' if code == 0 else 'failed', returncode=code)
    save()
    if code:
        raise RuntimeError('Stage B evaluation failed; inspect saved log')


if __name__ == '__main__':
    main()
