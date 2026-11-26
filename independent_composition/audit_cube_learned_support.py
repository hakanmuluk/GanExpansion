"""OFFLINE finite-code learned-family capacity audit, never an inference rule.

Native rollouts measure original single-adapter outputs at a fixed code grid.
Any-code coverage is a diagnostic, not deployable success or proof of absence
outside the grid. No winning code is used as a policy initialization or label.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from cube_addons.env import make_env
from cube_composition.evaluate import load_cases
from cube_composition.rollout import rollout
from .cube_checkpoint_bank import CheckpointBank
from .inference_guard import tripwires
from .legacy import digest_file

_BANK = _ENV = None


def initialize(args):
    global _BANK, _ENV
    torch.set_num_threads(1)
    _BANK = CheckpointBank(args.keepout, args.waypoint, args.base, 'cpu')
    _ENV = make_env()


class FixedCode:
    def __init__(self, role, value):
        self.role, self.value = role, value

    def plan(self, state, contexts, base_z, unused_runtime):
        started = time.perf_counter()
        with tripwires(), torch.no_grad():
            actor = _BANK.k if self.role == 'keepout' else _BANK.w
            j = 0 if self.role == 'keepout' else 1
            chunk = actor(torch.as_tensor(state)[None], torch.as_tensor(base_z)[None],
                          torch.tensor([[self.value]], dtype=torch.float32),
                          torch.as_tensor(contexts[j])[None])[0].numpy()
        code = np.array([self.value if j == 0 else .05, self.value if j == 1 else .05])
        return chunk, code, {'seconds': time.perf_counter() - started, 'simulator_queries': 0}


def execute(job):
    case, context, role, code = job
    row, _, _ = rollout(_ENV, _BANK, case, planner_override=FixedCode(role, code), execute=8)
    own = bool(row['released_success'] and row[role + '_satisfied']
               and not row['dropped'] and not row['robot_table_contact_in_transport'])
    return {'context': context, 'pair_id': int(case['paired_id']), 'variant': str(case['paired_variant']),
            'role': role, 'code': code, 'own_success': own, 'joint_success': bool(row['success']),
            'goal_success': bool(row['released_success'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['tasks', 'base', 'keepout', 'waypoint', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    out = Path(args.out)
    assert not out.exists() and not Path(args.tasks).name.startswith(('test', 'final'))
    out.mkdir(parents=True)
    values = [-2.5, -1.5, -.8, -.05, .05, .8, 1.5, 2.5]
    cases = load_cases(args.tasks, args.count)
    jobs = [(case, i, role, code) for i, case in enumerate(cases)
            for role in ['keepout', 'waypoint'] for code in values]
    (out / 'config.json').write_text(json.dumps({'args': vars(args), 'scope': __doc__,
        'codes': values, 'source_sha256': digest_file(__file__), 'task_sha256': digest_file(args.tasks),
        'models': {name: digest_file(getattr(args, name)) for name in ['base', 'keepout', 'waypoint']},
        'new_policy_initializations_or_training_labels_produced': False}, indent=2) + '\n')
    records = []
    started = time.time()
    with mp.get_context('spawn').Pool(args.workers, initializer=initialize, initargs=(args,)) as pool:
        for row in pool.imap_unordered(execute, jobs, chunksize=1):
            records.append(row)
            (out / 'status.json').write_text(json.dumps({'finished': len(records), 'total': len(jobs)}) + '\n')
    metrics = []
    for role in ['keepout', 'waypoint']:
        for variant in ['inner', 'outer']:
            rows = [r for r in records if r['role'] == role and r['variant'] == variant]
            ids = {r['context'] for r in rows}
            metrics.append({'role': role, 'variant': variant, 'native_episodes': len(rows),
                'contexts': len(ids), 'contexts_with_any_joint_grid_code': sum(
                    any(r['joint_success'] for r in rows if r['context'] == i) for i in ids),
                'joint_successful_grid_rollouts': sum(r['joint_success'] for r in rows)})
    (out / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    report = {'scope': __doc__, 'metrics': metrics, 'seconds': time.time() - started}
    (out / 'aggregate.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'native_rollouts': len(records), 'offline_only': True}) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
