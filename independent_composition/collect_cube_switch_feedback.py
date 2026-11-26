"""Own-skill corrective data after changing existing latent codes, training only.

Original actor/rollout and feedback tensor format. A Gaussian code walk explores
states using the chosen donor policy. The target expert receives only its own
instruction, sampled independently of the donor instruction when requested.
Snapshots may receive multiple counterfactual codes at exactly the same state,
region and base code. Only target-safe native windows are retained. No joint
success labels, added neural components or inference mechanism are produced.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from cube.cube_gan.style import decode_style
from cube.cube_gan.expert import Style
from cube_addons.contacts import contact_info, force_holding
from cube_addons.data import atomic_npz, modifier_codes
from cube_addons.env import make_env, restore, step_traced, features, snapshot
from cube_addons.evaluate import load_adapter, load_contexts
from cube_addons.geometry import Observer, MAX_STEPS, CUBE_BOUND_RADIUS
from cube_addons.rollout import rollout
from .cube_diverse_demonstrations import DiverseSkillExpert
from .legacy import digest_file
from .observation_replay import restore_full_native_observation, install_single_skill_restore

_MODEL = _ENV = _ARGS = None


class LatentWalk:
    def __init__(self, initial, seed, rho):
        self.current = np.asarray(initial, np.float32).copy()
        self.rng = np.random.default_rng(seed)
        self.rho = rho

    def standard_normal(self, shape):
        self.current = (self.rho * self.current + np.sqrt(1 - self.rho ** 2)
                        * self.rng.standard_normal(shape)).astype(np.float32)
        return self.current.copy()


def initialize(args):
    global _MODEL, _ENV, _ARGS
    torch.set_num_threads(1)
    _ARGS = args
    _MODEL, _, _ = load_adapter(args.checkpoint, args.base, 'cpu')
    _MODEL.eval().requires_grad_(False)
    install_single_skill_restore()
    _ENV = make_env()


def learner_rollout(env, model, case, initial_code, args, walk):
    """Record more states without changing the original eight-control execution.

    Dense capture wraps the native step only to copy its BEFORE state. The
    original forward calls, native actions, integration and replanning schedule
    are unchanged. Each worker owns its process and restores the step function.
    """
    kwargs = dict(gate=args.learner_gate, resample=True, rng=walk, save_snapshots=False)
    module = importlib.import_module('cube_addons.rollout')
    original_step = module.step_traced
    captures, frames, force_caches, full_snapshots = [], [], [], []
    counter = 0

    def record_step(active_env, action):
        nonlocal counter
        if counter % args.snapshot_every == 0:
            captures.append(snapshot(active_env))
            frames.append(counter)
            force_caches.append(active_env._data.cfrc_ext.copy())
            native_copy = mujoco.MjData(active_env._model)
            mujoco.mj_copyData(native_copy, active_env._model, active_env._data)
            full_snapshots.append(native_copy)
        counter += 1
        return original_step(active_env, action)

    module.step_traced = record_step
    try:
        row, traj = rollout(env, model, case, case['spec'], args.skill, initial_code, **kwargs)
    finally:
        module.step_traced = original_step
    traj['snapshots'] = np.asarray(captures)
    traj['snapshot_frames'] = np.asarray(frames, np.int16)
    traj['snapshot_force_caches'] = np.asarray(force_caches)
    traj['_full_native_snapshots'] = full_snapshots
    return row, traj


def collect(job):
    case, initial_code, index = job
    args, env = _ARGS, _ENV
    walk = LatentWalk(initial_code, args.seed + index * 7919, args.rho)
    source_args = SimpleNamespace(**vars(args))
    source_args.skill = args.learner_skill or args.skill
    row, traj = learner_rollout(env, _MODEL, case, initial_code, source_args, walk)
    out = Path(args.out)
    atomic_npz(out / ('learner_%04d.npz' % index),
        **{k: v for k, v in traj.items() if not k.startswith('_')}, base_z=case['base_z'],
        initial=case['initial_cube'], goal=case['goal'], spec=case['spec'], seed=case['seed'])
    rng = np.random.default_rng(args.seed + index)
    frames = traj['snapshot_frames']
    available = np.arange(len(frames))
    context = traj['contexts'][frames]
    carry = np.flatnonzero((context[:, 13] > .5) & (context[:, 14] > .5)
                          & (context[:, 10] > .1) & (context[:, 10] < .9))
    if args.local_keepout_contexts or args.transport_only:
        selected = rng.choice(carry, min(len(carry), args.frames), replace=False).tolist()
    elif args.focus_own_boundary and args.skill == 'keepout' and len(carry):
        distance = np.linalg.norm(traj['poses'][frames[carry], 3:] - case['spec'][:3], axis=1)
        clearance = distance - case['spec'][3] - CUBE_BOUND_RADIUS
        eligible = np.flatnonzero(clearance >= args.minimum_query_clearance)
        order = eligible[np.argsort(clearance[eligible])]
        selected = carry[order[:min(len(order), args.frames // 2)]].tolist()
    else:
        selected = rng.choice(carry, min(len(carry), args.frames // 2), replace=False).tolist()
    if not (args.local_keepout_contexts or args.transport_only):
        remaining = np.setdiff1d(available, selected)
        selected += rng.choice(remaining, min(len(remaining), args.frames - len(selected)), replace=False).tolist()
    samples = {k: [] for k in ['states', 'contexts', 'actions', 'mask', 'base_z', 'new_z']}
    rejected, maximum_replay_error = 0, 0.
    guard_interventions, code_changes, base_changes, actual_queries = 0, [], [], 0
    frame_queries, frame_bases = {}, {}
    for chosen, region_index, query_index in [(i, g, j) for i in sorted(selected)
            for g in range(args.query_regions) for j in range(args.query_codes)]:
        query_key = (chosen, region_index)
        if query_index and frame_queries.get(query_key) is None:
            continue
        t = int(frames[chosen])
        ob, info = restore_full_native_observation(env, traj['_full_native_snapshots'][chosen])
        contact_info(env, info)
        error = float(np.max(np.abs(info['privileged/block_0_pos'] - traj['poses'][t, 3:])))
        assert error < 5e-6
        maximum_replay_error = max(maximum_replay_error, error)
        query_spec = case['spec'].copy() if not query_index else frame_queries[query_key][0].copy()
        if args.local_keepout_contexts and not query_index:
            # The donor's waypoint instruction is discarded. Sample the target
            # K region from physical state alone, independently of that donor.
            cube = np.asarray(info['privileged/block_0_pos'])
            found = False
            for attempt in range(64):
                radius = rng.uniform(.014, .030)
                direction = rng.standard_normal(3)
                direction /= max(float(np.linalg.norm(direction)), 1e-9)
                center = cube+(radius+CUBE_BOUND_RADIUS+rng.uniform(.003, .035))*direction
                bound = radius+CUBE_BOUND_RADIUS+.004
                if (.17 <= center[2] <= .23 and
                    np.linalg.norm(center-case['initial_cube']) > bound and
                    np.linalg.norm(center-case['goal']) > bound):
                    query_spec = np.r_[center, radius]
                    found = True
                    break
            if not found:
                frame_queries[query_key] = None
                continue
        observer = Observer(case['initial_cube'], case['goal'], query_spec, args.skill, force_holding)
        observer.has_lifted = bool(traj['contexts'][t, 14] > .5)
        observer.visited = (bool(traj['contexts'][t, 15] > .5)
                            if source_args.skill == args.skill else False)
        observer.previous_holding = observer.is_holding(info)
        code = (modifier_codes(args.seed+index*7919+t)[query_index].copy() if args.query_codes == 4
                else rng.standard_normal(1).astype(np.float32) if args.query_code_mode == 'independent'
                else traj['codes'][t].copy())
        code_changes.append(float(np.abs(code - traj['codes'][t]).mean()))
        query_base_z = (frame_bases[chosen].copy() if chosen in frame_bases else
            rng.standard_normal(2).astype(np.float32) if args.query_base_code_mode == 'independent'
            else case['base_z'])
        frame_bases[chosen] = query_base_z.copy()
        if not query_index:
            frame_queries[query_key] = (query_spec.copy(), query_base_z.copy())
        base_changes.append(float(np.abs(query_base_z - case['base_z']).mean()))
        mode, amplitude, speed = decode_style(torch.as_tensor(query_base_z))
        style = Style(int(mode), float(amplitude), float(speed))
        expert_type = DiverseSkillExpert
        if args.guard_own_keepout:
            from .cube_recovery_expert import RecoverySkillExpert
            expert_type = RecoverySkillExpert
        expert = expert_type(case['initial_cube'], case['goal'], style, query_spec, code, args.skill)
        cube = np.asarray(info['privileged/block_0_pos'])
        progress = float(traj['contexts'][t, 10])
        expert.ready = bool(observer.has_lifted and observer.is_holding(info) and cube[2] > .08
                            and (cube[2] >= .165 or progress > .12))
        expert.releasing = bool((args.skill == 'keepout' or observer.visited) and observer.has_lifted
            and cube[2] < .04 and np.linalg.norm(cube[:2] - case['goal'][:2]) < .035)
        target, teacher_states, teacher_contexts = [], [], []
        plan_visited = observer.visited
        previous_ee = traj['poses'][t - 1, :3].copy() if t else None
        controls = min(args.query_controls, MAX_STEPS - t) if args.query_controls > 16 else 16
        for j in range(controls):
            teacher_states.append(traj['states'][t] if j == 0 and not args.local_keepout_contexts else
                                  features(env, ob, info, case['initial_cube'], previous_ee))
            teacher_contexts.append(traj['contexts'][t] if j == 0 and not args.local_keepout_contexts
                                    else observer.features(info, t + j))
            if j % 8 == 0:
                plan_visited = observer.visited
            action, _ = expert.action(info, observer,
                visited_override=plan_visited if args.skill == 'waypoint' else None)
            target.append(action)
            before = info['privileged/block_0_pos'].copy()
            previous_ee = info['proprio/effector_pos'].copy()
            result, path = step_traced(env, action)
            ob, _, _, _, info = result
            contact_info(env, info)
            observer.update(before, info['privileged/block_0_pos'], info, path)
        guard_interventions += getattr(expert, 'guard_interventions', 0)
        actual_queries += controls
        starts = list(range(0, max(1, controls - 16 + 1), args.query_stride))
        if observer.violated or observer.table_violation:
            rejected += len(starts)
            continue
        for start in starts:
            length = min(16, controls - start)
            action_target = np.zeros((16, 5), np.float32)
            action_target[:length] = target[start:start + length]
            mask = np.zeros(16, np.float32)
            mask[:length] = 1
            samples['states'].append(teacher_states[start])
            samples['contexts'].append(teacher_contexts[start])
            samples['actions'].append(action_target)
            samples['mask'].append(mask)
            samples['base_z'].append(query_base_z)
            samples['new_z'].append(code)
    if samples['states']:
        atomic_npz(out / ('queries_%04d.npz' % index), **{k: np.asarray(v, np.float32) for k, v in samples.items()})
    switches = int((np.linalg.norm(np.diff(traj['codes'], axis=0), axis=-1) > 1e-6).sum())
    row.update(context_seed=int(case['seed']), query_chunks=len(samples['states']),
        rejected_own_unsafe_chunks=rejected, expert_action_queries=actual_queries,
        latent_switches=switches, maximum_snapshot_error_m=maximum_replay_error,
        other_skill_observed=False, training_guard_interventions=guard_interventions,
        target_expert_observed_donor_instruction=False,
        mean_requested_code_change=float(np.mean(code_changes)) if code_changes else 0.,
        mean_requested_base_code_change=float(np.mean(base_changes)) if base_changes else 0.)
    (out / ('record_%04d.json' % index)).write_text(json.dumps(row, indent=2) + '\n')
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'base', 'contexts', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--skill', choices=['keepout', 'waypoint'], required=True)
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--codes', type=int, default=2)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--seed', type=int, default=2501001)
    p.add_argument('--rho', type=float, default=.98)
    p.add_argument('--query-code-mode', choices=['applied', 'independent'], default='applied')
    p.add_argument('--guard-own-keepout', action='store_true')
    p.add_argument('--query-base-code-mode', choices=['applied', 'independent'], default='applied')
    p.add_argument('--query-controls', type=int, default=16)
    p.add_argument('--query-stride', type=int, default=8)
    p.add_argument('--focus-own-boundary', action='store_true')
    p.add_argument('--append-feedback')
    p.add_argument('--learner-gate', type=int, choices=[0, 1], default=1,
        help='Original binary switch during training-state collection; zero explores base-policy states')
    p.add_argument('--snapshot-every', type=int, choices=[1, 2, 4, 8], default=8,
        help='Training-only state capture interval; original eight-control execution stays unchanged')
    p.add_argument('--minimum-query-clearance', type=float, default=.006)
    p.add_argument('--learner-skill', choices=['keepout', 'waypoint'],
        help='Optional donor task for state coverage; target expert still sees its own region only')
    p.add_argument('--local-keepout-contexts', action='store_true',
        help='Sample K regions around observed held states independently of the donor instruction')
    p.add_argument('--target-validation-contexts',
        help='Exclude target-task validation physical seeds from donor training contexts')
    p.add_argument('--query-codes', type=int, choices=[1, 4], default=1,
        help='Four uses stratified Gaussian paired signs at the SAME state/region/base input')
    p.add_argument('--query-regions', type=int, choices=[1, 2], default=1,
        help='Two samples distinct own instructions at the SAME state/base/private code set')
    p.add_argument('--transport-only', action='store_true',
        help='Sample every query frame from observed held/lifted transport states')
    args = p.parse_args()
    assert Path(args.contexts).name == 'train_contexts.npz' and 0 <= args.rho < 1
    assert not args.guard_own_keepout or args.skill == 'keepout'
    assert args.query_controls >= 16 and args.query_stride >= 1
    assert args.count > 0 and args.codes > 0
    assert 0 <= args.minimum_query_clearance <= .05
    assert not args.local_keepout_contexts or args.skill == 'keepout'
    assert not args.learner_skill or args.learner_skill == args.skill or args.local_keepout_contexts
    assert args.query_codes == 1 or args.query_code_mode == 'independent'
    assert args.query_regions == 1 or (args.local_keepout_contexts and args.query_codes == 4)
    if args.append_feedback:
        previous_config = json.loads(Path(args.append_feedback).with_name('config.json').read_text())
        assert previous_config['args']['skill'] == args.skill
        assert digest_file(previous_config['args']['base']) == digest_file(args.base)
    out = Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    cases = load_contexts(args.contexts)
    if args.target_validation_contexts:
        forbidden_seeds = {int(c['seed']) for c in load_contexts(args.target_validation_contexts)}
        cases = [c for c in cases if int(c['seed']) not in forbidden_seeds]
    assert cases
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(cases), min(args.count, len(cases)), replace=False)
    selected = [cases[i] for i in chosen]
    val = Path(args.contexts).with_name('val_contexts.npz')
    if val.exists():
        assert {int(c['seed']) for c in selected}.isdisjoint(int(c['seed']) for c in load_contexts(val))
    checksum = digest_file(args.checkpoint)
    (out / 'config.json').write_text(json.dumps({'args': vars(args), 'scope': __doc__,
        'source_sha256': digest_file(__file__), 'actor_sha256': checksum,
        'appended_feedback_sha256': digest_file(args.append_feedback) if args.append_feedback else None,
        'train_contexts_sha256': digest_file(args.contexts), 'training_contexts_only': True,
        'donor_skill': args.learner_skill or args.skill,
        'target_skill': args.skill,
        'target_region_independent_of_donor_instruction': args.local_keepout_contexts,
        'target_validation_contexts_sha256': digest_file(args.target_validation_contexts) if args.target_validation_contexts else None,
        'no_joint_data': True}, indent=2) + '\n')
    jobs = []
    for case in selected:
        if args.codes % 2:
            initial_codes = rng.standard_normal(args.codes)
        else:
            values = np.abs(rng.standard_normal(args.codes // 2))
            initial_codes = np.r_[-values, values]
        for code in initial_codes:
            jobs.append((case, np.array([code], np.float32), len(jobs)))
    started = time.time()
    records = []
    with mp.get_context('spawn').Pool(args.workers, initializer=initialize, initargs=(args,)) as pool:
        for row in pool.imap_unordered(collect, jobs, chunksize=1):
            records.append(row)
            (out / 'status.json').write_text(json.dumps({'finished': len(records), 'total': len(jobs),
                'query_chunks': sum(r['query_chunks'] for r in records)}) + '\n')
    arrays = {k: [] for k in ['states', 'contexts', 'actions', 'mask', 'base_z', 'new_z']}
    for f in sorted(out.glob('queries_*.npz')):
        with np.load(f) as data:
            for k in arrays:
                arrays[k].append(data[k])
    assert arrays['states'], 'No own-safe expert queries; inspect records'
    appended = 0
    if args.append_feedback:
        with np.load(args.append_feedback) as data:
            appended = len(data['states'])
            for k in arrays:
                arrays[k].append(data[k])
    merged = {k: np.concatenate(v) for k, v in arrays.items()}
    atomic_npz(out / 'feedback.npz', **merged)
    assert digest_file(args.checkpoint) == checksum
    report = {'scope': __doc__, 'actual_learner_rollouts': len(records),
        'own_success_with_switching': float(np.mean([r['success'] for r in records])),
        'goal_success_with_switching': float(np.mean([r['released_success'] for r in records])),
        'expert_action_queries': sum(r['expert_action_queries'] for r in records),
        'retained_chunks': sum(r['query_chunks'] for r in records),
        'appended_previous_feedback_chunks': appended,
        'total_feedback_chunks': sum(r['query_chunks'] for r in records) + appended,
        'rejected_own_unsafe_chunks': sum(r['rejected_own_unsafe_chunks'] for r in records),
        'mean_latent_switches': float(np.mean([r['latent_switches'] for r in records])),
        'training_guard_interventions': sum(r['training_guard_interventions'] for r in records),
        'mean_requested_code_change': float(np.mean([r['mean_requested_code_change'] for r in records])),
        'mean_requested_base_code_change': float(np.mean([r['mean_requested_base_code_change'] for r in records])),
        'maximum_snapshot_error_m': max(r['maximum_snapshot_error_m'] for r in records),
        'seconds': time.time() - started, 'actor_unchanged': True, 'joint_labels_used': False}
    report['learner_binary_gate'] = args.learner_gate
    report['donor_skill'] = args.learner_skill or args.skill
    report['target_skill'] = args.skill
    report['target_region_independent_of_donor_instruction'] = args.local_keepout_contexts
    report['counterfactual_codes_per_snapshot'] = args.query_codes
    report['counterfactual_regions_per_snapshot'] = args.query_regions
    report['source_task_success'] = report['own_success_with_switching']
    report['source_metrics_are_target_skill_metrics'] = (args.learner_skill or args.skill) == args.skill
    if args.skill == 'keepout':
        c = merged['contexts']
        clearance = np.linalg.norm(c[:, 4:7] * .15, axis=1) - c[:, 3] * .1 - CUBE_BOUND_RADIUS
        held = (c[:, 13] > .5) & (c[:, 14] > .5)
        report['held_feedback_windows'] = int(held.sum())
        report['held_windows_within_15mm_of_boundary'] = int((held & (clearance >= 0) & (clearance <= .015)).sum())
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'feedback_ready': True, 'original_feedback_format': True}) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
