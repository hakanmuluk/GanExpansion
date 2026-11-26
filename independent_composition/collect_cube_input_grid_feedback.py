"""Native single-skill same-state private/amplitude supervision, training only.

Each actual held-state snapshot receives a Cartesian grid of existing Gaussian
private and base-amplitude inputs. The original own-skill expert supplies a
native 16-control target. A matched reference repeats the applied-input teacher
target at that same state, with exactly equal row counts. Both arms retain the
same older feedback. No joint instructions, joint scoring, network changes, or
inference mechanism are created. Raw native arrays stay on the collection host.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import importlib
import json
import multiprocessing as mp
from pathlib import Path
from statistics import NormalDist
import time
import numpy as np
import torch
from cube.cube_gan.expert import Style
from cube.cube_gan.style import decode_style
from cube_addons.contacts import contact_info, force_holding
from cube_addons.data import atomic_npz
from cube_addons.env import make_env, features, step_traced
from cube_addons.evaluate import load_adapter, load_contexts
from cube_addons.geometry import Observer, MAX_STEPS
from .collect_cube_switch_feedback import learner_rollout, LatentWalk
from .cube_radial_demonstrations import RadialSkillExpert
from .cube_recovery_expert import RecoverySkillExpert
from .observation_replay import install_single_skill_restore, restore_full_native_observation
from .inference_guard import inference_active
from .legacy import digest_file

KEYS = ['states', 'contexts', 'actions', 'mask', 'base_z', 'new_z']
_MODEL = _ENV = _ARGS = None


class RadialRecoveryExpert(RecoverySkillExpert):
    def __init__(self, initial, goal, style, spec, code, skill):
        super().__init__(initial, goal, style, spec, code, skill)
        # Reuse the existing native own-sphere recovery rule with the wider
        # training path. No recovery rule participates in deployed inference.
        self.points = RadialSkillExpert(initial, goal, style, spec, code, skill).points


def initialize(args):
    global _MODEL, _ENV, _ARGS
    assert not inference_active()
    _ARGS = args; torch.set_num_threads(1)
    _MODEL, _, _ = load_adapter(args.checkpoint, args.base, 'cpu')
    _MODEL.eval().requires_grad_(False)
    install_single_skill_restore(); _ENV = make_env()


def branch(case, traj, chosen, base, code):
    """Targets come from real native execution, never invented future actions."""
    args, env = _ARGS, _ENV; t = int(traj['snapshot_frames'][chosen])
    ob, info = restore_full_native_observation(env, traj['_full_native_snapshots'][chosen])
    contact_info(env, info)
    previous = traj['_previous_ee_exact'][t]
    state = features(env, ob, info, case['initial_cube'], previous)
    observer = Observer(case['initial_cube'], case['goal'], case['spec'], args.skill, force_holding)
    observer.has_lifted = bool(traj['contexts'][t, 14] > .5)
    observer.visited = bool(traj['contexts'][t, 15] > .5)
    observer.previous_holding = observer.is_holding(info)
    context = observer.features(info, t)
    state_error = float(np.max(np.abs(state - traj['states'][t])))
    context_error = float(np.max(np.abs(context - traj['contexts'][t])))
    assert max(state_error, context_error) < 2e-6, (state_error, context_error,
        np.flatnonzero(np.abs(state - traj['states'][t]) > 1e-7).tolist())
    mode, amplitude, speed = decode_style(torch.tensor(base))
    style = Style(int(mode), float(amplitude), float(speed))
    cls = RadialRecoveryExpert if args.skill == 'keepout' else RadialSkillExpert
    expert = cls(case['initial_cube'], case['goal'], style, case['spec'], code, args.skill)
    cube = np.asarray(info['privileged/block_0_pos'])
    progress = float(traj['contexts'][t, 10])
    expert.ready = bool(observer.has_lifted and observer.is_holding(info) and cube[2] > .08
        and (cube[2] >= .165 or progress > .12))
    target = []; held = []
    plan_visited = observer.visited
    for j in range(16):
        if j % 8 == 0:
            plan_visited = observer.visited
        action, _ = expert.action(info, observer,
            visited_override=plan_visited if args.skill == 'waypoint' else None)
        target.append(action)
        before = info['privileged/block_0_pos'].copy()
        result, path = step_traced(env, action); ob, _, _, _, info = result
        contact_info(env, info)
        observer.update(before, info['privileged/block_0_pos'], info, path)
        held.append(bool(force_holding(info)))
    safe = not observer.violated and not observer.table_violation
    value = dict(states=traj['states'][t].copy(), contexts=traj['contexts'][t].copy(),
        actions=np.asarray(target, np.float32), mask=np.ones(16, np.float32),
        base_z=np.asarray(base, np.float32).copy(), new_z=np.asarray(code, np.float32).copy())
    return value, safe, {'state_error': state_error, 'context_error': context_error,
        'guard_interventions': getattr(expert, 'guard_interventions', 0),
        'held_fraction': float(np.mean(held)), 'native_controls': 16}


def collect(job):
    assert not inference_active()
    case, initial_code, index = job; args = _ARGS
    walk = LatentWalk(initial_code, args.seed + index * 7919, args.rho)
    # The ordinary saved poses are float32. Their historical EE coordinate can
    # lose precision when the observation function computes a finite difference.
    # Capture the original argument without changing features, controls, or any
    # native integration. Each worker owns its module and restores it promptly.
    module = importlib.import_module('cube_addons.rollout')
    original_features = module.features; previous_history = []

    def record_features(env, ob, info, initial, previous):
        previous_history.append(None if previous is None else previous.copy())
        return original_features(env, ob, info, initial, previous)

    module.features = record_features
    try:
        row, traj = learner_rollout(_ENV, _MODEL, case, initial_code, args, walk)
    finally:
        module.features = original_features
    assert len(previous_history) == len(traj['states'])
    traj['_previous_ee_exact'] = previous_history
    out = Path(args.out)
    atomic_npz(out / ('learner_%04d.npz' % index),
        **{k: v for k, v in traj.items() if not k.startswith('_')},
        base_z=case['base_z'], initial=case['initial_cube'], goal=case['goal'], spec=case['spec'])
    frames = traj['snapshot_frames']; ctx = traj['contexts'][frames]
    eligible = (ctx[:, 13] > .5) & (ctx[:, 14] > .5) & (ctx[:, 10] > .1) & (ctx[:, 10] < .9)
    eligible &= frames <= MAX_STEPS - 16
    if args.skill == 'waypoint':
        eligible &= ctx[:, 15] < .5
    available = np.flatnonzero(eligible)
    rng = np.random.default_rng(args.seed + index)
    selected = sorted(rng.choice(available, min(args.frames, len(available)), replace=False))
    quantiles = [.02, .25, .49, .51, .75, .98]
    codes = [np.array([NormalDist().inv_cdf(q)], np.float32) for q in quantiles]
    amplitudes = [NormalDist().inv_cdf((1 + p) / 2) for p in [.1, .5, .9]]
    arrays = {family: {k: [] for k in KEYS} for family in ['grid', 'applied']}
    diagnostics = []; rejected = 0; skipped = 0; retained_frames = 0
    for chosen in selected:
        t = int(frames[chosen])
        applied, safe, diag = branch(case, traj, chosen, case['base_z'], traj['codes'][t])
        diagnostics.append(diag)
        if not safe:
            skipped += 1
            continue
        kept = 0
        for amplitude in amplitudes:
            base = case['base_z'].copy()
            base[0] = (-1 if base[0] < 0 else 1) * amplitude
            for code in codes:
                example, safe, diag = branch(case, traj, chosen, base, code)
                diagnostics.append(diag)
                if not safe:
                    rejected += 1
                    continue
                for key in KEYS:
                    arrays['grid'][key].append(example[key])
                    arrays['applied'][key].append(applied[key])
                kept += 1
        retained_frames += bool(kept)
    count = len(arrays['grid']['states'])
    if count:
        for family in arrays:
            values = {k: np.asarray(v, np.float32) for k, v in arrays[family].items()}
            assert all(np.isfinite(x).all() for x in values.values())
            atomic_npz(out / (family + '_%04d.npz' % index), **values)
        assert np.array_equal(arrays['grid']['states'], arrays['applied']['states'])
        assert np.array_equal(arrays['grid']['contexts'], arrays['applied']['contexts'])
    report = {'source_context_seed': int(case['seed']), 'source_own_success': bool(row['success']),
        'source_goal_success': bool(row['released_success']), 'selected_frames': len(selected),
        'retained_frames': retained_frames, 'matched_rows_per_arm': count,
        'rejected_grid_branches': rejected, 'rejected_applied_frames': skipped,
        'native_teacher_controls': sum(x['native_controls'] for x in diagnostics),
        'maximum_state_replay_error': max((x['state_error'] for x in diagnostics), default=0.),
        'maximum_context_replay_error': max((x['context_error'] for x in diagnostics), default=0.),
        'training_guard_interventions': sum(x['guard_interventions'] for x in diagnostics),
        'mean_branch_held_fraction': float(np.mean([x['held_fraction'] for x in diagnostics])) if diagnostics else None,
        'joint_labels_used': False}
    (out / ('record_%04d.json' % index)).write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    assert not inference_active()
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'base', 'contexts', 'out', 'append-feedback']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--skill', choices=['keepout', 'waypoint'], required=True)
    p.add_argument('--count', type=int, default=128)
    p.add_argument('--codes', type=int, default=2)
    p.add_argument('--frames', type=int, default=4)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=2821001)
    p.add_argument('--rho', type=float, default=.999999)
    p.add_argument('--snapshot-every', type=int, choices=[1, 8], default=8)
    p.add_argument('--learner-gate', type=int, choices=[1], default=1)
    a = p.parse_args(); out = Path(a.out)
    assert not out.exists() and Path(a.contexts).name == 'train_contexts.npz'
    assert a.count > 0 and a.codes > 0 and a.frames > 0 and 0 <= a.rho < 1
    out.mkdir(parents=True)
    cases = load_contexts(a.contexts); rng = np.random.default_rng(a.seed)
    chosen = rng.choice(len(cases), min(a.count, len(cases)), replace=False)
    selected = [cases[i] for i in chosen]
    val = Path(a.contexts).with_name('val_contexts.npz')
    if val.exists():
        assert {int(c['seed']) for c in selected}.isdisjoint(int(c['seed']) for c in load_contexts(val))
    checkpoint_hash = digest_file(a.checkpoint)
    config = {'args': vars(a), 'scope': __doc__, 'source_sha256': digest_file(__file__),
        'actor_sha256': checkpoint_hash, 'appended_feedback_sha256': digest_file(a.append_feedback),
        'private_gaussian_quantiles': [.02, .25, .49, .51, .75, .98],
        'base_amplitude_half_gaussian_quantiles': [.1, .5, .9],
        'base_route_sign_and_speed_fixed': True, 'target_controls_per_same_state': 16,
        'own_native_safety_filter_only': True, 'waypoint_query_states_before_completion': True,
        'matched_applied_reference': 'Same state/context/count, repeated native applied-input own-teacher target',
        'no_joint_instructions_or_labels': True, 'original_actor_execution_interval': 8}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    jobs = []
    for case in selected:
        values = np.abs(rng.standard_normal((a.codes + 1) // 2))
        initial = np.r_[-values, values][:a.codes]
        for code in initial:
            jobs.append((case, np.array([code], np.float32), len(jobs)))
    started = time.time(); records = []
    with mp.get_context('spawn').Pool(a.workers, initializer=initialize, initargs=(a,)) as pool:
        for row in pool.imap_unordered(collect, jobs, chunksize=1):
            records.append(row)
            progress = {'finished': len(records), 'total': len(jobs),
                'matched_rows_per_arm': sum(x['matched_rows_per_arm'] for x in records)}
            (out / 'status.json').write_text(json.dumps(progress) + '\n')
            print(json.dumps(progress), flush=True)
    with np.load(a.append_feedback) as x:
        previous = {k: x[k].copy() for k in KEYS}
    counts = {}
    for family in ['grid', 'applied']:
        dest = out / family; dest.mkdir()
        parts = {k: [] for k in KEYS}
        for path in sorted(out.glob(family + '_*.npz')):
            with np.load(path) as x:
                for k in KEYS:
                    parts[k].append(x[k])
        assert parts['states'], 'No own-safe matched queries'
        for k in KEYS:
            parts[k].append(previous[k])
        values = {k: np.concatenate(x) for k, x in parts.items()}
        counts[family] = len(values['states'])
        atomic_npz(dest / 'feedback.npz', **values)
        (dest / 'config.json').write_text(json.dumps({**config, 'feedback_arm': family}, indent=2) + '\n')
        (dest / 'DONE.json').write_text(json.dumps({'feedback_ready': True, 'original_feedback_format': True}) + '\n')
    assert counts['grid'] == counts['applied'] and digest_file(a.checkpoint) == checkpoint_hash
    summary = {'scope': __doc__, 'source_learner_rollouts': len(records),
        'new_matched_rows_per_arm': sum(x['matched_rows_per_arm'] for x in records),
        'total_rows_per_arm': counts['grid'], 'appended_old_rows': len(previous['states']),
        'native_teacher_controls': sum(x['native_teacher_controls'] for x in records),
        'source_own_success': float(np.mean([x['source_own_success'] for x in records])),
        'rejected_grid_branches': sum(x['rejected_grid_branches'] for x in records),
        'rejected_applied_frames': sum(x['rejected_applied_frames'] for x in records),
        'maximum_state_replay_error': max(x['maximum_state_replay_error'] for x in records),
        'maximum_context_replay_error': max(x['maximum_context_replay_error'] for x in records),
        'training_guard_interventions': sum(x['training_guard_interventions'] for x in records),
        'mean_branch_held_fraction': float(np.mean([x['mean_branch_held_fraction'] for x in records if x['mean_branch_held_fraction'] is not None])),
        'seconds': time.time() - started, 'actor_unchanged': True, 'new_learned_modules': 0,
        'matched_rows_states_contexts': True, 'no_joint_labels': True}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'matched_original_feedback_ready': True}) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
