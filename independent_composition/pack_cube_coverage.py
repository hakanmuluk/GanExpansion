"""Append own-skill coverage demonstrations, retaining the earlier dataset.

One independent skill at a time. Existing successful Stage A episodes are retained, with
the same angular latent labels. Fresh validation cases mix the earlier broader
distribution and coverage of original small targets; no joint outcomes are used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cube_addons.data import atomic_npz
from cube_addons.env import make_env, close_env
from cube_addons.probe import case
from .collect_cube_coverage import coverage_spec as waypoint_spec
from .pack_cube_diverse import CONTEXT_KEYS, EPISODE_KEYS
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['previous', 'raw', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--val-seed', type=int, default=2001001)
    p.add_argument('--skill', choices=['keepout', 'waypoint'], default='waypoint')
    args = p.parse_args()
    skill = args.skill
    if skill == 'keepout':
        from .collect_cube_keepout_coverage import coverage_spec
    else:
        coverage_spec = waypoint_spec
    previous, raw, out = [Path(v).resolve() for v in [args.previous, args.raw, args.out]]
    assert (raw / 'DONE.json').exists() and not out.exists()
    out.mkdir(parents=True)
    # The other skill data are available for the original trainer's
    # conventional two-skill directory structure, without copying raw arrays.
    other = 'keepout' if skill == 'waypoint' else 'waypoint'
    (out / other).symlink_to(previous / other, target_is_directory=True)
    dest = out / skill
    dest.mkdir()
    with np.load(previous / skill / 'train.npz') as data:
        parts = {k: [data[k]] for k in EPISODE_KEYS + CONTEXT_KEYS}
        old_episodes = len(data['lengths'])
    with np.load(previous / skill / 'train_contexts.npz') as data:
        contexts = {k: [data[k]] for k in CONTEXT_KEYS}
    kept, total, radius_strata = 0, 0, {'small_le_025': [0, 0], 'large_gt_025': [0, 0]}
    for f in sorted((raw / skill / 'raw').glob('*.npz')):
        meta = json.loads(f.with_suffix('.json').read_text())
        selected = np.array([r['success'] for r in meta['records']], dtype=bool)
        total += len(selected)
        count = int(selected.sum())
        kept += count
        with np.load(f) as data:
            label = 'small_le_025' if data['spec'][3] <= .025 else 'large_gt_025'
            radius_strata[label][0] += count
            radius_strata[label][1] += len(selected)
            for k in CONTEXT_KEYS:
                contexts[k].append(data[k][None])
                if count:
                    parts[k].append(np.repeat(data[k][None], count, axis=0))
            if count:
                for k in EPISODE_KEYS:
                    parts[k].append(data[k][selected])
    assert kept and all(total > 0 for _, total in radius_strata.values())
    atomic_npz(dest / 'train.npz', **{k: np.concatenate(v) for k, v in parts.items()})
    train_contexts = {k: np.concatenate(v) for k, v in contexts.items()}
    assert len(np.unique(train_contexts['seed'])) == len(train_contexts['seed'])
    atomic_npz(dest / 'train_contexts.npz', **train_contexts)
    with np.load(previous / skill / 'val_contexts.npz') as data:
        old_val = {k: data[k] for k in CONTEXT_KEYS}
    fresh = {k: [] for k in CONTEXT_KEYS}
    env = make_env()
    try:
        for seed in range(args.val_seed, args.val_seed + len(old_val['seed'])):
            c, style = case(env, seed)
            c.update(spec=coverage_spec(c, style, skill, seed), force_contacts=True,
                     macro_waypoint=skill == 'waypoint', planning_interval=8)
            for k in CONTEXT_KEYS:
                fresh[k].append(np.asarray(c[k]))
    finally:
        close_env(env)
    # Interleaving makes partial original-trainer validation cover both banks.
    val = {k: np.stack([old_val[k], np.stack(fresh[k])], axis=1).reshape(
        (-1, *old_val[k].shape[1:])) for k in CONTEXT_KEYS}
    assert set(map(int, val['seed'])).isdisjoint(map(int, train_contexts['seed']))
    atomic_npz(dest / 'val_contexts.npz', **val)
    report = {'scope': __doc__, 'skill': skill, 'previous_episodes_retained': old_episodes,
              'new_episodes_requested': total, 'new_successful_episodes': kept,
              'own_teacher_success_by_radius': radius_strata,
              'training_episodes': old_episodes + kept, 'unfiltered_validation_contexts': len(val['seed']),
              'train_sha256': digest_file(dest / 'train.npz'),
              'raw_config_sha256': digest_file(raw / 'config.json'),
              'source_sha256': digest_file(__file__), 'joint_labels_used': False}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'coverage_packed': True}) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
