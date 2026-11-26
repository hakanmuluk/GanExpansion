"""Pack equal-scene angular/radial own-skill datasets; arrays stay remote.

Retain an episode in both datasets only when both teachers solve that SAME
single skill. This is not joint-task filtering. Original validation pools are
copied unfiltered. Old feedback is used at the canonical amplitude slice where
the radial teacher recovers the old teacher, rather than introducing mutually
inconsistent amplitude labels. Both training arms receive identical feedback.
"""
import argparse
import json
from pathlib import Path
import shutil
from statistics import NormalDist
import numpy as np
from cube_addons.data import atomic_npz
from .pack_cube_diverse import CONTEXT_KEYS, EPISODE_KEYS
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['radial', 'angular', 'out', 'keepout-validation', 'waypoint-validation',
                 'keepout-feedback', 'waypoint-feedback']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args(); out = Path(a.out)
    roots = {f: Path(getattr(a, f)) for f in ['radial', 'angular']}
    assert not out.exists() and all((x / 'DONE.json').exists() for x in roots.values())
    configs = {f: json.loads((x / 'config.json').read_text()) for f, x in roots.items()}
    for key in ['contexts', 'codes', 'seed', 'jitter']:
        assert configs['radial']['args'][key] == configs['angular']['args'][key]
    out.mkdir(); (out / 'common_feedback').mkdir()
    report = {'scope': __doc__, 'no_joint_labels': True, 'matched_inclusion': True,
              'raw_config_sha256': {f: digest_file(x / 'config.json') for f, x in roots.items()},
              'skills': {}}
    for skill in ['keepout', 'waypoint']:
        paths = {f: sorted((x / skill / 'raw').glob('*.npz')) for f, x in roots.items()}
        assert [x.name for x in paths['radial']] == [x.name for x in paths['angular']]
        assert len(paths['radial']) == configs['radial']['args']['contexts']
        parts = {f: {k: [] for k in CONTEXT_KEYS + EPISODE_KEYS} for f in roots}
        contexts = {k: [] for k in CONTEXT_KEYS}
        kept = 0; requested = 0; strata = np.zeros(configs['radial']['args']['codes'], dtype=int)
        for rp, ap in zip(paths['radial'], paths['angular']):
            meta = [json.loads(x.with_suffix('.json').read_text()) for x in [rp, ap]]
            selected = np.logical_and(*[np.array([r['success'] for r in m['records']], bool) for m in meta])
            count = int(selected.sum()); requested += len(selected); kept += count; strata += selected
            with np.load(rp) as radial, np.load(ap) as angular:
                for key in CONTEXT_KEYS:
                    assert np.array_equal(radial[key], angular[key]), (rp.name, key)
                    contexts[key].append(radial[key].copy())
                assert np.array_equal(radial['new_z'], angular['new_z'])
                if count:
                    for family, data in [('radial', radial), ('angular', angular)]:
                        for key in CONTEXT_KEYS:
                            parts[family][key].append(np.repeat(data[key][None], count, axis=0))
                        for key in EPISODE_KEYS:
                            parts[family][key].append(data[key][selected])
        assert kept > 0 and np.all(strata > 0)
        validation = Path(getattr(a, skill + '_validation'))
        with np.load(validation) as v:
            assert set(map(int, v['seed'])).isdisjoint(map(int, np.stack(contexts['seed'])))
        for family in roots:
            dest = out / family / skill; dest.mkdir(parents=True)
            atomic_npz(dest / 'train.npz', **{k: np.concatenate(v) for k, v in parts[family].items()})
            atomic_npz(dest / 'train_contexts.npz', **{k: np.stack(v) for k, v in contexts.items()})
            shutil.copy2(validation, dest / 'val_contexts.npz')
        source_feedback = Path(getattr(a, skill + '_feedback'))
        with np.load(source_feedback) as values:
            feedback = {k: values[k].copy() for k in ['states', 'contexts', 'actions', 'mask', 'base_z', 'new_z']}
        amplitude_code = 1e-4 if skill == 'keepout' else NormalDist().inv_cdf(13. / 14.)
        feedback['base_z'][:, 0] = np.where(feedback['base_z'][:, 0] < 0, -amplitude_code, amplitude_code)
        target_feedback = out / 'common_feedback' / (skill + '.npz')
        atomic_npz(target_feedback, **feedback)
        report['skills'][skill] = {
            'requested_episodes_per_family': requested, 'matched_training_episodes_per_family': kept,
            'matched_successful_code_strata': strata.tolist(), 'contexts': len(paths['radial']),
            'validation_sha256': digest_file(validation), 'validation_unfiltered': True,
            'train_sha256': {f: digest_file(out / f / skill / 'train.npz') for f in roots},
            'feedback': {'source_sha256': digest_file(source_feedback), 'target_sha256': digest_file(target_feedback),
                'canonical_base_amplitude_code_magnitude': amplitude_code,
                'other_feedback_arrays_unchanged': True,
                'keepout_canonical_radius_label_tolerance_m': 3.6e-6 if skill == 'keepout' else 0.},
        }
    for family in roots:
        (out / family / 'DONE.json').write_text(json.dumps({'original_dataset_format': True,
            'matched_single_skill_inclusion': True, 'no_joint_labels': True}) + '\n')
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'matched_data_ready': True}) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
