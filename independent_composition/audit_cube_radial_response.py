"""Offline held-world audit of existing base-amplitude action sensitivity.

This is NOT a planner, success score, learned component, or training loss.
At real recorded single-skill carry observations, vary only the existing base
amplitude input and compare original frozen generator first actions with each
training teacher's carry action. No physics integration is performed here.
Only aggregate errors are written; native arrays remain on the remote machine.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cube.cube_gan.expert import Style
from cube.cube_gan.style import decode_style
from cube_addons.evaluate import load_adapter
from .cube_diverse_demonstrations import DiverseSkillExpert, lookahead
from .cube_radial_demonstrations import RadialSkillExpert
from .inference_guard import inference_active
from .legacy import digest_file


def observations(raw, skill, frames):
    result = []
    for path in sorted((raw / skill / 'raw').glob('*.npz')):
        with np.load(path) as x:
            initial, goal, spec = x['initial_cube'], x['goal'], x['spec']
            base = x['base_z'].astype(np.float32)
            for i, code in enumerate(x['new_z']):
                length = int(x['lengths'][i]); context = x['contexts'][i, :length]
                eligible = (x['phase'][i, :length] == 4) & (context[:, 13] > .5)
                eligible &= (context[:, 10] > .15) & (context[:, 10] < .85)
                if skill == 'waypoint':
                    eligible &= context[:, 15] < .5
                indices = np.flatnonzero(eligible)
                if not len(indices):
                    continue
                indices = indices[np.unique(np.linspace(0, len(indices) - 1, min(frames, len(indices))).astype(int))]
                for t in indices:
                    result.append((x['states'][i, t].copy(), context[t].copy(),
                        x['poses'][i, t, 3:].copy(), base.copy(), code.copy(),
                        initial.copy(), goal.copy(), spec.copy()))
    assert result
    return result


def main():
    assert not inference_active(), 'Offline analysis cannot participate in planning'
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['raw', 'root', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--frames-per-trajectory', type=int, default=8)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); raw, root, out = Path(a.raw), Path(a.root), Path(a.out)
    assert (raw / 'DONE.json').exists() and not out.exists()
    out.mkdir(); torch.set_num_threads(2)
    nodes = [.25, 1., 2.]
    base_checkpoint = root / 'cube_diverse_v35_stageB/base_policy.pt'
    result = {'scope': __doc__, 'base_amplitude_code_magnitudes': nodes,
        'source_sha256': digest_file(__file__), 'observations_raw_config_sha256': digest_file(raw / 'config.json'),
        'no_physics_queries': True, 'no_model_updates': True,
        'observation_distribution': 'Held-world radial-teacher observations; compare conditional amplitude response, not native model success or overall model quality',
        'not_a_native_task_success_estimate': True, 'skills': {}}
    for skill in ['keepout', 'waypoint']:
        rows = observations(raw, skill, a.frames_per_trajectory)
        # Audit worlds must not have entered either matched training dataset.
        observed = set()
        for path in (raw / skill / 'raw').glob('*.npz'):
            with np.load(path) as x:
                observed.add(int(x['seed']))
        for family in ['angular', 'radial']:
            with np.load(root / 'cube_radial_v109_paired_data' / family / skill / 'train_contexts.npz') as x:
                assert observed.isdisjoint(map(int, x['seed']))
        states = []; contexts = []; bases = []; codes = []
        targets = {'angular': [], 'radial': []}
        for state, context, cube, base, code, initial, goal, spec in rows:
            for magnitude in nodes:
                bz = base.copy(); bz[0] = (-1. if bz[0] < 0 else 1.) * magnitude
                mode, amplitude, speed = decode_style(torch.tensor(bz))
                style = Style(int(mode), float(amplitude), float(speed))
                states.append(state); contexts.append(context); bases.append(bz); codes.append(code)
                for family, cls in [('angular', DiverseSkillExpert), ('radial', RadialSkillExpert)]:
                    teacher = cls(initial, goal, style, spec, code, skill)
                    points = teacher.points[:3] if skill == 'waypoint' else teacher.points
                    target = lookahead(points, cube, distance=.032 if skill == 'waypoint' else .022)
                    targets[family].append(np.clip((target - cube) * 15 * style.speed, -style.speed, style.speed))
        inputs = [torch.tensor(np.asarray(x), device=a.device, dtype=torch.float32) for x in [states, contexts, bases, codes]]
        targets = {f: np.asarray(x).reshape(len(rows), 3, 3) for f, x in targets.items()}
        info = {'held_observations': len(rows), 'worlds': len(observed), 'models': {}}
        for model_family in ['parent100', 'angular', 'radial']:
            checkpoint = (root / ('cube_magnitude_v100_' + skill + '_control') / 'best.pt'
                if model_family == 'parent100' else root / ('cube_radial_v110_' + model_family + '_' + skill) / 'best.pt')
            model, encoder, _ = load_adapter(checkpoint, base_checkpoint, a.device)
            model.eval().requires_grad_(False); encoder.eval().requires_grad_(False)
            predictions = []
            with torch.no_grad():
                for start in range(0, len(states), 256):
                    s, c, b, z = [x[start:start + 256] for x in inputs]
                    predictions.append(model(s, b, z, c)[:, 0, :3].cpu().numpy())
            pred = np.concatenate(predictions).reshape(len(rows), 3, 3)
            assert np.isfinite(pred).all() and all(np.isfinite(x).all() for x in targets.values())
            delta = pred[:, -1] - pred[:, 0]
            metrics = {'checkpoint_sha256': digest_file(checkpoint),
                'predicted_amplitude_delta_rms_by_xyz': np.sqrt(np.mean(delta ** 2, axis=0)).tolist()}
            for family, target in targets.items():
                expected_delta = target[:, -1] - target[:, 0]
                metrics[family + '_teacher'] = {
                    'action_rmse_by_xyz': np.sqrt(np.mean((pred - target) ** 2, axis=(0, 1))).tolist(),
                    'amplitude_difference_rmse_by_xyz': np.sqrt(np.mean((delta - expected_delta) ** 2, axis=0)).tolist(),
                    'teacher_amplitude_delta_rms_by_xyz': np.sqrt(np.mean(expected_delta ** 2, axis=0)).tolist(),
                }
            info['models'][model_family] = metrics
            del model, encoder
        result['skills'][skill] = info
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'offline_response_audit_complete': True}) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
