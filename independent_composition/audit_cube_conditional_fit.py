"""Offline known-code action fit on recorded native single-skill targets.

No teacher approximation, native queries, fitting, or inference use. Raw data
stay on the collection host. Group-centered errors measure conditional response
separately from the mean action at each identical observation/context.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cube_addons.evaluate import load_adapter
from .legacy import digest_file


def metrics(pred, target, groups):
    result = {}
    for horizon in [1, 8, 16]:
        delta = pred[:, :horizon] - target[:, :horizon]
        result['prefix_' + str(horizon)] = {
            'action_mse': float(np.mean(delta ** 2)),
            'xyz_mse': float(np.mean(delta[..., :3] ** 2)),
            'per_action_mse': np.mean(delta ** 2, axis=(0, 1)).tolist(),
        }
    centered_error = []; centered_target = []; centered_pred = []
    for ids in groups:
        if len(ids) < 2:
            continue
        p = pred[ids, :8]; t = target[ids, :8]
        p = p - p.mean(0); t = t - t.mean(0)
        centered_error.append(np.mean((p - t) ** 2))
        centered_target.append(np.mean(t ** 2)); centered_pred.append(np.mean(p ** 2))
    result['within_state_prefix8'] = {
        'groups': len(centered_error),
        'conditional_mse': float(np.mean(centered_error)),
        'teacher_conditional_variance': float(np.mean(centered_target)),
        'model_conditional_variance': float(np.mean(centered_pred)),
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--held-prefix', default='cube_conditional_v121_held')
    p.add_argument('--require-held', action='store_true')
    p.add_argument('--extra-prefix', action='append', default=[],
                   help='Additional checkpoint folder prefix; appends _SKILL/best.pt')
    a = p.parse_args(); root = Path(a.root); out = Path(a.out)
    assert not out.exists(); out.mkdir(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'scope': __doc__, 'source_sha256': digest_file(__file__),
              'no_model_updates': True, 'no_native_queries': True, 'skills': {}}
    for role in ['keepout', 'waypoint']:
        datasets = {'training113': root / ('cube_input_grid_v113_' + role + '_data')}
        held = root / (a.held_prefix + '_' + role)
        if a.require_held:
            assert (held / 'DONE.json').exists()
        if (held / 'DONE.json').exists():
            datasets['held_from_adapter_training'] = held
        role_result = {}
        for split, folder in datasets.items():
            count = json.loads((folder / 'summary.json').read_text())['new_matched_rows_per_arm']
            with np.load(folder / 'grid/feedback.npz') as x:
                data = {k: x[k][:count].copy() for k in ['states', 'contexts', 'actions', 'base_z', 'new_z']}
            groups = {}
            for i in range(count):
                key = data['states'][i].tobytes() + data['contexts'][i].tobytes()
                groups.setdefault(key, []).append(i)
            split_result = {'rows': count, 'distinct_states': len(groups), 'models': {}}
            models = {'parent110': root / ('cube_radial_v110_radial_' + role) / 'best.pt'}
            models.update({family + '114': root / ('cube_input_grid_v114_' + family + '_' + role) / 'best.pt'
                           for family in ['grid', 'applied']})
            models.update({prefix: root / (prefix + '_' + role) / 'best.pt' for prefix in a.extra_prefix})
            for family, checkpoint in models.items():
                model, encoder, _ = load_adapter(checkpoint, root / 'cube_diverse_v35_stageB/base_policy.pt', 'cuda')
                model.eval().requires_grad_(False); before = model.frozen_digest()
                predictions = []
                with torch.no_grad():
                    for start in range(0, count, 256):
                        tensors = {k: torch.tensor(data[k][start:start + 256], device='cuda')
                                   for k in ['states', 'contexts', 'base_z', 'new_z']}
                        predictions.append(model(tensors['states'], tensors['base_z'], tensors['new_z'], tensors['contexts']).cpu().numpy())
                predictions = np.concatenate(predictions)
                assert np.isfinite(predictions).all() and before == model.frozen_digest()
                split_result['models'][family] = {'checkpoint_sha256': digest_file(checkpoint),
                    **metrics(predictions, data['actions'], list(groups.values()))}
                del model, encoder
            role_result[split] = split_result
        result['skills'][role] = role_result
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'offline_fit_audit_complete': True}) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
