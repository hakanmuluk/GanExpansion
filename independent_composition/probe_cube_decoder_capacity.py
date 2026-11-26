"""Diagnostic only: fit one real conditional grid with the original adapter.

This is a small architecture/optimization probe, not a task-performance model.
Only a temporary instance's original adapter parameters train. Base/old encoder
remain frozen; no weights are saved or adopted. Targets are native first XYZ
actions from one actually recorded same-state 18-condition group.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cube_addons.evaluate import load_adapter
from cube_addons.models import state_digest
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=1000); p.add_argument('--lr', type=float, default=.001)
    a = p.parse_args(); root = Path(a.root); out = Path(a.out)
    assert not out.exists(); out.mkdir(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'scope': __doc__, 'source_sha256': digest_file(__file__), 'no_weights_saved': True,
        'no_native_task_success_claim': True, 'roles': {}}
    for skill in ['keepout', 'waypoint']:
        folder = root / ('cube_input_grid_v113_' + skill + '_data')
        count = json.loads((folder / 'summary.json').read_text())['new_matched_rows_per_arm']
        with np.load(folder / 'grid/feedback.npz') as x:
            values = {k: x[k][:count].copy() for k in ['states', 'contexts', 'actions', 'base_z', 'new_z']}
        groups = {}
        for i in range(count):
            key = values['states'][i].tobytes() + values['contexts'][i].tobytes()
            groups.setdefault(key, []).append(i)
        selected = None
        for ids in groups.values():
            if len(ids) != 18:
                continue
            amp_effect = []
            for z in np.unique(values['new_z'][ids]):
                own = [i for i in ids if values['new_z'][i, 0] == z]
                amp_effect.append(np.std(values['actions'][own, 0, :3], axis=0).mean())
            if np.mean(amp_effect) > .005:
                selected = ids
                break
        assert selected is not None, 'No complete amplitude-sensitive training group'
        tensors = {k: torch.tensor(v[selected], dtype=torch.float32, device='cuda') for k, v in values.items()}
        checkpoint = root / ('cube_input_grid_v114_grid_' + skill) / 'best.pt'
        original_file_hash = digest_file(checkpoint)
        model, encoder, _ = load_adapter(checkpoint, root / 'cube_diverse_v35_stageB/base_policy.pt', 'cuda')
        assert not model.config.finetune_base
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not name.startswith(('base.', 'original_encoder.')))
        model.train(); frozen = state_digest(model.base, model.original_encoder)
        optimizer = torch.optim.Adam(model.trainable(), lr=a.lr)
        trace = []
        target = tensors['actions'][:, 0, :3]
        for step in range(a.steps + 1):
            pred = model(tensors['states'], tensors['base_z'], tensors['new_z'], tensors['contexts'])[:, 0, :3]
            loss = (pred - target).square().mean()
            assert torch.isfinite(loss)
            if step == 0 or step % 100 == 0:
                trace.append({'step': step, 'first_action_xyz_mse': float(loss.detach())})
            if step == a.steps:
                break
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable(), 5.)
            optimizer.step()
        assert frozen == state_digest(model.base, model.original_encoder)
        assert digest_file(checkpoint) == original_file_hash
        result['roles'][skill] = {'conditions': 18, 'fixed_observed_states': 1,
            'checkpoint_sha256': original_file_hash, 'trace': trace,
            'teacher_first_action_std_xyz': target.std(0, unbiased=False).detach().cpu().tolist(),
            'fitted_first_action_std_xyz': pred.std(0, unbiased=False).detach().cpu().tolist(),
            'original_frozen_base_encoder_unchanged': True,
            'temporary_original_adapter_instance_only': True}
        del model, encoder, optimizer, tensors
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'capacity_probe_complete': True, 'weights_not_adopted': True}) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
