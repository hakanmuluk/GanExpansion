"""Own-task paired action differences, for TRAINING ONLY.

Match teacher action differences between codes at exactly identical observed
state, own instruction, and base code. Unlike unanchored diversity rewards,
this does not reward arbitrary separation or alter the mean action target.
The existing generator is the only neural component used here.
"""
from itertools import combinations

import numpy as np
import torch

from .inference_guard import inference_active


def build_pairs(feedback):
    """Exact float32 input equality; no grouping by approximate geometry."""
    assert not inference_active(), 'Paired teacher loss is training-only'
    keys = ['states', 'contexts', 'base_z', 'new_z']
    assert all(feedback[k].dtype == torch.float32 for k in keys)
    values = {k: feedback[k].detach().cpu().contiguous().numpy() for k in keys}
    groups = {}
    for i in range(len(values['states'])):
        key = b''.join(values[k][i].tobytes() for k in keys[:-1])
        groups.setdefault(key, []).append(i)
    pairs, group_count = [], 0
    for indices in groups.values():
        # Identical duplicate labels need no extra supervision. Cap duplicate
        # copies per code, while retaining every distinct counterfactual code.
        unique = {}
        for i in indices:
            unique.setdefault(values['new_z'][i].tobytes(), i)
        if len(unique) < 2:
            continue
        group_count += 1
        pairs.extend(combinations(unique.values(), 2))
    assert pairs, 'Requires different-code teacher examples at identical inputs'
    indices = torch.as_tensor(np.asarray(pairs, np.int64), device=feedback['states'].device)
    return indices, {'same_input_groups': group_count, 'distinct_code_pairs': len(pairs),
                     'input_comparison': 'exact float32 bytes for state, own context, base code',
                     'joint_data_or_task_geometry': False}


def action_difference_loss(model, feedback, pairs, generator, batch=64, horizon=8):
    assert not inference_active(), 'Paired teacher loss is training-only'
    assert batch > 0 and 1 <= horizon <= feedback['actions'].shape[-2]
    selected = torch.randint(len(pairs), (batch,), generator=generator, device=pairs.device)
    ids = pairs[selected]
    flat = ids.flatten()
    f = {k: feedback[k][flat] for k in ['states', 'contexts', 'base_z', 'new_z']}
    predicted = model(f['states'], f['base_z'], f['new_z'], f['contexts'])
    predicted = predicted[:, :horizon].reshape(batch, 2, horizon, -1)
    target = feedback['actions'][ids, :horizon]
    mask = feedback['mask'][ids, :horizon].amin(dim=1)
    difference = predicted[:, 0] - predicted[:, 1]
    target_difference = target[:, 0] - target[:, 1]
    denominator = (mask.sum() * predicted.shape[-1]).clamp_min(1)
    loss = ((difference - target_difference).square() * mask[..., None]).sum() / denominator
    metrics = {'counterfactual': loss.detach(),
        'counterfactual_predicted_delta_rms': (
            (difference.square() * mask[..., None]).sum() / denominator).sqrt().detach(),
        'counterfactual_teacher_delta_rms': (
            (target_difference.square() * mask[..., None]).sum() / denominator).sqrt().detach()}
    return loss, metrics
