"""Own-teacher magnitude-effect supervision for a scalar Gaussian skill code.

Training only. Compare the even component at two code magnitudes, using teacher
queries at +a, -a, +b, -b with identical observed state and own instruction.
This separates a magnitude-dependent effect from the easier sign-dependent
effect, without choosing physical action axes or adding a neural component.
The averaged sequences are loss terms, never executed actions.
"""
import numpy as np
import torch

from .inference_guard import inference_active


def build_quartets(feedback):
    assert not inference_active(), 'Magnitude supervision is training-only'
    assert feedback['new_z'].shape[-1] == 1
    keys = ['states', 'contexts', 'base_z', 'new_z']
    assert all(feedback[k].dtype == torch.float32 for k in keys)
    arrays = {k: feedback[k].detach().cpu().contiguous().numpy() for k in keys}
    groups = {}
    for i in range(len(arrays['states'])):
        key = b''.join(arrays[k][i].tobytes() for k in keys[:-1])
        groups.setdefault(key, []).append(i)
    quartets = []
    for ids in groups.values():
        by_code = {float(arrays['new_z'][i, 0]): i for i in ids}
        magnitudes = sorted({abs(u) for u in by_code if u and -u in by_code})
        if len(magnitudes) != 2:
            continue
        a, b = magnitudes
        quartets.append([by_code[a], by_code[-a], by_code[b], by_code[-b]])
    assert quartets, 'Requires symmetric four-code teacher queries at identical inputs'
    indices = torch.as_tensor(np.array(quartets, np.int64), device=feedback['states'].device)
    return indices, {'quartets': len(quartets), 'order': ['+small', '-small', '+large', '-large'],
                     'fixed_inputs': 'exact same state, own context, base code',
                     'joint_data_or_physical_axes': False}


def even_difference(values):
    return .5 * (values[:, 0] + values[:, 1] - values[:, 2] - values[:, 3])


def magnitude_difference_loss(model, feedback, quartets, generator, batch=32, horizon=8):
    assert not inference_active(), 'Magnitude supervision is training-only'
    assert batch > 0 and 1 <= horizon <= feedback['actions'].shape[-2]
    selected = torch.randint(len(quartets), (batch,), generator=generator, device=quartets.device)
    ids = quartets[selected]
    flat = ids.flatten()
    predicted = model(feedback['states'][flat], feedback['base_z'][flat],
                      feedback['new_z'][flat], feedback['contexts'][flat])
    predicted = predicted[:, :horizon].reshape(batch, 4, horizon, -1)
    target = feedback['actions'][ids, :horizon]
    mask = feedback['mask'][ids, :horizon].amin(dim=1)[..., None]
    delta, teacher_delta = even_difference(predicted), even_difference(target)
    denominator = (mask.sum() * predicted.shape[-1]).clamp_min(1)
    loss = ((delta - teacher_delta).square() * mask).sum() / denominator
    metrics = {'magnitude_difference': loss.detach(),
        'magnitude_predicted_delta_rms': ((delta.square() * mask).sum() / denominator).sqrt().detach(),
        'magnitude_teacher_delta_rms': ((teacher_delta.square() * mask).sum() / denominator).sqrt().detach()}
    return loss, metrics
