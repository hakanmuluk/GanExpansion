"""Own-keepout validity regularization, TRAINING ONLY.

No new neural component. A short-step approximation calibrated from own native
training episodes gives a differentiable penalty for inward controls near the
own forbidden region. It is never a planning objective or policy dependency.
Native single-skill evaluation remains the checkpoint selection criterion.
"""
import math
import torch
from .inference_guard import inference_active


def observed_step_penalty(actions, states, contexts, mask, coefficients, margin=.006):
    """Actual observed states plus one predicted control; all lengths in metres.

    Original state47 tail: cube velocity[-7:-4]/5 and measured effector
    velocity[-3:]/5. Original context20: own cube-relative offset[4:7]*.15,
    radius[3]*.1, holding[13], lifted[14], progress[10].
    """
    assert not inference_active(), 'Own validity penalty is training-only'
    assert actions.shape[-1] == 5 and states.shape[-1] == 47 and contexts.shape[-1] == 20
    coefficient = coefficients.to(actions)
    delta = (actions[..., :3]*coefficient[:, 0]
        + states[..., -7:-4]/5*.05*coefficient[:, 1]
        + states[..., -3:]/5*.05*coefficient[:, 2] + coefficient[:, 3])
    relative = contexts[..., 4:7]*.15
    radius = contexts[..., 3]*.1 + math.sqrt(3)*.02
    next_gap = (relative+delta).norm(dim=-1)-radius
    fraction = (-(relative*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-10)).clamp(0, 1)
    swept_gap = (relative+fraction[..., None]*delta).norm(dim=-1)-radius
    penalty = ((margin-next_gap).relu().square()+(-swept_gap).relu().square())/.02**2
    active = (contexts[..., 13] > .5) & (contexts[..., 14] > .5)
    active &= (contexts[..., 10] > .1) & (contexts[..., 10] < .9)
    weights = mask*active.to(mask.dtype)
    return (weights*penalty).sum()/weights.sum().clamp_min(1)


def unary_keepout_penalty(model, feedback, batch, known_prediction, coefficients, margin=.006):
    """Fresh prior codes at observed states plus known-code forecast heads.

    Forecast heads are checked against the REAL teacher future observations
    already in the original training episode. They are not imagined or queried
    during inference. The additional fresh-code forward uses the existing actor.
    """
    assert not inference_active(), 'Own validity penalty is training-only'
    assert not model.config.finetune_base
    f = feedback
    prior = torch.randn_like(f['new_z'])
    pred = model(f['states'], f['base_z'], prior, f['contexts'])[:, 0]
    current = observed_step_penalty(pred, f['states'], f['contexts'], f['mask'][:, 0],
                                    coefficients, margin)
    horizon = 8
    times = (batch['contexts'][..., 16]*240).round().long()[..., None]
    times = times+torch.arange(horizon, device=times.device)
    times = times.clamp_max(batch['episode_states'].shape[1]-1)
    rows = torch.arange(len(times), device=times.device)[:, None, None]
    real_states = batch['episode_states'][rows, times]
    real_contexts = batch['episode_contexts'][rows, times]
    heads = observed_step_penalty(known_prediction[..., :horizon, :], real_states, real_contexts,
                                  batch['mask'][..., :horizon], coefficients, margin)
    return .5*(current+heads)
