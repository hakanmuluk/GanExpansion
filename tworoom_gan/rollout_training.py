"""Differentiable physical rollouts and expert feedback relabeling.

These optional diagnostics-driven objectives address the gap between the
proposal's teacher-state latent cycle and actual 8-step replanned execution.
They do not add new-skill adapters or replace the generative policy with an expert.
"""

import torch
from .env import TASK, EXECUTE, observation, transition, expert_chunk


def differentiable_rollout(generator, starts, goals, z, max_steps=TASK.max_steps):
    """Use the exact same physics/replanning schedule as evaluation, with grads."""
    pos = starts
    alive = torch.ones(len(starts), device=starts.device, dtype=torch.bool)
    lengths = torch.full((len(starts),), max_steps, device=starts.device, dtype=torch.long)
    states, actions = [], []
    for t in range(max_steps):
        state = observation(pos, goals)
        states.append(state)
        if t % EXECUTE == 0:
            chunk = generator(state, z)
        a = torch.where(alive[:, None], chunk[:, t % EXECUTE], 0)
        actions.append(a)
        pos, _ = transition(pos, a)
        success = (pos - goals).norm(dim=-1) < TASK.success_radius
        lengths = torch.where(alive & success, t + 1, lengths)
        alive = alive & ~success
    return torch.stack(states, 1), torch.stack(actions, 1), lengths


@torch.no_grad()
def collect_feedback_states(generator, encoder, batch, batch_size=32):
    """Roll out encoded demonstration styles and relabel encountered states.

    The route/speed/offset are those of the source demonstration. Only the
    training feedback target uses the expert; inference uses G alone.
    """
    count = min(batch_size, len(batch["lengths"]))
    states = batch["episode_states"][:count]
    mu, _ = encoder(states, batch["episode_actions"][:count], batch["lengths"][:count])
    starts, goals = (states[:, 0, :2] + 1) * 112, (states[:, 0, 2:4] + 1) * 112
    st, ac, lengths = differentiable_rollout(generator, starts, goals, mu)
    sample_t = torch.arange(0, TASK.max_steps, EXECUTE, device=starts.device)
    chosen = st[:, sample_t]
    valid = sample_t[None] < lengths[:, None]
    pos, goal = (chosen[..., :2] + 1) * 112, (chosen[..., 2:4] + 1) * 112
    mode = batch["mode"][:count, None].expand(-1, len(sample_t))
    speed = batch["speed"][:count, None].expand_as(mode)
    offset = batch["offset"][:count, None].expand_as(mode)
    targets = expert_chunk(pos, goal, mode, speed, offset=offset)
    codes = mu[:, None].expand(-1, len(sample_t), -1)
    return chosen[valid], codes[valid], targets[valid]
