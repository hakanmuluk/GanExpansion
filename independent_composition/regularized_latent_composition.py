"""Private-code prior/cycle penalties on the existing frozen latent composer.

The original generator interfaces, action agreement, base tether, candidate
search, Adam loop and output executor are unchanged. Only private encoders may
be used for a window-calibrated cycle. No future state or simulator is used.
"""
from dataclasses import dataclass
import torch
from .latent_action_composition import AmplitudeConfig, AmplitudeComposer
from .unchanged_composition import action_merge


def window_encode(encoder, state, context, base_z, actions):
    """Repeat the CURRENT measured observation, not imagined future states."""
    n, horizon = actions.shape[:2]
    lengths = torch.full((n,), horizon, dtype=torch.long, device=state.device)
    states = state[:, None].expand(-1, horizon, -1)
    contexts = context[:, None].expand(-1, horizon, -1)
    return encoder(states, actions, contexts, base_z, lengths)


def issued_controls(actions):
    if actions.shape[-1] == 5:
        # Match the deployed gripper; analog confidence must not encode z.
        grip = torch.where(actions[..., 4:] >= 0, 1., -1.)
        return torch.cat([actions[..., :4], grip], -1)
    return actions


@dataclass
class RegularizedConfig(AmplitudeConfig):
    private_prior_weight: float = 0.
    cycle_weight: float = 0.
    cycle_horizon: int = 8
    cycle_target: str = 'self'


class RegularizedComposer(AmplitudeComposer):
    def __init__(self, bank, config, seed=241):
        assert config.private_prior_weight >= 0 and config.cycle_weight >= 0
        assert 1 <= config.cycle_horizon <= 16
        assert config.cycle_target in ['self', 'executed']
        if config.cycle_weight:
            assert getattr(bank, 'cycle_interface', None) == 'static_observation_issued_chunk'
            assert getattr(bank, 'cycle_horizon', None) == config.cycle_horizon
        super().__init__(bank, config, seed)
        if config.cycle_weight:
            # The original composer forbids all encoders. Permit ONLY the two
            # existing private encoders for this explicitly enabled cycle.
            from .inference_guard import forbidden
            for encoder in [bank.ek, bank.ew]:
                for key, hook in list(encoder._forward_pre_hooks.items()):
                    if hook is forbidden:
                        del encoder._forward_pre_hooks[key]

    def objective(self, state, bz, contexts, z, history=None, previous=None):
        loss, k, w, details = super().objective(state, bz, contexts, z, history, previous)
        c = self.config
        # Standard-normal negative log-prior, omitting its constant.
        prior = .5 * z[:, :2].square().sum(1)
        cycle = torch.zeros_like(prior)
        if c.cycle_weight:
            bk = torch.stack([bz[:, 0].sign() * z[:, 2], bz[:, 1]], -1)
            bw = torch.stack([bz[:, 0].sign() * z[:, 3], bz[:, 1]], -1)
            if c.cycle_target == 'executed':
                executed = action_merge(k, w, self.bank.base(state, bz), c)
                chunks = (executed, executed)
            else:
                chunks = (k, w)
            for j, (encoder, base, chunk) in enumerate(zip([self.bank.ek, self.bank.ew], [bk, bw], chunks)):
                actions = issued_controls(chunk[:, :c.cycle_horizon])
                mu, _ = window_encode(encoder, state, contexts[j], base, actions)
                cycle = cycle + .5 * (mu - z[:, j:j+1]).abs().mean(1)
            self.bank.count('private_cycle_encoder', 2 * len(state), 2)
        return (loss + c.private_prior_weight * prior + c.cycle_weight * cycle,
                k, w, {**details, 'private_gaussian_nll': prior, 'private_cycle_mae': cycle})
