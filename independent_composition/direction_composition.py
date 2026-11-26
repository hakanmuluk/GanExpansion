"""Code-only agreement tolerant of policy action-scale differences.

All network forwards, latent search, history, priors and execution rules are
the existing originals. Smooth normalization changes only the agreement term;
near-zero actions remain distinct from moving actions. No geometry or simulator.
"""
from dataclasses import dataclass
import torch
from .unchanged_composition import CompositionConfig, OriginalComposer


@dataclass
class DirectionConfig(CompositionConfig):
    direction_epsilon: float = .05
    speed_weight: float = .05


def agreement(a, b, epsilon):
    a = a / torch.sqrt(a.square().sum(-1, keepdim=True) + epsilon ** 2)
    b = b / torch.sqrt(b.square().sum(-1, keepdim=True) + epsilon ** 2)
    return (a - b).square().mean((1, 2))


class DirectionComposer(OriginalComposer):
    def objective(self, state, bz, contexts, z, history=None, previous=None):
        _, k, w, details = super().objective(state, bz, contexts, z, history, previous)
        c = self.config
        assert c.direction_epsilon > 0 and c.speed_weight >= 0
        raw = details['agreement']
        direction = agreement(self.coordinates(k[:, :c.agreement_horizon]),
                              self.coordinates(w[:, :c.agreement_horizon]), c.direction_epsilon)
        loss = direction + c.speed_weight * raw + c.prior * details['latent_prior']
        loss = loss + c.trust * details['prior_trust'] + c.continuity * details['continuity']
        loss = loss + c.history * details['past_action_reconstruction']
        return loss, k, w, {**details, 'agreement': direction,
                           'raw_action_agreement': raw}
