"""Experimental local action-set consensus using frozen original generators.

Finite, in-prior private codes define two local action sets. A convex QP finds
nearby mixtures of their first actions. This is a relaxation of point-code
agreement, NOT a safety certificate: convex action mixtures need not preserve
a skill. Execute one control and re-observe. No simulator, geometry, encoder,
or additional learned component participates in this optimizer.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .inference_guard import forbidden


def simplex_projection(value):
    value = np.asarray(value, dtype=np.float64)
    ordered = np.sort(value)[::-1]
    excess = np.cumsum(ordered) - 1.
    active = ordered - excess / np.arange(1, len(value) + 1) > 0
    index = np.flatnonzero(active)[-1]
    return np.maximum(value - excess[index] / (index + 1), 0.)


def consensus_weights(k, w, regularization=1e-3, iterations=100):
    """Product-simplex projected FISTA; inputs are action coordinates only."""
    k, w = np.asarray(k, np.float64), np.asarray(w, np.float64)
    assert k.ndim == w.ndim == 2 and k.shape[1] == w.shape[1]
    assert np.isfinite(k).all() and np.isfinite(w).all()
    assert regularization > 0 and iterations > 0
    nk, nw = len(k), len(w)
    prior = np.r_[np.full(nk, 1. / nk), np.full(nw, 1. / nw)]
    matrix = np.concatenate([k.T, -w.T], axis=1)
    hessian = matrix.T @ matrix / k.shape[1] + regularization * np.eye(nk + nw)
    lipschitz = float(np.linalg.eigvalsh(hessian)[-1])
    value = prior.copy(); accelerated = value.copy(); momentum = 1.
    best = value.copy(); best_objective = float('inf')
    for _ in range(iterations):
        step = accelerated - (hessian @ accelerated - regularization * prior) / lipschitz
        updated = np.r_[simplex_projection(step[:nk]), simplex_projection(step[nk:])]
        difference = matrix @ updated
        objective = float(np.mean(difference ** 2) + regularization * np.sum((updated - prior) ** 2))
        if objective < best_objective:
            best_objective = objective; best = updated.copy()
        next_momentum = .5 * (1. + np.sqrt(1. + 4. * momentum ** 2))
        accelerated = updated + (momentum - 1.) / next_momentum * (updated - value)
        value, momentum = updated, next_momentum
    assert np.min(best) >= 0 and abs(best[:nk].sum() - 1.) < 1e-10
    assert abs(best[nk:].sum() - 1.) < 1e-10
    return best[:nk], best[nk:], {
        'objective': best_objective,
        'agreement': float(np.mean((best[:nk] @ k - best[nk:] @ w) ** 2)),
        'prior_agreement': float(np.mean((k.mean(0) - w.mean(0)) ** 2)),
    }


@dataclass
class MixtureConfig:
    candidates: int = 8
    iterations: int = 100
    regularization: float = 1e-3
    executor: str = 'keepout'


class LatentMixtureComposer:
    trace_noise_name = 'fixed_in_prior_private_code_candidates'

    def __init__(self, bank, config, seed=241):
        self.bank, self.config = bank, config
        assert config.candidates >= 2 and config.candidates % 2 == 0
        assert config.executor in ['keepout', 'waypoint']
        self.noise = None; self.last_plan = {}
        bank.eval().requires_grad_(False)
        for module in [bank.ek, bank.ew, bank.k.original_encoder, bank.w.original_encoder]:
            module.register_forward_pre_hook(forbidden)

    @torch.no_grad()
    def sample(self, state, contexts, base_z):
        c = self.config; n = len(state); m = c.candidates
        self.bank.reset_counts()
        probabilities = (torch.arange(m, device=state.device, dtype=state.dtype) + .5) / m
        values = np.sqrt(2.) * torch.erfinv(2 * probabilities - 1)
        codes = torch.stack([values, values], -1).repeat(n, 1)
        self.noise = codes.reshape(n, m, 2).clone()
        k, w = self.bank.pair(state.repeat_interleave(m, 0), base_z.repeat_interleave(m, 0),
            codes, tuple(x.repeat_interleave(m, 0) for x in contexts))
        k = k.reshape(n, m, *k.shape[1:]); w = w.reshape(n, m, *w.shape[1:])
        assert k.shape[-1] == 5
        k_first = k[:, :, 0, :3].cpu().numpy(); w_first = w[:, :, 0, :3].cpu().numpy()
        weights = [consensus_weights(a, b, c.regularization, c.iterations) for a, b in zip(k_first, w_first)]
        chosen = 0 if c.executor == 'keepout' else 1
        coefficients = state.new_tensor(np.stack([row[chosen] for row in weights]))
        result = torch.einsum('nm,nmha->nha', coefficients, k if chosen == 0 else w)
        self.last_plan = {
            'optimizer_iterations': c.iterations, 'network_candidates_per_role': m,
            'agreement_horizon': 1, 'required_execution_horizon': 1,
            'simulator_queries': 0, 'convex_mixture_safety_guarantee': False,
            'executor': c.executor,
            'keepout_weights': [row[0].tolist() for row in weights],
            'waypoint_weights': [row[1].tolist() for row in weights],
            **{key: [row[2][key] for row in weights] for key in weights[0][2]},
        }
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result
