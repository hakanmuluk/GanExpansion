"""Network-only consensus on the exact conditional training input grid.

Evaluate 18 original-generator proposals per skill, compare their action
prefixes, and execute one original proposal. No continuous latent interpolation,
action averaging, learned judge, encoder, geometry or simulator is used. Grid
membership does not certify the neural proposal's physical validity.
"""
from dataclasses import dataclass
from statistics import NormalDist
import torch
from .inference_guard import forbidden


@dataclass
class CodebookConfig:
    private_quantiles: tuple = (.02, .25, .49, .51, .75, .98)
    amplitude_half_quantiles: tuple = (.1, .5, .9)
    agreement_horizon: int = 8
    executor: str = 'keepout'


class CodebookComposer:
    trace_noise_name = 'fixed_conditional_training_input_grid'

    def __init__(self, bank, config, seed=241):
        self.bank, self.config = bank, config
        assert 1 <= config.agreement_horizon <= 16
        assert config.executor in ['keepout', 'waypoint']
        assert all(0 < p < 1 for p in tuple(config.private_quantiles) + tuple(config.amplitude_half_quantiles))
        assert all(float(p) != .5 for p in config.private_quantiles), 'Use the two nonzero sign charts'
        self.values = [(NormalDist().inv_cdf(float(q)), NormalDist().inv_cdf((1 + float(p)) / 2))
            for q in config.private_quantiles for p in config.amplitude_half_quantiles]
        self.noise = None; self.last_plan = {}
        bank.eval().requires_grad_(False)
        for module in [bank.ek, bank.ew, bank.k.original_encoder, bank.w.original_encoder]:
            module.register_forward_pre_hook(forbidden)

    @torch.no_grad()
    def sample(self, state, contexts, base_z):
        assert base_z.shape[-1] == 2 and (base_z[:, 0] != 0).all()
        n = len(state); grid = state.new_tensor(self.values); m = len(grid)
        self.noise = grid[None].expand(n, -1, -1).clone()
        self.bank.reset_counts()
        s = state.repeat_interleave(m, 0)
        b = base_z.repeat_interleave(m, 0).clone()
        b[:, 0] = b[:, 0].sign() * grid[:, 1].repeat(n)
        z = grid[:, :1].repeat(n, 1)
        self.bank.count('independent', 2 * n * m, 2)
        k = self.bank.k(s, b, z, contexts[0].repeat_interleave(m, 0), gate=getattr(self.bank, 'keepout_gate', 1))
        w = self.bank.w(s, b, z, contexts[1].repeat_interleave(m, 0), gate=getattr(self.bank, 'waypoint_gate', 1))
        k = k.reshape(n, m, *k.shape[1:]); w = w.reshape(n, m, *w.shape[1:])
        assert k.shape[-1] == 5 and torch.isfinite(k).all() and torch.isfinite(w).all()
        h = self.config.agreement_horizon
        cost = (k[:, :, None, :h, :3] - w[:, None, :, :h, :3]).square().mean((-1, -2))
        selected = cost.flatten(1).argmin(1)
        ik, iw = selected // m, selected % m
        batch = torch.arange(n, device=state.device)
        result = k[batch, ik] if self.config.executor == 'keepout' else w[batch, iw]
        selected_inputs = torch.stack([grid[ik, 0], grid[iw, 0], grid[ik, 1], grid[iw, 1]], -1)
        self.last_plan = {'optimizer_iterations': 0, 'discrete_code_pairs_scored': m * m,
            'network_proposals_per_role': m, 'agreement_horizon': h,
            'simulator_queries': 0, 'encoder_queries': 0,
            'selection': 'Global minimum of network-only action-prefix MSE on fixed training grid',
            'continuous_latent_refinement': False, 'action_mixing': False,
            'grid_membership_is_not_safety_certificate': True,
            'executor': self.config.executor, 'selected_existing_inputs': selected_inputs.tolist(),
            'agreement': cost[batch, ik, iw].tolist(),
            'base_route_sign_and_speed_preserved': True}
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result
