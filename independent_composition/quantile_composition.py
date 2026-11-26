"""Original code-only inversion with prior-quantile coarse initialization.

A small Cartesian grid of Gaussian quantiles is scored by the same network-only
agreement objective. The best four starts receive the original Adam refinement.
No native rollout, region predicate, inverse encoder or learned judge is used.
"""
from dataclasses import dataclass
import math
import torch
from .unchanged_composition import CompositionConfig, OriginalComposer, action_merge


@dataclass
class QuantileConfig(CompositionConfig):
    seed_quantiles: tuple = (.01, .25, .5, .75, .99)


class QuantileComposer(OriginalComposer):
    trace_noise_name = 'initial_quantile_latents'

    def sample(self, state, contexts, base_z):
        c = self.config
        if not c.optimize:
            return super().sample(state, contexts, base_z)
        assert c.optimize_waypoint and not c.preserve_mode
        assert c.merge in ['blend', 'residual', 'keepout', 'waypoint', 'base', 'max_delta']
        n = len(state)
        self.bank.reset_counts()
        history = None
        if self.records and c.history:
            history = (torch.stack([r[0] for r in self.records], 1),
                       torch.stack([r[1] for r in self.records], 1),
                       tuple(torch.stack([r[2][j] for r in self.records], 1) for j in [0, 1]),
                       torch.stack([r[3] for r in self.records], 1))

        def expanded(count):
            s = state.repeat_interleave(count, 0)
            b = base_z.repeat_interleave(count, 0)
            ctx = tuple(x.repeat_interleave(count, 0) for x in contexts)
            past = None if history is None else (history[0].repeat_interleave(count, 0),
                history[1].repeat_interleave(count, 0),
                tuple(x.repeat_interleave(count, 0) for x in history[2]),
                history[3].repeat_interleave(count, 0))
            previous = None if self.previous is None else self.previous.repeat_interleave(count, 0)
            return s, b, ctx, past, previous

        quantiles = state.new_tensor(c.seed_quantiles)
        assert ((quantiles > 0) & (quantiles < 1)).all()
        values = math.sqrt(2) * torch.erfinv(2 * quantiles - 1)
        # The original generator has a sign feature. Represent the mean from
        # both sign charts instead of introducing an untrained exact-zero sign.
        nonzero = values[values.abs() > 1e-6]
        if (values.abs() <= 1e-6).any():
            nonzero = torch.cat([nonzero, state.new_tensor([-c.minimum_magnitude, c.minimum_magnitude])])
        values = nonzero.sign() * nonzero.abs().clamp(c.minimum_magnitude, c.maximum_magnitude)
        seeds = torch.cartesian_prod(values, values)
        if c.keepout_sign:
            seeds = seeds[seeds[:, 0].sign() == c.keepout_sign]
        if c.waypoint_sign:
            seeds = seeds[seeds[:, 1].sign() == c.waypoint_sign]
        candidates = seeds[None].expand(n, -1, -1).clone()
        if self.previous is not None:
            candidates = torch.cat([candidates, self.previous[:, None]], 1)
        coarse_count = candidates.shape[1]
        s, b, ctx, past, previous = expanded(coarse_count)
        with torch.no_grad():
            value = self.objective(s, b, ctx, candidates.reshape(-1, 2), past, previous)[0].reshape(n, coarse_count)
            starts = min(c.starts, coarse_count)
            chosen = value.topk(starts, largest=False, dim=1).indices
            initial = candidates.gather(1, chosen[:, :, None].expand(-1, -1, 2))
        self.noise = initial.detach().clone()
        signs = initial.sign().reshape(-1, 2)
        fraction = ((initial.abs() - c.minimum_magnitude) /
                    (c.maximum_magnitude - c.minimum_magnitude)).clamp(.001, .999)
        parameter = torch.logit(fraction).reshape(-1, 2).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([parameter], lr=c.lr)
        s, b, ctx, past, previous = expanded(starts)
        best = state.new_full((n, starts), float('inf'))
        best_z = initial.clone()
        for step in range(c.steps + 1):
            with torch.set_grad_enabled(step < c.steps):
                z = signs * (c.minimum_magnitude + (c.maximum_magnitude - c.minimum_magnitude) * parameter.sigmoid())
                value = self.objective(s, b, ctx, z, past, previous)[0].reshape(n, starts)
            with torch.no_grad():
                improved = value < best
                best = torch.minimum(best, value)
                best_z = torch.where(improved[:, :, None], z.detach().reshape(n, starts, 2), best_z)
            if step == c.steps:
                break
            gradient, = torch.autograd.grad(value.sum(), parameter)
            assert torch.isfinite(gradient).all()
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = gradient
            optimizer.step()
        with torch.no_grad():
            chosen = best.argmin(1)
            z = best_z[torch.arange(n, device=state.device), chosen]
            loss, k, w, details = self.objective(state, base_z, contexts, z, history, self.previous)
            base = self.bank.base(state, base_z)
            result = action_merge(k, w, base, c)
        self.last_plan = {'optimizer_iterations': c.steps, 'starts': starts,
            'coarse_network_candidates': coarse_count, 'simulator_queries': 0,
            'agreement_horizon': c.agreement_horizon, 'history_observations': len(self.records),
            'selected_codes': z.tolist(), 'objective': loss.tolist(),
            **{key: value.tolist() for key, value in details.items()}}
        self.previous = z.detach().clone()
        self.pending = (state.detach().clone(), base_z.detach().clone(),
                        tuple(x.detach().clone() for x in contexts))
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result.detach()
