"""Invert existing private codes AND each actor's existing base-speed input.

The original base-route input stays fixed. Each original network still receives
exactly its original state, two base coordinates, one private code and context.
No model, extra input feature, simulator or geometric score is introduced.
"""
from dataclasses import dataclass
import math
import torch
from .unchanged_composition import CompositionConfig, OriginalComposer, action_merge


@dataclass
class SpeedConfig(CompositionConfig):
    seed_quantiles: tuple = (.01, .25, .5, .75, .99)
    speed_bound: float = 2.
    speed_tether: float = .0005


class SpeedComposer(OriginalComposer):
    trace_noise_name = 'initial_existing_private_and_speed_codes'

    def pair_existing_inputs(self, state, bz, z, contexts):
        bk = torch.stack([bz[:, 0], z[:, 2]], -1)
        bw = torch.stack([bz[:, 0], z[:, 3]], -1)
        self.bank.count('independent', 2 * len(state), 2)
        return (self.bank.k(state, bk, z[:, :1], contexts[0], gate=getattr(self.bank, 'keepout_gate', 1)),
                self.bank.w(state, bw, z[:, 1:2], contexts[1], gate=getattr(self.bank, 'waypoint_gate', 1)))

    def objective(self, state, bz, contexts, z, history=None, previous=None):
        c = self.config
        k, w = self.pair_existing_inputs(state, bz, z, contexts)
        raw = (self.coordinates(k[:, :c.agreement_horizon]) -
               self.coordinates(w[:, :c.agreement_horizon])).square().mean((1, 2))
        tether = (z[:, 2:] - bz[:, 1:2]).square().mean(1)
        return raw + c.speed_tether * tether, k, w, {'agreement': raw, 'base_speed_tether': tether}

    def sample(self, state, contexts, base_z):
        c = self.config
        assert c.optimize and c.optimize_waypoint and not c.preserve_mode
        assert c.history == c.prior == c.continuity == c.trust == 0
        assert c.merge in ['keepout', 'waypoint', 'blend']
        assert c.speed_bound > 1 and c.speed_tether >= 0
        n = len(state)
        self.bank.reset_counts()
        q = state.new_tensor(c.seed_quantiles)
        assert ((q > 0) & (q < 1)).all()
        v = math.sqrt(2) * torch.erfinv(2 * q - 1)
        values = v[v.abs() > 1e-6]
        if (v.abs() <= 1e-6).any():
            values = torch.cat([values, state.new_tensor([-c.minimum_magnitude, c.minimum_magnitude])])
        values = values.sign() * values.abs().clamp(c.minimum_magnitude, c.maximum_magnitude)
        private = torch.cartesian_prod(values, values)
        if c.keepout_sign:
            private = private[private[:, 0].sign() == c.keepout_sign]
        if c.waypoint_sign:
            private = private[private[:, 1].sign() == c.waypoint_sign]
        speeds = torch.cartesian_prod(state.new_tensor([-1., 0., 1.]), state.new_tensor([-1., 0., 1.]))
        grid = torch.cat([private.repeat_interleave(len(speeds), 0), speeds.repeat(len(private), 1)], -1)
        candidates = grid[None].expand(n, -1, -1).clone()
        original_speed = base_z[:, 1:2].clamp(-c.speed_bound, c.speed_bound)
        unchanged = torch.cat([private[None].expand(n, -1, -1),
            original_speed[:, None].expand(-1, len(private), 2)], -1)
        candidates = torch.cat([candidates, unchanged], 1)
        if self.previous is not None:
            candidates = torch.cat([candidates, self.previous[:, None]], 1)

        def expanded(count):
            return (state.repeat_interleave(count, 0), base_z.repeat_interleave(count, 0),
                    tuple(x.repeat_interleave(count, 0) for x in contexts))

        count = candidates.shape[1]
        s, bz, ctx = expanded(count)
        with torch.no_grad():
            costs = self.objective(s, bz, ctx, candidates.reshape(-1, 4))[0].reshape(n, count)
            starts = min(c.starts, count)
            choice = costs.topk(starts, largest=False, dim=1).indices
            initial = candidates.gather(1, choice[:, :, None].expand(-1, -1, 4))
        self.noise = initial.detach().clone()
        signs = initial[:, :, :2].sign().reshape(-1, 2)
        fraction = ((initial[:, :, :2].abs() - c.minimum_magnitude) /
                    (c.maximum_magnitude - c.minimum_magnitude)).clamp(.001, .999)
        private_parameter = torch.logit(fraction)
        speed_parameter = torch.atanh((initial[:, :, 2:] / c.speed_bound).clamp(-.999, .999))
        parameter = torch.cat([private_parameter, speed_parameter], -1).reshape(-1, 4).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([parameter], lr=c.lr)
        s, bz, ctx = expanded(starts)
        best = state.new_full((n, starts), float('inf'))
        best_z = initial.clone()
        for step in range(c.steps + 1):
            with torch.set_grad_enabled(step < c.steps):
                private_z = signs * (c.minimum_magnitude + (c.maximum_magnitude - c.minimum_magnitude)
                                     * parameter[:, :2].sigmoid())
                speed_z = c.speed_bound * parameter[:, 2:].tanh()
                z = torch.cat([private_z, speed_z], -1)
                cost = self.objective(s, bz, ctx, z)[0].reshape(n, starts)
            with torch.no_grad():
                better = cost < best
                best = torch.minimum(best, cost)
                best_z = torch.where(better[:, :, None], z.detach().reshape(n, starts, 4), best_z)
            if step == c.steps:
                break
            gradient, = torch.autograd.grad(cost.sum(), parameter)
            assert torch.isfinite(gradient).all()
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = gradient
            optimizer.step()
        with torch.no_grad():
            choice = best.argmin(1)
            z = best_z[torch.arange(n, device=state.device), choice]
            loss, k, w, details = self.objective(state, base_z, contexts, z)
            result = action_merge(k, w, self.bank.base(state, base_z), c)
        self.previous = z.detach().clone()
        self.last_plan = {'optimizer_iterations': c.steps, 'starts': starts,
            'coarse_network_candidates': count, 'simulator_queries': 0,
            'agreement_horizon': c.agreement_horizon, 'selected_existing_inputs': z.tolist(),
            'base_route_input_unchanged': True, 'base_speed_inputs_optimized': True,
            'original_private_dimensions_per_actor': 1, 'original_base_dimensions_per_actor': 2,
            'objective': loss.tolist(), **{key: value.tolist() for key, value in details.items()}}
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result.detach()
