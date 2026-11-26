"""One semantic style code, two ORIGINAL independent skill adapters.

Applicable to the angular-label datasets: both independent teachers define the
same Gaussian code as the same direction in the goal/base-side frame. This is
an input constraint, not shared network weights. Only one scalar is optimized;
all neural classes, per-actor dimensions and action forwards stay unchanged.
It is not a valid assumption for arbitrarily unaligned private latent spaces.
"""
import math
import torch
from .unchanged_composition import OriginalComposer, action_merge


class SharedCodeComposer(OriginalComposer):
    trace_noise_name = 'initial_common_semantic_codes'

    def objective_shared(self, state, bz, contexts, code):
        return super().objective(state, bz, contexts, code.expand(-1, 2))

    def sample(self, state, contexts, base_z):
        c = self.config
        assert c.optimize and c.optimize_waypoint and not c.preserve_mode
        assert c.history == c.prior == c.trust == c.continuity == 0
        assert c.keepout_sign == c.waypoint_sign == 0
        n = len(state)
        self.bank.reset_counts()
        q = state.new_tensor(c.seed_quantiles)
        values = math.sqrt(2)*torch.erfinv(2*q-1)
        nonzero = values[values.abs() > 1e-6]
        if (values.abs() <= 1e-6).any():
            nonzero = torch.cat([nonzero, state.new_tensor([-c.minimum_magnitude, c.minimum_magnitude])])
        values = nonzero.sign()*nonzero.abs().clamp(c.minimum_magnitude, c.maximum_magnitude)
        candidates = values[None, :, None].expand(n, -1, -1).clone()
        if self.previous is not None:
            candidates = torch.cat([candidates, self.previous[:, None]], 1)

        def expanded(count):
            return (state.repeat_interleave(count, 0), base_z.repeat_interleave(count, 0),
                    tuple(x.repeat_interleave(count, 0) for x in contexts))

        count = candidates.shape[1]
        s, bz, ctx = expanded(count)
        with torch.no_grad():
            losses = self.objective_shared(s, bz, ctx, candidates.reshape(-1, 1))[0].reshape(n, count)
            starts = min(c.starts, count)
            indices = losses.topk(starts, largest=False, dim=1).indices
            initial = candidates.gather(1, indices[:, :, None])
        self.noise = initial.detach().clone()
        signs = initial.sign().reshape(-1, 1)
        fraction = ((initial.abs()-c.minimum_magnitude)/(c.maximum_magnitude-c.minimum_magnitude)).clamp(.001, .999)
        parameter = torch.logit(fraction).reshape(-1, 1).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([parameter], lr=c.lr)
        s, bz, ctx = expanded(starts)
        best = state.new_full((n, starts), float('inf'))
        best_code = initial.clone()
        for step in range(c.steps+1):
            with torch.set_grad_enabled(step < c.steps):
                code = signs*(c.minimum_magnitude+(c.maximum_magnitude-c.minimum_magnitude)*parameter.sigmoid())
                loss = self.objective_shared(s, bz, ctx, code)[0].reshape(n, starts)
            with torch.no_grad():
                better = loss < best
                best = torch.minimum(best, loss)
                best_code = torch.where(better[:, :, None], code.detach().reshape(n, starts, 1), best_code)
            if step == c.steps:
                break
            gradient, = torch.autograd.grad(loss.sum(), parameter)
            assert torch.isfinite(gradient).all()
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = gradient
            optimizer.step()
        with torch.no_grad():
            indices = best.argmin(1)
            code = best_code[torch.arange(n, device=state.device), indices]
            loss, k, w, details = self.objective_shared(state, base_z, contexts, code)
            result = action_merge(k, w, self.bank.base(state, base_z), c)
        self.previous = code.detach().clone()
        self.last_plan = {'optimizer_iterations': c.steps, 'starts': starts,
            'coarse_network_candidates': count, 'simulator_queries': 0,
            'agreement_horizon': c.agreement_horizon, 'optimized_scalars': 1,
            'selected_common_code': code.tolist(), 'selected_codes': code.expand(-1, 2).tolist(),
            'requires_semantically_aligned_private_codes': True,
            'private_adapter_weights_shared': False, 'original_base_inputs_fixed': True,
            'objective': loss.tolist(), **{key: value.tolist() for key, value in details.items()}}
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result.detach()
