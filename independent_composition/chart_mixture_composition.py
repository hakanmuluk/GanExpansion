"""Local mixture consensus without averaging across original sign charts.

The original generators contain an explicit private-code sign feature. Solve
one simplex problem within each sign chart, then choose a chart pair by action
agreement alone. This avoids convexifying across disconnected latent charts;
it still supplies no guarantee that a within-chart mixture preserves a skill.
"""
import numpy as np
import torch
from .latent_mixture_composition import MixtureConfig, LatentMixtureComposer, consensus_weights


class ChartMixtureComposer(LatentMixtureComposer):
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
        charts = [np.arange(m // 2), np.arange(m // 2, m)]
        weights = []; selected = []
        for a, b in zip(k_first, w_first):
            solutions = []
            for ki in charts:
                for wi in charts:
                    ka, wa, details = consensus_weights(a[ki], b[wi], c.regularization, c.iterations)
                    full_k = np.zeros(m); full_w = np.zeros(m)
                    full_k[ki] = ka; full_w[wi] = wa
                    solutions.append((full_k, full_w, details,
                        [-1 if ki[0] == 0 else 1, -1 if wi[0] == 0 else 1]))
            best = min(solutions, key=lambda x: x[2]['objective'])
            weights.append(best[:3]); selected.append(best[3])
        chosen = 0 if c.executor == 'keepout' else 1
        coefficients = state.new_tensor(np.stack([row[chosen] for row in weights]))
        result = torch.einsum('nm,nmha->nha', coefficients, k if chosen == 0 else w)
        self.last_plan = {
            'optimizer_iterations_per_chart_pair': c.iterations,
            'chart_pairs': 4, 'network_candidates_per_role': m,
            'agreement_horizon': 1, 'required_execution_horizon': 1,
            'simulator_queries': 0, 'convex_mixture_safety_guarantee': False,
            'executor': c.executor, 'selected_original_sign_charts': selected,
            'keepout_weights': [row[0].tolist() for row in weights],
            'waypoint_weights': [row[1].tolist() for row in weights],
            **{key: [row[2][key] for row in weights] for key in weights[0][2]},
        }
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result
