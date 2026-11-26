"""Search the original common base-route sign as well as private style codes.

Two existing input modes, the same frozen original networks, network-only
agreement and original latent inversion. No route network or geometric score.
"""
from dataclasses import dataclass
import torch
from .quantile_composition import QuantileComposer
from .speed_composition import SpeedConfig, SpeedComposer


@dataclass
class ModeConfig(SpeedConfig):
    optimize_speed_inputs: bool = False


class ModeComposer:
    trace_noise_name = 'initial_existing_mode_and_private_codes'

    def __init__(self, bank, config, seed=241):
        assert config.history == config.prior == config.continuity == config.trust == 0
        self.bank, self.config = bank, config
        cls = SpeedComposer if config.optimize_speed_inputs else QuantileComposer
        self.branches = [cls(bank, config, seed), cls(bank, config, seed)]
        self.noise = None
        self.last_plan = {}

    @torch.no_grad()
    def observe_action(self, action):
        # These hypotheses deliberately use only current observed state/context.
        pass

    def sample(self, state, contexts, base_z):
        outputs, plans, noises, calls = [], [], [], []
        for branch, sign in zip(self.branches, [-1., 1.]):
            bz = base_z.clone()
            bz[:, 0] = sign * bz[:, 0].abs().clamp_min(.01)
            outputs.append(branch.sample(state, contexts, bz))
            plans.append(branch.last_plan)
            noises.append(branch.noise)
            calls.append({key: value.copy() for key, value in self.bank.calls.items()})
        with torch.no_grad():
            costs = state.new_tensor([plan['objective'] for plan in plans]).T
            selected = costs.argmin(1)
            stack = torch.stack(outputs, 1)
            result = stack[torch.arange(len(state), device=state.device), selected]
            noise_stack = torch.stack(noises, 1)
            self.noise = noise_stack[torch.arange(len(state), device=state.device), selected].detach()
        total_calls = {}
        for branch_calls in calls:
            for key, values in branch_calls.items():
                row = total_calls.setdefault(key, {name: 0 for name in values})
                for name, value in values.items():
                    row[name] += value
        self.bank.calls = total_calls
        self.last_plan = {'route_modes_considered': 2, 'simulator_queries': 0,
            'original_base_route_input_optimized': True,
            'original_base_route_magnitude_preserved': True,
            'original_base_speed_inputs_optimized': self.config.optimize_speed_inputs,
            'selected_base_route_sign': (selected * 2 - 1).tolist(),
            'branch_objectives': costs.tolist(), 'branch_plans': plans,
            'new_learned_modules': 0}
        return result.detach()
