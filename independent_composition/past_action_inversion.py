"""Constrain existing-code inversion by actually observed past controls.

Original frozen generators, private codes and base-amplitude inputs only. The
same candidate inputs must explain the next action prefix and recent issued
first controls at their real observed states. No new encoder/network, predicted
future state, simulation, geometric task score, or output mixing is used.
"""
from dataclasses import dataclass
import torch
from .amplitude_composition import AmplitudeConfig, AmplitudeComposer


@dataclass
class PastActionConfig(AmplitudeConfig):
    past_weight: float = 1.
    max_history: int = 4


class PastActionComposer(AmplitudeComposer):
    def __init__(self, bank, config, seed=241):
        assert config.past_weight >= 0 and config.max_history > 0
        super().__init__(bank, config, seed)

    def objective(self, state, bz, contexts, z, history=None, previous=None):
        loss, k, w, details = super().objective(state, bz, contexts, z, history, previous)
        reconstruction = torch.zeros_like(loss)
        if self.records and self.config.past_weight:
            original_batch = self.records[0][0].shape[0]
            assert len(state) % original_batch == 0
            repeats = len(state) // original_batch
            hs = torch.stack([r[0] for r in self.records], 1).repeat_interleave(repeats, 0)
            hb = torch.stack([r[1] for r in self.records], 1).repeat_interleave(repeats, 0)
            hc = tuple(torch.stack([r[2][j] for r in self.records], 1).repeat_interleave(repeats, 0)
                       for j in [0, 1])
            ha = torch.stack([r[3] for r in self.records], 1).repeat_interleave(repeats, 0)
            n, t = hs.shape[:2]
            gates = [getattr(self.bank, 'keepout_gate', 1), getattr(self.bank, 'waypoint_gate', 1)]
            assert sum(gates) > 0
            for j, (actor, gate) in enumerate(zip([self.bank.k, self.bank.w], gates)):
                if not gate:
                    continue  # An ablated skill's context cannot enter this term.
                base = torch.stack([hb[..., 0].sign() * z[:, j + 2:j + 3], hb[..., 1]], -1)
                code = z[:, None, j:j + 1].expand(-1, t, -1).reshape(n * t, 1)
                self.bank.count('observed_past_reconstruction', n * t, 1)
                prediction = actor(hs.flatten(0, 1), base.flatten(0, 1), code,
                                   hc[j].flatten(0, 1), gate=gate)[:, 0].reshape(n, t, -1)
                reconstruction = reconstruction + (
                    self.coordinates(prediction) - self.coordinates(ha)).square().mean((1, 2)) / sum(gates)
        details['observed_past_action_mse'] = reconstruction
        return loss + self.config.past_weight * reconstruction, k, w, details

    def sample(self, state, contexts, base_z):
        result = super().sample(state, contexts, base_z)
        self.pending = (state.detach().clone(), base_z.detach().clone(),
                        tuple(x.detach().clone() for x in contexts))
        self.last_plan.update(observed_history_replans=len(self.records),
                              past_reconstruction_weight=self.config.past_weight,
                              history_contains_observed_states_and_issued_first_controls_only=True)
        return result

    @torch.no_grad()
    def observe_action(self, action):
        if self.pending is None:
            return
        value = torch.as_tensor(action, device=self.device, dtype=self.pending[0].dtype)
        if value.ndim == 1:
            value = value[None]
        self.records.append((*self.pending, value.detach().clone()))
        self.pending = None
        self.records = self.records[-self.config.max_history:]
