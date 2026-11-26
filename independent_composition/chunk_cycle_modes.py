"""Existing base-input inversion plus independently calibrated private cycles.

No new neural module. This combines existing algorithm variants with a common
network-only objective; each private encoder receives its actor's actual base
input, including that actor's optimized speed coordinate.
"""
from dataclasses import dataclass
import torch
from .chunk_cycle_composition import ChunkCycleComposer
from .mode_composition import ModeConfig, ModeComposer
from .speed_composition import SpeedComposer
from .unchanged_composition import action_merge
from .calibrate_original_chunk_encoder import encode_chunk
from .inference_guard import forbidden


@dataclass
class ChunkCycleModeConfig(ModeConfig):
    latent_cycle: float = .002
    action_cycle: float = .5


class ChunkCycleSpeedComposer(SpeedComposer):
    def __init__(self, bank, config, seed=241):
        assert bank.original_private_encoders_chunk_calibrated
        assert config.history == config.prior == config.trust == config.continuity == 0
        assert config.latent_cycle >= 0 and config.action_cycle >= 0
        super().__init__(bank, config, seed)
        for encoder in [bank.ek, bank.ew]:
            for key, hook in list(encoder._forward_pre_hooks.items()):
                if hook is forbidden:
                    del encoder._forward_pre_hooks[key]

    def objective(self, state, bz, contexts, z, history=None, previous=None):
        loss, k, w, details = super().objective(state, bz, contexts, z, history, previous)
        c = self.config
        latent_loss, action_loss = torch.zeros_like(loss), torch.zeros_like(loss)
        if c.latent_cycle or c.action_cycle:
            chosen = action_merge(k, w, self.bank.base(state, bz), c)
            own_base = [torch.stack([bz[:, 0], z[:, 2]], -1),
                        torch.stack([bz[:, 0], z[:, 3]], -1)]
            active = self.bank.keepout_gate + self.bank.waypoint_gate
            assert active > 0
            for j, (actor, encoder, gate) in enumerate(zip(
                    [self.bank.k, self.bank.w], [self.bank.ek, self.bank.ew],
                    [self.bank.keepout_gate, self.bank.waypoint_gate])):
                if not gate:
                    continue
                self.bank.count('private_inverse', len(state), 1)
                mu, _ = encode_chunk(encoder, state, contexts[j], own_base[j], chosen)
                latent_loss = latent_loss + (mu-z[:, j:j+1]).square().mean(1)/active
                if c.action_cycle:
                    recovered = mu.clamp(-c.maximum_magnitude, c.maximum_magnitude)
                    self.bank.count('private_reconstruction', len(state), 1)
                    reconstructed = actor(state, own_base[j], recovered, contexts[j])
                    delta = (self.coordinates(reconstructed[:, :c.agreement_horizon]) -
                             self.coordinates(chosen[:, :c.agreement_horizon]))
                    action_loss = action_loss + delta.square().mean((1, 2))/active
        details.update(execution_code_cycle=latent_loss, execution_action_cycle=action_loss)
        return loss+c.latent_cycle*latent_loss+c.action_cycle*action_loss, k, w, details


class ChunkCycleModeComposer(ModeComposer):
    def __init__(self, bank, config, seed=241):
        self.bank, self.config = bank, config
        cls = ChunkCycleSpeedComposer if config.optimize_speed_inputs else ChunkCycleComposer
        self.branches = [cls(bank, config, seed), cls(bank, config, seed)]
        self.noise = None
        self.last_plan = {}
