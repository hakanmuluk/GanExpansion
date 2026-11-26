"""Code and action cycles using each adapter's EXISTING private encoder.

Only original one-dimensional private codes are optimized. Original encoder
weights must have been calibrated explicitly on static current-state/action
chunks. No imagined future states, shared parameters, new neural architectures,
simulator, analytic task score or generator training occurs during inference.
"""
from dataclasses import dataclass
import torch
from cube_addons.models import state_digest
from .gate_ablation_bank import GateAblationBank
from .legacy import digest_file
from .quantile_composition import QuantileConfig, QuantileComposer
from .unchanged_composition import action_merge
from .calibrate_original_chunk_encoder import encode_chunk
from .inference_guard import forbidden


class ChunkCycleBank(GateAblationBank):
    def __init__(self, keepout, waypoint, base, keepout_encoder, waypoint_encoder,
                 device='cuda', **kwargs):
        super().__init__(keepout, waypoint, base, device, **kwargs)
        for actor_path, path, encoder in zip([keepout, waypoint],
                [keepout_encoder, waypoint_encoder], [self.ek, self.ew]):
            ck = torch.load(path, map_location=device, weights_only=False)
            assert ck['encoder_class'] == type(encoder).__name__ == 'SkillEncoder'
            assert ck['calibration_interface'] == 'static_current_observation_action_chunk'
            assert ck['config']['checkpoint_sha256'] == digest_file(actor_path)
            assert ck['config']['base_sha256'] == digest_file(base)
            assert ck['config']['new_learned_modules'] == 0
            encoder.load_state_dict(ck['encoder'], strict=True)
            self.files[str(path)] = digest_file(path)
        self.eval().requires_grad_(False)
        self.fingerprint = state_digest(self)
        self.original_private_encoders_chunk_calibrated = True


@dataclass
class ChunkCycleConfig(QuantileConfig):
    latent_cycle: float = .002
    action_cycle: float = .5


class ChunkCycleComposer(QuantileComposer):
    def __init__(self, bank, config, seed=241):
        assert bank.original_private_encoders_chunk_calibrated
        assert config.history == config.prior == config.trust == config.continuity == 0
        assert config.latent_cycle >= 0 and config.action_cycle >= 0
        super().__init__(bank, config, seed)
        # OriginalComposer forbids uncalibrated full-trajectory inverses. Only
        # these two recalibrated existing private encoders are now permitted.
        # Both immutable ORIGINAL BASE encoders retain their tripwires.
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
            active = self.bank.keepout_gate + self.bank.waypoint_gate
            assert active > 0
            for j, (actor, encoder, gate) in enumerate(zip(
                    [self.bank.k, self.bank.w], [self.bank.ek, self.bank.ew],
                    [self.bank.keepout_gate, self.bank.waypoint_gate])):
                if not gate:
                    continue  # Disabled skill's context must not leak via E.
                self.bank.count('private_inverse', len(state), 1)
                mu, _ = encode_chunk(encoder, state, contexts[j], bz, chosen)
                latent_loss = latent_loss + (mu-z[:, j:j+1]).square().mean(1)/active
                if c.action_cycle:
                    recovered = mu.clamp(-c.maximum_magnitude, c.maximum_magnitude)
                    self.bank.count('private_reconstruction', len(state), 1)
                    reconstructed = actor(state, bz, recovered, contexts[j])
                    delta = (self.coordinates(reconstructed[:, :c.agreement_horizon]) -
                             self.coordinates(chosen[:, :c.agreement_horizon]))
                    action_loss = action_loss + delta.square().mean((1, 2))/active
        loss = loss + c.latent_cycle*latent_loss + c.action_cycle*action_loss
        details.update(execution_code_cycle=latent_loss, execution_action_cycle=action_loss)
        return loss, k, w, details
