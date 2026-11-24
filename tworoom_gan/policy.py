"""Reusable inference policy: sample z ONCE at reset, execute 8 of 16 actions."""

import numpy as np
import torch
from .env import EXECUTE
from .evaluate import load_checkpoint


class TrajectoryPolicy:
    def __init__(self, checkpoint, device="cuda"):
        self.generator, _, self.metadata = load_checkpoint(checkpoint, device)
        self.device = device
        self.reset()

    def reset(self, seed=None, z=None):
        rng = torch.Generator(device=self.device)
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)
        self.z = torch.randn((1, self.generator.config.z_dim), device=self.device, generator=rng) if z is None else torch.as_tensor(z, device=self.device, dtype=torch.float32).reshape(1, -1)
        self.pending = []

    @torch.no_grad()
    def act(self, observation, normalized=False):
        """Accept original SWM's 10D state (pixel coordinates) by default."""
        if not self.pending:
            state = torch.as_tensor(observation, device=self.device, dtype=torch.float32).reshape(1, -1)
            if not normalized:
                state = state / 112.0 - 1.0
            chunk = self.generator(state, self.z)[0, :EXECUTE].cpu().numpy()
            self.pending = list(chunk)
        return np.asarray(self.pending.pop(0), dtype=np.float32)
