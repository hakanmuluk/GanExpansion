"""Deployment API: fixed episode latent, cached 16-action prediction, 8 executed."""
import numpy as np
import torch
from .models import ModelConfig,Generator


class CubePolicy:
    def __init__(self,path,device="cpu"):
        checkpoint=torch.load(path,map_location=device,weights_only=False)
        self.generator=Generator(ModelConfig(**checkpoint["model"])).to(device).eval()
        weights=checkpoint.get("generator_ema",checkpoint["generator"])
        self.generator.load_state_dict(weights)
        self.device=device
        self.execute=8
        self.z=None
        self.chunk=None
        self.index=0

    def reset(self,z=None,rng=None):
        rng=np.random.default_rng() if rng is None else rng
        self.z=np.asarray(rng.standard_normal(2) if z is None else z,np.float32).copy()
        if self.z.shape!=(2,) or not np.isfinite(self.z).all():raise ValueError("Expected a finite 2D episode latent")
        self.chunk=None;self.index=0
        return self.z.copy()

    @torch.inference_mode()
    def act(self,state):
        if self.z is None:raise RuntimeError("Call reset once per episode")
        if self.index%self.execute==0:
            x=torch.as_tensor(state,dtype=torch.float32,device=self.device)
            z=torch.as_tensor(self.z,device=self.device)
            self.chunk=self.generator(x[None],z[None])[0].cpu().numpy()
        action=self.chunk[self.index%self.execute].copy()
        self.index+=1
        action[4]=1 if action[4]>=0 else -1
        return action
