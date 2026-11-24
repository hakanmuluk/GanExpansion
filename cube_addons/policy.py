"""Deployment with independent fixed episode codes and a binary skill switch."""
import numpy as np
import torch
from .evaluate import load_adapter


class CubeSkillPolicy:
    def __init__(self,checkpoint,base='cube/artifacts/final/base_policy.pt',device='cpu'):
        self.generator,self.encoder,self.metadata=load_adapter(checkpoint,base,device)
        self.device=device;self.base_z=None;self.new_z=None;self.gate=0;self.chunk=None;self.index=0;self.execute=8

    def reset(self,base_z=None,new_z=None,gate=1,rng=None):
        rng=rng or np.random.default_rng()
        if gate not in (0,1):raise ValueError('Skill switch must be 0 or 1')
        self.base_z=np.asarray(rng.standard_normal(2) if base_z is None else base_z,np.float32).copy()
        sampled_new=(rng.standard_normal(self.generator.config.new_z_dim) if gate else np.zeros(self.generator.config.new_z_dim)) if new_z is None else new_z
        self.new_z=np.asarray(sampled_new,np.float32).copy()
        if self.base_z.shape!=(2,) or not np.isfinite(self.base_z).all():raise ValueError('Expected finite original code of shape (2,)')
        if self.new_z.shape!=(self.generator.config.new_z_dim,) or not np.isfinite(self.new_z).all():raise ValueError('Invalid skill code')
        self.gate=int(gate);self.chunk=None;self.index=0;self.execute=self.generator.execute_steps if self.gate else 8
        return self.base_z.copy(),self.new_z.copy()

    @torch.inference_mode()
    def act(self,state,context=None):
        if self.base_z is None:raise RuntimeError('Call reset once per episode')
        if self.index%self.execute==0:
            s=torch.as_tensor(state,dtype=torch.float32,device=self.device)[None]
            z=torch.as_tensor(self.base_z,device=self.device)[None]
            n=torch.as_tensor(self.new_z,device=self.device)[None]
            c=torch.as_tensor(context,dtype=torch.float32,device=self.device)[None] if context is not None else None
            self.chunk=self.generator(s,z,n,c,self.gate)[0].cpu().numpy()
        action=self.chunk[self.index%self.execute].copy();self.index+=1
        action[4]=1. if action[4]>=0 else -1.
        return action
