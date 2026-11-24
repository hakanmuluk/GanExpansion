"""Frozen original Cube generators and explicit summed-adapter execution."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from cube_addons.evaluate import load_adapter
from cube_addons.models import state_digest


class FrozenBank(nn.Module):
    def __init__(self,seed=42,device='cuda',base='cube/artifacts/final/base_policy.pt'):
        super().__init__();paths=[f'cube_addon_artifacts/final/{s}_s{seed}/adapter.pt' for s in ['keepout','waypoint']]
        self.k,self.ek,self.meta_k=load_adapter(paths[0],base,device)
        self.w,self.ew,self.meta_w=load_adapter(paths[1],base,device)
        assert state_digest(self.k.base,self.k.original_encoder)==state_digest(self.w.base,self.w.original_encoder)
        assert all(not m.config.finetune_base and m.execute_steps==8 for m in [self.k,self.w])
        self.eval().requires_grad_(False);self.fingerprint=state_digest(self);self.calls={};self.seed=seed
        self.files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [base,*paths]}
        self.dynamics=None;self.common_margin=.005

    def reset_counts(self):self.calls={}
    def count(self,key,n,invocations=1):
        row=self.calls.setdefault(key,{'batched_calls':0,'candidate_equivalents':0});row['batched_calls']+=invocations;row['candidate_equivalents']+=n

    def pair(self,state,bz,z,ctx):
        self.count('independent',2*state.shape[0],2)
        return self.k(state,bz,z[:,:1],ctx[0]),self.w(state,bz,z[:,1:],ctx[1])

    def base(self,state,bz):
        self.count('base',state.shape[0]);return self.k.base(state,bz)

    def summed(self,state,bz,z,ctx,gates=(1,1)):
        if gates==(1,0):return self.k(state,bz,z[:,:1],ctx[0])
        if gates==(0,1):return self.w(state,bz,z[:,1:],ctx[1])
        if gates==(0,0):return self.base(state,bz)
        if any(m.config.transport_gate for m in [self.k,self.w]):raise ValueError('Summed baseline has not been defined for transport-gated checkpoints')
        self.count('summed_adapter',state.shape[0]);base=self.k.base;b=base.config;x=base.normalize(state);conditions=[]
        for model,code,context in zip([self.k,self.w],[z[:,:1],z[:,1:]],ctx):
            c=model.config
            if c.no_new_latent:code=torch.zeros_like(code)
            if c.sign_feature:code=torch.cat([code,torch.sign(code[:,:1])],-1)
            if c.ignore_time:context=torch.cat([context[:,:16],torch.zeros_like(context[:,16:17]),context[:,17:]],-1)
            conditions.append(model.context(torch.cat([x,context,bz,model.mapping(code)],-1)))
        code=torch.cat([bz,torch.sign(bz[:,:1])],-1) if b.route_feature else bz;style=base.mapping(code)
        h=base.state_input(x)
        for model in [self.k,self.w]:
            if model.input_lora is not None:h=h+model.input_lora(x)
        h=F.leaky_relu(h,.2)
        for i,block in enumerate(base.blocks):
            linear=block.linear(h)
            for model in [self.k,self.w]:
                if model.layer_lora is not None:linear=linear+model.layer_lora[i](h)
            scale,shift=block.style(style).chunk(2,-1)
            for model,condition in zip([self.k,self.w],conditions):
                ds,dt=model.modulations[i](condition).chunk(2,-1);scale=scale+ds;shift=shift+dt
            h=(h+F.leaky_relu(block.norm(linear)*(1+scale)+shift,.2))*2**-.5
        out=base.to_actions(h)
        for model in [self.k,self.w]:
            if model.output_lora is not None:out=out+model.output_lora(h)
        scales=torch.zeros_like(out);shifts=torch.zeros_like(out)
        for model,condition in zip([self.k,self.w],conditions):
            if model.output_modulation is not None:
                ds,dt=model.output_modulation(condition).chunk(2,-1);scales=scales+ds;shifts=shifts+dt
        return (out*(1+scales)+shifts).tanh().reshape(-1,b.horizon,b.action_dim)

    def execute(self,state,bz,z,ctx,family='sum',pair=None):
        if family=='grip_safe_blend':
            a,w=pair if pair is not None else self.pair(state,bz,z,ctx);k=ctx[0]
            cube_z=.2*k[:,2]+.2+.15*k[:,6];active=(k[:,13]>.5)&(k[:,14]>.5)&(cube_z>.10)
            xyz=torch.where(active[:,None,None],.75*w[:,:,:3]+.25*a[:,:,:3],w[:,:,:3])
            return torch.cat([xyz,w[:,:,3:]],-1)
        if family in ['projected_waypoint','common']:
            from .common import project
            if self.dynamics is None:
                import json
                self.dynamics=json.loads(Path(f'cube_composition_artifacts/dynamics/seed{self.seed}.json').read_text())
            a,w=pair if pair is not None else self.pair(state,bz,z,ctx)
            self.count('common_projection',len(state))
            return project(w,state,ctx,self.dynamics['coefficients'],a if family=='common' else None,margin=self.common_margin)
        if family=='sum':return self.summed(state,bz,z,ctx)
        if family=='base':return self.base(state,bz)
        if pair is None and family=='keepout':
            self.count('keepout',len(state));return self.k(state,bz,z[:,:1],ctx[0])
        if pair is None and family=='waypoint':
            self.count('waypoint',len(state));return self.w(state,bz,z[:,1:],ctx[1])
        a,b=pair if pair is not None else self.pair(state,bz,z,ctx)
        if family=='keepout':return a
        if family=='waypoint':return b
        if family=='average':return (a+b)/2
        if family=='residual':return (a+b-self.base(state,bz)).clamp(-1,1)
        raise ValueError(family)

    def assert_frozen(self):
        assert state_digest(self)==self.fingerprint
        assert all(not p.requires_grad and p.grad is None for p in self.parameters())
