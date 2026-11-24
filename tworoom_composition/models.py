"""Read-only skill bank and explicit additive-weight composition baseline."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from tworoom_addons.evaluate import load
from tworoom_addons.models import state_digest
from tworoom_gan.style import canonical_state


class FrozenBank(nn.Module):
    def __init__(self,keepout='addon_artifacts/final/keepout.pt',waypoint='addon_artifacts/final/waypoint.pt',
                 base='artifacts/final/base_policy.pt',device='cuda'):
        super().__init__();self.k,self.ek,self.meta_k=load(keepout,device,base)
        self.w,self.ew,self.meta_w=load(waypoint,device,base)
        assert state_digest(self.k.base)==state_digest(self.w.base)
        assert state_digest(self.k.original_encoder)==state_digest(self.w.original_encoder)
        assert not self.k.config.finetune_base and not self.w.config.finetune_base
        self.eval().requires_grad_(False);self.fingerprint=state_digest(self)
        self.files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [base,keepout,waypoint]}
        self.calls={};self.dimensions=(self.k.config.context_dim,self.w.config.context_dim)

    def reset_counts(self):self.calls={}
    def count(self,key,n,invocations=1):
        row=self.calls.setdefault(key,{'batched_calls':0,'candidate_equivalents':0});row['batched_calls']+=invocations;row['candidate_equivalents']+=n

    def pair(self,state,bz,z,ctx):
        n=state.numel()//10;self.count('independent',2*n,2)
        return self.k(state,bz,z[...,:1],ctx[0]),self.w(state,bz,z[...,1:],ctx[1])

    def base(self,state,bz):
        self.count('base',state.numel()//10);return self.k.base(state,bz)

    def additive(self,state,bz,z,ctx,gates=(1,1)):
        if gates==(1,0):return self.k(state,bz,z[...,:1],ctx[0])
        if gates==(0,1):return self.w(state,bz,z[...,1:],ctx[1])
        if gates==(0,0):return self.base(state,bz)
        self.count('summed_adapter',state.numel()//10)
        leading=state.shape[:-1];x,reflection=canonical_state(state);x=x.reshape(-1,10)
        conditions=[]
        for model,code,context in zip([self.k,self.w],[z[...,:1],z[...,1:]],ctx):
            code=code.reshape(-1,1)
            if model.config.sign_feature:code=torch.cat([code,torch.where(code>=0,1.,-1.)],-1)
            conditions.append(model.context(torch.cat([model.mapping(code),x,context.reshape(-1,model.config.context_dim)],-1)))
        base=self.k.base;style=base.mapping(bz.reshape(-1,2))
        h=base.state_input(x)
        for model in [self.k,self.w]:
            if model.config.rank:h=h+model.input_lora(x)
        h=F.leaky_relu(h,.2)
        for i,block in enumerate(base.blocks):
            linear=block.linear(h)
            for model in [self.k,self.w]:
                if model.config.rank:linear=linear+model.layer_lora[i](h)
            scale,shift=block.style(style).chunk(2,-1)
            for model,condition in zip([self.k,self.w],conditions):
                ds,dt=model.modulations[i](condition).chunk(2,-1);scale=scale+ds;shift=shift+dt
            h=(h+F.leaky_relu(block.norm(linear)*(1+scale)+shift,.2))*2**-.5
        out=base.to_actions(h)
        for model in [self.k,self.w]:
            if model.config.rank:out=out+model.output_lora(h)
        # Simultaneous scale/shift addition, rather than sequential overwriting.
        ds_total=torch.zeros_like(out);dt_total=torch.zeros_like(out)
        for model,condition in zip([self.k,self.w],conditions):
            if model.output_modulation is not None:
                ds,dt=model.output_modulation(condition).chunk(2,-1);ds_total=ds_total+ds;dt_total=dt_total+dt
        out=out*(1+ds_total)+dt_total
        result=out.tanh().reshape(*leading,16,2)
        return result*torch.stack([reflection,torch.ones_like(reflection)],-1)[...,None,:]

    def execute(self,state,bz,z,ctx,family='sum',pair=None):
        if family=='sum':return self.additive(state,bz,z,ctx)
        if family=='base':return self.base(state,bz)
        a,b=pair if pair is not None else self.pair(state,bz,z,ctx)
        if family=='keepout':return a
        if family=='waypoint':return b
        if family=='average':return (a+b)/2
        if family=='residual':return (a+b-self.base(state,bz)).clamp(-1,1)
        if family=='handoff':
            x=(state[...,:2]+1)*112;goal=(state[...,2:4]+1)*112
            canonical_x=112+torch.where(goal[...,0]>=112,1.,-1.)*(x[...,0]-112)
            return torch.where((canonical_x<128)[...,None,None],a,b)
        raise ValueError(family)

    def assert_frozen(self):
        assert state_digest(self)==self.fingerprint
        assert all(not p.requires_grad and p.grad is None for p in self.parameters())
