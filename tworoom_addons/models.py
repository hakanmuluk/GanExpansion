"""Frozen base + independent LoRA and conditional style corrections."""
from dataclasses import dataclass,asdict
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from tworoom_gan.evaluate import load_checkpoint
from tworoom_gan.models import Discriminator as BaseDiscriminator
from tworoom_gan.style import canonical_state
from .geometry import CONTEXT_DIM


def state_digest(*modules):
    h=hashlib.sha256()
    for i,module in enumerate(modules):
        for key,value in sorted(module.state_dict().items()):
            h.update(f"{i}:{key}".encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@dataclass
class AdapterConfig:
    rank:int=4
    style_dim:int=64
    context_hidden:int=64
    modulation_rank:int=8
    new_z_dim:int=1
    sign_feature:bool=True
    no_new_latent:bool=False
    finetune_base:bool=False
    context_dim:int=CONTEXT_DIM
    output_modulation:bool=False


class LowRank(nn.Module):
    def __init__(self,inputs,outputs,rank):
        super().__init__();self.rank=rank
        if rank:
            self.down=nn.Linear(inputs,rank,bias=False)
            self.up=nn.Linear(rank,outputs,bias=False)
            nn.init.zeros_(self.up.weight)
    def forward(self,x):
        if not self.rank:return torch.zeros((*x.shape[:-1],0),device=x.device,dtype=x.dtype)
        return self.up(self.down(x))


class GatedGenerator(nn.Module):
    def __init__(self,base_path,config:AdapterConfig,device="cuda"):
        super().__init__();self.config=config
        self.base,self.original_encoder,self.base_metadata=load_checkpoint(base_path,device)
        self.base.eval().requires_grad_(config.finetune_base)
        self.original_encoder.eval().requires_grad_(False)
        b=self.base.config;self.base_config=b
        dz=config.new_z_dim+int(config.sign_feature)
        self.mapping=nn.Sequential(nn.Linear(dz,config.style_dim),nn.LeakyReLU(.2),
                                   nn.Linear(config.style_dim,config.style_dim),nn.LeakyReLU(.2),
                                   nn.Linear(config.style_dim,config.style_dim),nn.LeakyReLU(.2))
        self.context=nn.Sequential(nn.Linear(config.style_dim+10+config.context_dim,config.context_hidden),nn.LeakyReLU(.2),
                                   nn.Linear(config.context_hidden,config.context_hidden),nn.LeakyReLU(.2))
        self.input_lora=LowRank(10,b.hidden,config.rank)
        self.layer_lora=nn.ModuleList([LowRank(b.hidden,b.hidden,config.rank) for _ in self.base.blocks])
        self.output_lora=LowRank(b.hidden,b.horizon*2,config.rank)
        self.modulations=nn.ModuleList([nn.Sequential(nn.Linear(config.context_hidden,config.modulation_rank),nn.LeakyReLU(.2),
                                                      nn.Linear(config.modulation_rank,2*b.hidden)) for _ in self.base.blocks])
        for mod in self.modulations:
            nn.init.zeros_(mod[-1].weight);nn.init.zeros_(mod[-1].bias)
        self.output_modulation=None
        if config.output_modulation:
            self.output_modulation=nn.Sequential(nn.Linear(config.context_hidden,config.modulation_rank),nn.LeakyReLU(.2),nn.Linear(config.modulation_rank,2*b.horizon*2))
            nn.init.zeros_(self.output_modulation[-1].weight);nn.init.zeros_(self.output_modulation[-1].bias)
        self.to(device)

    def train(self,mode=True):
        super().train(mode)
        if not self.config.finetune_base:self.base.eval()
        self.original_encoder.eval()
        return self

    def forward(self,state,base_z,new_z=None,context=None,gate=1):
        # Exact bypass: no new latent, context processing, or extra arithmetic.
        if not torch.is_tensor(gate) and not gate:return self.base(state,base_z)
        if torch.is_tensor(gate) and not gate.bool().any():return self.base(state,base_z)
        leading=state.shape[:-1]
        if self.base_config.canonical:canonical,reflection=canonical_state(state)
        else:canonical,reflection=state,torch.ones_like(state[...,0])
        if new_z is None:raise ValueError("An active adapter needs its new latent")
        if context is None:raise ValueError("An active adapter needs observed constraint context")
        z=new_z.reshape(-1,self.config.new_z_dim)
        if self.config.no_new_latent:z=torch.zeros_like(z)
        if self.config.sign_feature:z=torch.cat([z,torch.where(z[...,:1]>=0,1.,-1.)],-1)
        w_new=self.mapping(z)
        c=self.context(torch.cat([w_new,canonical.reshape(-1,10),context.reshape(-1,self.config.context_dim)],-1))
        w=self.base.mapping(base_z.reshape(-1,self.base_config.z_dim))
        gate_flat=gate.reshape(-1,1).to(state.dtype) if torch.is_tensor(gate) else float(gate)
        inp=canonical.reshape(-1,10)
        h=self.base.state_input(inp)
        if self.config.rank:h=h+gate_flat*self.input_lora(inp)
        h=F.leaky_relu(h,.2)
        for block,lora,mod in zip(self.base.blocks,self.layer_lora,self.modulations):
            linear=block.linear(h)
            if self.config.rank:linear=linear+gate_flat*lora(h)
            scale,shift=block.style(w).chunk(2,-1)
            ds,dt=mod(c).chunk(2,-1)
            update=F.leaky_relu(block.norm(linear)*(1+scale+gate_flat*ds)+shift+gate_flat*dt,.2)
            h=(h+update)*2**-.5
        output=self.base.to_actions(h)
        if self.config.rank:output=output+gate_flat*self.output_lora(h)
        if self.output_modulation is not None:
            ds,dt=self.output_modulation(c).chunk(2,-1)
            output=output*(1+gate_flat*ds)+gate_flat*dt
        actions=output.tanh().reshape(*leading,self.base_config.horizon,2)
        if self.base_config.canonical:actions=actions*torch.stack([reflection,torch.ones_like(reflection)],-1)[...,None,:]
        if torch.is_tensor(gate):
            unchanged=self.base(state,base_z)
            actions=torch.where(gate.bool()[...,None,None],actions,unchanged)
        return actions

    def trainable(self):return [p for p in self.parameters() if p.requires_grad]
    def addon_state(self):return {k:v for k,v in self.state_dict().items() if not k.startswith(("base.","original_encoder."))}
    def frozen_digest(self):return state_digest(self.base,self.original_encoder)


class SkillEncoder(nn.Module):
    def __init__(self,z_dim=1,hidden=128,context_dim=CONTEXT_DIM):
        super().__init__();self.z_dim=z_dim
        self.temporal=nn.Sequential(nn.Conv1d(10+2+context_dim+2,hidden,5,stride=2,padding=2),nn.LeakyReLU(.2),
                                    nn.Conv1d(hidden,hidden,5,stride=2,padding=2),nn.LeakyReLU(.2),
                                    nn.Conv1d(hidden,hidden,3,padding=1),nn.LeakyReLU(.2))
        self.head=nn.Sequential(nn.Linear(2*hidden+10+context_dim+2,hidden),nn.LeakyReLU(.2))
        self.mu=nn.Linear(hidden,z_dim);self.logvar=nn.Linear(hidden,z_dim)
        nn.init.constant_(self.logvar.bias,-3)

    def forward(self,states,actions,contexts,base_z,lengths):
        t=actions.shape[1]
        states,sign=canonical_state(states[:,:t])
        actions=actions*torch.stack([sign,torch.ones_like(sign)],-1)
        z=base_z[:,None].expand(-1,t,-1)
        mask=torch.arange(t,device=states.device)[None]<lengths[:,None]
        x=torch.cat([states[:,:t],actions,contexts[:,:t],z],-1)*mask[...,None]
        h=self.temporal(x.transpose(1,2));n=(lengths+3)//4
        valid=torch.arange(h.shape[-1],device=h.device)[None]<n[:,None]
        mean=(h*valid[:,None]).sum(-1)/n.clamp_min(1)[:,None]
        maximum=h.masked_fill(~valid[:,None],-1e4).amax(-1)
        summary=self.head(torch.cat([mean,maximum,states[:,0],contexts[:,0],base_z],-1))
        return self.mu(summary),self.logvar(summary).clamp(-7,2)


class SkillDiscriminator(nn.Module):
    def __init__(self,base_config,source=None,context_dim=CONTEXT_DIM):
        super().__init__();h=base_config.hidden
        self.net=nn.Sequential(spectral_norm(nn.Linear(10+context_dim+base_config.horizon*3,h)),nn.LeakyReLU(.2),
                               spectral_norm(nn.Linear(h,h)),nn.LeakyReLU(.2),spectral_norm(nn.Linear(h,h//2)),nn.LeakyReLU(.2),spectral_norm(nn.Linear(h//2,1)))
        if source is not None:
            # Copy the old discriminator, with initially zero context weights.
            old=BaseDiscriminator(base_config);old.load_state_dict(source)
            for j in [2,4,6]:self.net[j].load_state_dict(old.net[j].state_dict())
            with torch.no_grad():
                a=self.net[0].parametrizations.weight.original;b=old.net[0].parametrizations.weight.original
                a.zero_();a[:,:10]=b[:,:10];a[:,10+context_dim:]=b[:,10:]
                self.net[0].bias.copy_(old.net[0].bias)
                new_sn=self.net[0].parametrizations.weight[0];old_sn=old.net[0].parametrizations.weight[0]
                new_sn._u.copy_(old_sn._u);new_sn._v.zero_()
                new_sn._v[:10]=old_sn._v[:10];new_sn._v[10+context_dim:]=old_sn._v[10:]

    def forward(self,states,actions,context,mask):
        states,sign=canonical_state(states)
        actions=actions*torch.stack([sign,torch.ones_like(sign)],-1)[...,None,:]
        x=torch.cat([states,context,(actions*mask[...,None]).flatten(-2),mask.to(states.dtype)],-1)
        return self.net(x).squeeze(-1)
