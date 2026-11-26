"""One-factor adapter variants with matched unchanged initial weights.

Only LoRA rank, or the presence of EXTRA skill scale/shift branches, changes.
Removing these branches intentionally disconnects both private u and context.
No replacement conditioning is introduced. Pretrained base style stays intact.
"""
from dataclasses import replace
import hashlib
import types
import torch
from torch import nn
from torch.nn import functional as F


def rng_state():
    return torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:torch.cuda.set_rng_state_all(state[1])


def only_lora_tworoom(self,state,base_z,new_z=None,context=None,gate=1):
    if not torch.is_tensor(gate) and not gate:return self.base(state,base_z)
    if torch.is_tensor(gate) and not gate.bool().any():return self.base(state,base_z)
    from tworoom_gan.style import canonical_state
    leading=state.shape[:-1]
    if self.base_config.canonical:canonical,reflection=canonical_state(state)
    else:canonical,reflection=state,torch.ones_like(state[...,0])
    w=self.base.mapping(base_z.reshape(-1,self.base_config.z_dim))
    g=gate.reshape(-1,1).to(state.dtype) if torch.is_tensor(gate) else float(gate)
    inp=canonical.reshape(-1,10);h=self.base.state_input(inp)
    if self.config.rank:h=h+g*self.input_lora(inp)
    h=F.leaky_relu(h,.2)
    for block,lora in zip(self.base.blocks,self.layer_lora):
        linear=block.linear(h)
        if self.config.rank:linear=linear+g*lora(h)
        scale,shift=block.style(w).chunk(2,-1)
        update=F.leaky_relu(block.norm(linear)*(1+scale)+shift,.2)
        h=(h+update)*2**-.5
    output=self.base.to_actions(h)
    if self.config.rank:output=output+g*self.output_lora(h)
    actions=output.tanh().reshape(*leading,self.base_config.horizon,2)
    if self.base_config.canonical:actions=actions*torch.stack([reflection,torch.ones_like(reflection)],-1)[...,None,:]
    if torch.is_tensor(gate):actions=torch.where(gate.bool()[...,None,None],actions,self.base(state,base_z))
    return actions


def only_lora_cube(self,state,base_z,new_z=None,context=None,gate=1,return_logits=False):
    assert not self.config.finetune_base and not self.config.transport_gate
    if not torch.is_tensor(gate):
        if gate not in (0,1):raise ValueError('The skill switch must be binary')
        if gate==0:
            a=self.base(state,base_z)
            return (a,self.base_logits(state,base_z)) if return_logits else a
    else:
        if not torch.all((gate==0)|(gate==1)):raise ValueError('The skill switch must be binary')
        if not gate.bool().any():
            a=self.base(state,base_z)
            return (a,self.base_logits(state,base_z)) if return_logits else a
    b=self.base_config;leading=state.shape[:-1]
    x=self.base.normalize(state).reshape(-1,b.obs_dim);bz=base_z.reshape(-1,b.z_dim)
    mapped=torch.cat([bz,torch.sign(bz[:,:1])],-1) if b.route_feature else bz
    w=self.base.mapping(mapped)
    g=gate.reshape(-1,1).to(x.dtype) if torch.is_tensor(gate) else gate
    h=self.base.state_input(x)
    if self.input_lora is not None:h=h+g*self.input_lora(x)
    h=F.leaky_relu(h,.2)
    for i,block in enumerate(self.base.blocks):
        linear=block.linear(h)
        if self.layer_lora is not None:linear=linear+g*self.layer_lora[i](h)
        scale,shift=block.style(w).chunk(2,-1)
        h=(h+F.leaky_relu(block.norm(linear)*(1+scale)+shift,.2))*2**-.5
    out=self.base.to_actions(h)
    if self.output_lora is not None:out=out+g*self.output_lora(h)
    actions=out.tanh().reshape(*leading,b.horizon,b.action_dim)
    logits=out.reshape(*leading,b.horizon,b.action_dim)
    if torch.is_tensor(gate):
        actions=torch.where(gate.bool()[...,None,None],actions,self.base(state,base_z))
        if return_logits:logits=torch.where(gate.bool()[...,None,None],logits,self.base_logits(state,base_z))
    return (actions,logits) if return_logits else actions


def remove_skill_modulation(model,environment):
    # These modules have no remaining path to actions once scale/shift is removed.
    model.mapping=None;model.context=None;model.modulations=nn.ModuleList();model.output_modulation=None
    model.forward=types.MethodType(only_lora_tworoom if environment=='tworoom' else only_lora_cube,model)
    model.ablation_kind='pure_skill_modulation_removal'
    return model


def install_adapter_factory(module,rank,reference_rank,lora_only):
    assert rank in [1,2,4,8,16]
    assert reference_rank in [1,2,4,8,16]
    original=module.GatedGenerator
    environment='tworoom' if module.__name__.startswith('tworoom') else 'cube'
    def factory(base_path,config,device='cpu'):
        before=rng_state()
        reference=original(base_path,replace(config,rank=reference_rank),device)
        after=rng_state()
        if rank==reference_rank:model=reference
        else:
            restore_rng(before);model=original(base_path,replace(config,rank=rank),device)
            current=model.state_dict();unchanged={n:v for n,v in reference.state_dict().items() if n in current and current[n].shape==v.shape}
            model.load_state_dict(unchanged,strict=False)
            assert all(torch.equal(model.state_dict()[n],v) for n,v in unchanged.items())
        restore_rng(after)
        if lora_only:remove_skill_modulation(model,environment)
        model.reference_initialization_matched=True
        return model
    module.GatedGenerator=factory

