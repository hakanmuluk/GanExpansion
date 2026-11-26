"""Original action-generating base plus classical adapters with wider codes.

No new learned module is introduced. Only the input width of the existing
adapter mapping layer changes. The action head and observed context interface
remain those of the original action adapter.
"""
from dataclasses import replace
import torch
from .legacy import load_skill,digest_file
from .action_adapter_diffusion import actor_class
from .inference_guard import forbidden


def initialize(environment,skill,code_dim,device='cuda'):
    original,unused,source=load_skill(environment,skill,device)
    G,_,base=actor_class(environment);config=replace(original.config,new_z_dim=code_dim)
    actor=G(base,config,device)
    weights={k:v.clone() for k,v in original.addon_state().items()}
    old=weights['mapping.0.weight'];new=actor.mapping[0].weight.detach().clone()*.05
    new[:,:original.config.new_z_dim]=old[:,:original.config.new_z_dim]
    if config.sign_feature:new[:,-1:]=old[:,-1:]
    weights['mapping.0.weight']=new
    missing,unexpected=actor.load_state_dict(weights,strict=False)
    assert not unexpected and all(n.startswith(('base.','original_encoder.')) for n in missing)
    actor.original_encoder.register_forward_pre_hook(forbidden)
    return actor,source


def load(path,device='cuda'):
    saved=torch.load(path,map_location=device,weights_only=False)
    G,C,base=actor_class(saved['environment']);assert digest_file(base)==saved['base_sha256']
    actor=G(base,C(**saved['adapter_config']),device)
    missing,unexpected=actor.load_state_dict(saved['addon'],strict=False)
    assert not unexpected and all(n.startswith(('base.','original_encoder.')) for n in missing)
    actor.original_encoder.register_forward_pre_hook(forbidden)
    return actor.eval().requires_grad_(False),saved


def expand(actor,environment,rank=16,hidden=256,modulation_rank=32,output_rank=64):
    """Widen the same adapter graph, keeping old outputs initially intact."""
    G,_,base=actor_class(environment);old=actor.config
    values=dict(rank=max(old.rank,rank),context_hidden=max(old.context_hidden,hidden),
                modulation_rank=max(old.modulation_rank,modulation_rank))
    if environment=='cube':values['output_rank']=max(old.output_rank,output_rank)
    enlarged=G(base,replace(old,**values),next(actor.parameters()).device)
    previous=actor.addon_state();current=enlarged.addon_state()
    assert previous.keys()==current.keys(),'Adapter module graph changed'
    for name,value in previous.items():
        target=current[name]
        if target.shape==value.shape:target.copy_(value);continue
        assert target.ndim==value.ndim and all(a>=b for a,b in zip(target.shape,value.shape)),name
        if target.ndim==1:target[:len(value)].copy_(value);continue
        assert target.ndim==2,name
        rows,cols=value.shape
        target[:rows,:cols].copy_(value)
        # Keep the old output units independent of new input units. New down
        # features retain random initialization; new up columns start at zero.
        if target.shape[1]>cols:target[:rows,cols:].zero_()
    missing,unexpected=enlarged.load_state_dict(current,strict=False)
    assert not unexpected and all(n.startswith(('base.','original_encoder.')) for n in missing)
    enlarged.original_encoder.register_forward_pre_hook(forbidden)
    assert actor.frozen_digest()==enlarged.frozen_digest()
    return enlarged
