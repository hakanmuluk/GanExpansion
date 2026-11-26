"""Denoising through the original action base and its LoRA/FiLM adapters.

This wrapper creates no learned layers. Noisy actions and fixed time features
are additional inputs to the adapter's existing context layer. Its original
action head predicts clean actions. No route, score MLP, judge or inverse model
is instantiated or called.
"""
from dataclasses import asdict,replace,dataclass
import math
import torch
from torch import nn
from .legacy import load_skill,digest_file
from .inference_guard import forbidden


@dataclass
class ActionNoiseConfig:
    state_dim:int
    context_dim:int
    action_dim:int
    horizon:int=16
    diffusion_steps:int=100
    base_z_dim:int=2
    score_residual:bool=False
    base_std:float=.2


def actor_class(environment):
    if environment=='tworoom':
        from tworoom_addons.models import GatedGenerator,AdapterConfig
        return GatedGenerator,AdapterConfig,'artifacts/final/base_policy.pt'
    if environment=='cube':
        from cube_addons.models import GatedGenerator,AdapterConfig
        return GatedGenerator,AdapterConfig,'cube/artifacts/final/base_policy.pt'
    raise ValueError(environment)


class DenoisingAdapter(nn.Module):
    def __init__(self,environment,actor,config):
        super().__init__();self.environment=environment;self.actor=actor;self.config=c=config
        assert not actor.config.finetune_base and c.base_std>0
        assert actor.config.context_dim==c.context_dim+c.horizon*c.action_dim+10
        actor.original_encoder.register_forward_pre_hook(forbidden)
        t=torch.linspace(0,1,c.diffusion_steps+1)
        a=torch.cos((t+.008)/1.008*math.pi/2).square();a=a/a[0]
        beta=(1-a[1:]/a[:-1]).clamp(.0001,.999)
        self.register_buffer('alpha_bar',(1-beta).cumprod(0))

    def predict_clean(self,noisy,time,state,context,base_z,present=None,return_logits=False):
        if present is None:present=state.new_ones(len(state),1)
        t=time.to(state.dtype)/max(self.config.diffusion_steps-1,1)
        features=torch.stack([t,*[(math.pi*f*t).sin() for f in [1,2,4,8]],
                              *[(math.pi*f*t).cos() for f in [1,2,4,8]]],-1)
        ctx=torch.cat([context*present,noisy.flatten(1),features,present],-1)
        # Noise chooses the action mode. No separate style encoder is needed.
        z=state.new_full((len(state),self.actor.config.new_z_dim),.8)
        if self.environment=='cube':
            return self.actor(state,base_z,z,ctx,return_logits=return_logits)
        if return_logits:raise ValueError('TwoRoom uses continuous controls only')
        return self.actor(state,base_z,z,ctx)

    def forward(self,noisy,time,state,context,base_z,base_actions,present=None):
        clean=self.predict_clean(noisy,time,state,context,base_z,present)
        a=self.alpha_bar[time,None,None]
        return (noisy-a.sqrt()*clean)/(1-a).sqrt().clamp_min(1e-5)

    def base_epsilon(self,noisy,time,base_actions):
        a=self.alpha_bar[time,None,None]
        return (1-a).sqrt()*(noisy-a.sqrt()*base_actions)/(1-a+a*self.config.base_std**2)


def initialize(environment,skill,device='cuda',seed=42):
    old,unused_encoder,source=load_skill(environment,skill,device,seed)
    G,_,base_path=actor_class(environment)
    b=old.base_config
    coordinates=2 if environment=='tworoom' else b.action_dim
    config=ActionNoiseConfig(10 if environment=='tworoom' else b.obs_dim,
                           old.config.context_dim,coordinates,horizon=b.horizon)
    extended=replace(old.config,context_dim=old.config.context_dim+b.horizon*coordinates+10)
    actor=G(base_path,extended,device)
    weights={k:v.detach().clone() for k,v in old.addon_state().items()}
    previous=weights['context.0.weight'];current=torch.zeros_like(actor.context[0].weight)
    prefix=previous.shape[1] if environment=='tworoom' else b.obs_dim+old.config.context_dim
    current[:,:prefix]=previous[:,:prefix]
    if prefix<previous.shape[1]:current[:,prefix+b.horizon*coordinates+10:]=previous[:,prefix:]
    weights['context.0.weight']=current
    missing,unexpected=actor.load_state_dict(weights,strict=False)
    assert not unexpected and all(k.startswith(('base.','original_encoder.')) for k in missing)
    for name,parameter in actor.named_parameters():
        parameter.requires_grad_(not name.startswith(('base.','original_encoder.')))
    assert not any(p.requires_grad for p in actor.base.parameters())
    del old,unused_encoder
    return DenoisingAdapter(environment,actor,config).to(device),source


def load(path,device='cuda'):
    saved=torch.load(path,map_location=device,weights_only=False)
    G,C,base_path=actor_class(saved['environment'])
    assert digest_file(base_path)==saved['base_sha256']
    actor=G(base_path,C(**saved['adapter_config']),device)
    missing,unexpected=actor.load_state_dict(saved['addon'],strict=False)
    assert not unexpected and all(k.startswith(('base.','original_encoder.')) for k in missing)
    model=DenoisingAdapter(saved['environment'],actor,ActionNoiseConfig(**saved['model_config'])).to(device)
    return model.eval().requires_grad_(False),saved


def compact(model):
    return {'environment':model.environment,'adapter_config':asdict(model.actor.config),
            'model_config':asdict(model.config),'addon':model.actor.addon_state()}
