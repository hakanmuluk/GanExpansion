"""Action refinement using ONLY the original base and classical skill adapters.

The original adapter's existing context layer additionally receives the current
candidate action. Its first action-head logits define a residual edit. There is
no learned route, executor, critic, inverse encoder or separate projection net.
"""
from dataclasses import replace,asdict
import torch
from torch import nn
from .action_adapter_diffusion import actor_class
from .legacy import load_skill,digest_file
from .inference_guard import forbidden


def physical_sign(actor,state):
    if getattr(actor.base_config,'canonical',False):
        from tworoom_gan.style import canonical_state
        _,reflection=canonical_state(state)
        return torch.stack([reflection,torch.ones_like(reflection)],-1)
    return state.new_ones(len(state),2)


def first_logits(actor,environment,state,base_z,code=None,context=None):
    if environment=='cube':
        return actor.base_logits(state,base_z)[:,0] if code is None else actor(state,base_z,code,context,return_logits=True)[1][:,0]
    collected={};handles=[]
    def capture(key):
        def hook(module,inputs,output):collected[key]=output
        return hook
    handles.append(actor.base.to_actions.register_forward_hook(capture('base')))
    if code is not None and actor.config.rank:handles.append(actor.output_lora.register_forward_hook(capture('lora')))
    if code is not None and actor.output_modulation is not None:handles.append(actor.output_modulation.register_forward_hook(capture('modulation')))
    try:
        if code is None:actor.base(state,base_z)
        else:actor(state,base_z,code,context)
    finally:
        for handle in handles:handle.remove()
    value=collected['base']
    if 'lora' in collected:value=value+collected['lora']
    if 'modulation' in collected:
        scale,shift=collected['modulation'].chunk(2,-1);value=value*(1+scale)+shift
    return value.reshape(len(state),actor.base_config.horizon,2)[:,0]*physical_sign(actor,state)


class RefinementAdapter(nn.Module):
    def __init__(self,environment,skill,actor,own_context_dim):
        super().__init__();self.environment=environment;self.skill=skill;self.actor=actor
        self.own_context_dim=own_context_dim;self.action_dim=2 if environment=='tworoom' else 5
        assert actor.config.context_dim==own_context_dim+self.action_dim
        actor.original_encoder.register_forward_pre_hook(forbidden)

    def delta(self,state,base_z,code,context,candidate,base_logits=None):
        if base_logits is None:base_logits=first_logits(self.actor,self.environment,state,base_z)
        observed_candidate=candidate
        if self.environment=='tworoom':observed_candidate=candidate*physical_sign(self.actor,state)
        own=torch.cat([context,observed_candidate],-1)
        value=first_logits(self.actor,self.environment,state,base_z,code,own)-base_logits
        if self.environment=='cube':value=torch.cat([value[:,:3],torch.zeros_like(value[:,3:])],-1)
        return value

    def forward(self,state,base_z,code,context,candidate,base_logits=None):
        delta=self.delta(state,base_z,code,context,candidate,base_logits)
        # tanh(atanh(candidate)+delta), expressed to preserve exact identity
        # when delta==0, including saturated unmodified action coordinates.
        d=delta.tanh();value=((candidate+d)/(1+candidate*d).clamp_min(1e-6)).clamp(-1,1)
        if self.environment=='cube':value=torch.cat([value[:,:3],candidate[:,3:]],-1)
        return value


def initialize(environment,skill,device='cuda'):
    original,unused,source=load_skill(environment,skill,device)
    G,_,base=actor_class(environment);dim=2 if environment=='tworoom' else 5
    options=dict(rank=8,context_hidden=128,modulation_rank=16,output_modulation=True,
                 context_dim=original.config.context_dim+dim,new_z_dim=1)
    if environment=='cube':options['output_rank']=64
    config=replace(original.config,**options)
    # Fresh zero-initialized LoRA/FiLM up projections make the initial operator
    # an identity edit, using the unchanged original action base.
    actor=G(base,config,device)
    result=RefinementAdapter(environment,skill,actor,original.config.context_dim).to(device)
    del original,unused
    return result,source


def compact(model):
    base='artifacts/final/base_policy.pt' if model.environment=='tworoom' else 'cube/artifacts/final/base_policy.pt'
    return {'environment':model.environment,'skill':model.skill,'adapter_config':asdict(model.actor.config),
            'own_context_dim':model.own_context_dim,'addon':model.actor.addon_state(),'base_sha256':digest_file(base)}


def load(path,device='cuda'):
    saved=torch.load(path,map_location=device,weights_only=False)
    G,C,base=actor_class(saved['environment']);assert digest_file(base)==saved['base_sha256']
    actor=G(base,C(**saved['adapter_config']),device)
    missing,unexpected=actor.load_state_dict(saved['addon'],strict=False)
    assert not unexpected and all(n.startswith(('base.','original_encoder.')) for n in missing)
    model=RefinementAdapter(saved['environment'],saved['skill'],actor,saved['own_context_dim']).to(device)
    return model.eval().requires_grad_(False),saved
