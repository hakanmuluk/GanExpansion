"""Optimize existing private adapter mapping outputs (W), with frozen models.

The original forwards run with temporary overrides of their private mapping
outputs. Base mapping, observations, contexts, modules and weights are unchanged.
Each W is confined to a scaled neighborhood of an actual M(z) output. This is
a trust-region heuristic, not proof that all perturbed styles remain in-domain.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from .unchanged_composition import CompositionConfig,OriginalComposer,action_merge


@dataclass
class StyleCompositionConfig(CompositionConfig):
    optimize:bool=True
    history:float=0.
    keepout_sign:int=1
    preserve_mode:bool=True
    style_radius:float=.5
    style_tether:float=.05
    style_continuity:float=.01
    style_samples:int=1024
    style_condition_on_sign:bool=True


def mapped_style(actor,code):
    assert not actor.config.no_new_latent
    features=torch.cat([code,torch.sign(code[:,:1])],-1) if actor.config.sign_feature else code
    return actor.mapping(features)


@contextmanager
def mapping_override(actor,style):
    def replace(module,inputs,original):
        assert original.shape==style.shape and original.device==style.device and original.dtype==style.dtype
        return style
    handle=actor.mapping.register_forward_hook(replace)
    try:yield
    finally:handle.remove()


def style_pair(bank,state,bz,codes,contexts,styles):
    with mapping_override(bank.k,styles[0]),mapping_override(bank.w,styles[1]):
        return bank.pair(state,bz,codes,contexts)


class StyleComposer(OriginalComposer):
    trace_noise_name='initial_W_anchor_codes'

    def __init__(self,bank,config,seed=241):
        super().__init__(bank,config,seed);c=config
        assert c.optimize and not c.history and c.optimize_waypoint
        assert c.merge!='sum','W overrides for summed-modulation execution are not part of this test'
        assert c.style_radius>=0 and c.style_tether>=0 and c.style_samples>=64
        self.previous_styles=None;self.scales=[]
        self.style_dimensions=[actor.config.style_dim for actor in [bank.k,bank.w]]
        # Separate RNG preserves the candidate-code sequence used by Z runs.
        rng=torch.Generator(device=self.device).manual_seed(328)
        with torch.no_grad():
            u=torch.randn(c.style_samples,1,device=self.device,generator=rng).abs().clamp_min(1e-6)
            for actor in [bank.k,bank.w]:
                by_sign=[]
                if c.style_condition_on_sign:
                    for sign in [-1,1]:
                        w=mapped_style(actor,sign*u);std=w.std(0,unbiased=False)
                        floor=.05*std.square().mean().sqrt()+1e-6
                        by_sign.append(std.clamp_min(floor))
                else:
                    w=mapped_style(actor,torch.cat([-u,u]));std=w.std(0,unbiased=False)
                    floor=.05*std.square().mean().sqrt()+1e-6
                    by_sign=[std.clamp_min(floor)]*2
                self.scales.append(torch.stack(by_sign))

    def objective_w(self,state,bz,contexts,codes,styles,anchors,scales,previous=None):
        c=self.config;k,w=style_pair(self.bank,state,bz,codes,contexts,styles)
        h=c.agreement_horizon
        agreement=(self.coordinates(k[:,:h])-self.coordinates(w[:,:h])).square().mean((1,2))
        tether=.5*sum(((value-anchor)/scale).square().mean(1) for value,anchor,scale in zip(styles,anchors,scales))
        continuity=torch.zeros_like(tether) if previous is None else .5*sum(
            ((value-old)/scale).square().mean(1) for value,old,scale in zip(styles,previous,scales))
        code_prior=codes.square().mean(1);code_trust=(codes.abs()-.8).square().mean(1)
        loss=agreement+c.style_tether*tether+c.style_continuity*continuity+c.prior*code_prior+c.trust*code_trust
        return loss,k,w,{'agreement':agreement,'style_tether':tether,'style_continuity':continuity}

    def sample(self,state,contexts,base_z):
        c=self.config;n=len(state);m=c.starts;self.bank.reset_counts()
        sectors=[[k,w] for k in ([c.keepout_sign] if c.keepout_sign else [-1,1])
                 for w in ([c.waypoint_sign] if c.waypoint_sign else [-1,1])]
        signs=state.new_tensor(sectors).repeat(m//len(sectors),1)[None].expand(n,-1,-1).clone()
        if c.preserve_mode and self.previous is not None:signs=self.previous.sign()[:,None].expand(n,m,2).clone()
        codes=signs*(.4+.8*torch.rand(n,m,2,device=self.device,generator=self.rng))
        warm_index=None
        if self.previous is not None:
            matches=(signs==self.previous.sign()[:,None]).all(-1)
            warm_index=matches.to(torch.int64).argmax(1)
            codes[torch.arange(n,device=self.device),warm_index]=self.previous
        self.noise=codes.detach().clone();codes=codes.flatten(0,1)
        with torch.no_grad():
            anchors=tuple(mapped_style(actor,codes[:,j:j+1]) for j,actor in enumerate([self.bank.k,self.bank.w]))
            scales=tuple(self.scales[j][(codes[:,j]>0).long()] for j in [0,1])
        parameters=[]
        for j,(anchor,scale) in enumerate(zip(anchors,scales)):
            raw=torch.zeros_like(anchor)
            if self.previous_styles is not None and c.style_radius:
                ids=torch.arange(n,device=self.device)*m+warm_index
                relative=(self.previous_styles[j]-anchor[ids])/(c.style_radius*scale[ids])
                raw[ids]=torch.atanh(relative.clamp(-.999,.999))
            parameters.append(raw.requires_grad_(True))
        optimizer=torch.optim.Adam(parameters,lr=c.lr)
        s=state.repeat_interleave(m,0);bz=base_z.repeat_interleave(m,0)
        ctx=tuple(x.repeat_interleave(m,0) for x in contexts)
        previous=None if self.previous_styles is None else tuple(x.repeat_interleave(m,0) for x in self.previous_styles)
        best=state.new_full((n,m),float('inf'));best_styles=[a.detach().clone().reshape(n,m,-1) for a in anchors]
        for step in range(c.steps+1):
            with torch.set_grad_enabled(step<c.steps):
                styles=tuple(a+c.style_radius*scale*p.tanh() for a,scale,p in zip(anchors,scales,parameters))
                values=self.objective_w(s,bz,ctx,codes,styles,anchors,scales,previous)[0].reshape(n,m)
            with torch.no_grad():
                improved=values<best;best=torch.minimum(best,values)
                best_styles=[torch.where(improved[:,:,None],w.detach().reshape(n,m,-1),old)
                             for w,old in zip(styles,best_styles)]
            if step==c.steps:break
            gradients=torch.autograd.grad(values.sum(),parameters)
            assert all(torch.isfinite(g).all() for g in gradients)
            optimizer.zero_grad(set_to_none=True)
            for p,g in zip(parameters,gradients):p.grad=g
            optimizer.step()
        with torch.no_grad():
            choice=best.argmin(1);rows=torch.arange(n,device=self.device);ids=rows*m+choice
            selected=tuple(w[rows,choice] for w in best_styles);chosen_codes=codes[ids]
            chosen_anchors=tuple(a[ids] for a in anchors);chosen_scales=tuple(x[ids] for x in scales)
            loss,k,w,details=self.objective_w(state,base_z,contexts,chosen_codes,selected,chosen_anchors,chosen_scales,self.previous_styles)
            result=action_merge(k,w,self.bank.base(state,base_z),c)
            displacement=torch.stack([((w-a)/scale).abs().amax(1) for w,a,scale in zip(selected,chosen_anchors,chosen_scales)],1)
            assert (displacement<=c.style_radius+1e-4).all()
        self.previous=chosen_codes.detach().clone();self.previous_styles=tuple(w.detach().clone() for w in selected)
        self.last_plan={'optimizer_space':'private_adapter_W','optimizer_iterations':c.steps,'starts':m,
            'style_dimensions':self.style_dimensions,'agreement_horizon':c.agreement_horizon,
            'anchor_codes':chosen_codes.tolist(),'maximum_scaled_style_displacement':displacement.tolist(),
            'objective':loss.tolist(),'simulator_queries':0,**{key:value.tolist() for key,value in details.items()}}
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result.detach()
