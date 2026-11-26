"""Independent conditional action-distribution adapters and score composition.

No environment or task-geometry imports. Each denoiser has independent weights;
the only reused model is the ORIGINAL frozen base action generator. Conditional
and context-dropped predictions from each own model support density-ratio-style
score composition, rather than averaging two final action trajectories.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class DiffusionConfig:
    state_dim: int
    context_dim: int
    action_dim: int
    horizon: int = 16
    hidden: int = 384
    time_dim: int = 64
    diffusion_steps: int = 100
    base_z_dim: int = 2
    score_residual: bool = False
    base_std: float = .2
    background_threshold: float = .1


class Denoiser(nn.Module):
    def __init__(self, config):
        super().__init__();self.config=c=config
        self.condition_dim=c.state_dim+c.context_dim+c.base_z_dim+c.horizon*c.action_dim
        self.register_buffer('condition_mean',torch.zeros(self.condition_dim))
        self.register_buffer('condition_std',torch.ones(self.condition_dim))
        self.condition=nn.Sequential(nn.Linear(self.condition_dim+1,c.hidden),nn.SiLU(),nn.Linear(c.hidden,c.hidden))
        self.input=nn.Linear(c.horizon*c.action_dim+c.time_dim,c.hidden)
        self.layers=nn.ModuleList([nn.Linear(c.hidden,c.hidden) for _ in range(4)])
        self.modulations=nn.ModuleList([nn.Linear(c.hidden,2*c.hidden) for _ in range(4)])
        self.norms=nn.ModuleList([nn.LayerNorm(c.hidden,elementwise_affine=False) for _ in range(4)])
        self.output=nn.Linear(c.hidden,c.horizon*c.action_dim)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        t=torch.linspace(0,1,c.diffusion_steps+1)
        a=torch.cos((t+.008)/1.008*math.pi/2).square();a=a/a[0]
        beta=(1-a[1:]/a[:-1]).clamp(.0001,.999)
        self.register_buffer('alpha_bar',(1-beta).cumprod(0))

    def condition_vector(self,state,context,base_z,base_actions):
        return torch.cat([state,context,base_z,base_actions.flatten(-2)],-1)

    @torch.no_grad()
    def fit_normalization(self,condition):
        self.condition_mean.copy_(condition.mean(0));self.condition_std.copy_(condition.std(0).clamp_min(.01))

    def forward(self,noisy,time,state,context,base_z,base_actions,present=None):
        c=self.config
        condition=self.condition_vector(state,context,base_z,base_actions)
        condition=((condition-self.condition_mean)/self.condition_std).clamp(-10,10)
        if present is None:present=torch.ones((len(state),1),device=state.device,dtype=state.dtype)
        left=c.state_dim;right=left+c.context_dim
        condition=torch.cat([condition[:,:left],condition[:,left:right]*present,condition[:,right:],present],-1)
        v=self.condition(condition)
        frequencies=torch.exp(torch.linspace(0,-math.log(10000),c.time_dim//2,device=time.device))
        angle=time[:,None].float()*frequencies[None]
        embedding=torch.cat([angle.sin(),angle.cos()],-1)
        h=self.input(torch.cat([noisy.flatten(-2),embedding],-1))
        for layer,norm,mod in zip(self.layers,self.norms,self.modulations):
            scale,shift=mod(v).chunk(2,-1)
            h=(h+torch.nn.functional.silu(norm(layer(h))*(1+.1*scale)+.1*shift))/math.sqrt(2)
        prediction=self.output(h).reshape_as(noisy)
        if c.score_residual:
            prediction=self.base_epsilon(noisy,time,base_actions)+present[:,:,None]*prediction
        return prediction

    def base_epsilon(self,noisy,time,base_actions):
        a=self.alpha_bar[time,None,None]
        variance=1-a+a*self.config.base_std**2
        return (1-a).sqrt()*(noisy-a.sqrt()*base_actions)/variance


def load_denoiser(path,device='cuda'):
    saved=torch.load(path,map_location=device,weights_only=False)
    model=Denoiser(DiffusionConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['ema'])
    return model.eval().requires_grad_(False),saved


class ScoreComposer:
    """No simulator handle. Samples actions using frozen per-skill score fields.

Composition of noisy scores is a heuristic approximation to a product or
density ratio of action distributions, not a guarantee of exact PoE sampling.
"""
    def __init__(self,models,base,method='ratio',guidance=1.,steps=20,seed=41,warm_time=99,execute=8,fixed_noise=True,cold_time=99):
        if method not in ['keepout','waypoint','mean','sum','ratio']:raise ValueError(method)
        self.models=models;self.base=base;self.method=method;self.guidance=guidance
        self.steps=steps;self.warm_time=warm_time;self.execute=execute;self.previous=None
        self.fixed_noise=fixed_noise;self.noise=None
        if not 0 <= cold_time < models[0].config.diffusion_steps:raise ValueError('Invalid initial noise level')
        self.cold_time=cold_time
        self.device=next(base.parameters()).device
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        for field in ['action_dim','horizon','diffusion_steps','base_z_dim']:
            assert getattr(models[0].config,field)==getattr(models[1].config,field),f'Incompatible {field}'
        assert torch.equal(models[0].alpha_bar,models[1].alpha_bar),'Different diffusion schedules'
        if method=='ratio' and all(m.config.score_residual for m in models):
            assert models[0].config.base_std==models[1].config.base_std,'Different reference priors'
        for module in [base,*models]:module.eval().requires_grad_(False)
        a={p.data_ptr() for p in models[0].parameters()}
        assert not a & {p.data_ptr() for p in models[1].parameters()},'Shared denoiser weights'

    @torch.no_grad()
    def sample(self,state,contexts,base_z):
        base_actions=self.base(state,base_z)
        if self.noise is None or not self.fixed_noise:
            self.noise=torch.randn(base_actions.shape,device=self.device,generator=self.rng)
        noise=self.noise
        start=self.models[0].config.diffusion_steps-1
        if self.previous is not None and self.warm_time<start:
            start=self.warm_time
            shifted=torch.cat([self.previous[:,self.execute:],self.previous[:,-1:].expand(-1,self.execute,-1)],1)
            a=self.models[0].alpha_bar[start]
            x=a.sqrt()*shifted+(1-a).sqrt()*noise
        elif self.cold_time<start:
            # Start from the original base distribution at a finite noise level.
            # This also avoids amplifying small score errors by 1/sqrt(alpha)
            # at the nearly pure-noise endpoint of the cosine schedule.
            start=self.cold_time
            a=self.models[0].alpha_bar[start]
            x=a.sqrt()*base_actions+(1-a).sqrt()*noise
        else:x=noise
        times=torch.linspace(start,0,self.steps,device=self.device).round().long().unique_consecutive()
        for index,time in enumerate(times):
            tt=time.expand(len(state))
            scores={};unconditional={}
            active=[0] if self.method=='keepout' else [1] if self.method=='waypoint' else [0,1]
            for j in active:
                model,ctx=self.models[j],contexts[j]
                if self.method=='ratio' and model.config.score_residual:
                    scores[j]=model(x,tt,state,ctx,base_z,base_actions)
                    unconditional[j]=model.base_epsilon(x,tt,base_actions)
                elif self.method=='ratio':
                    n=len(state);present=torch.cat([state.new_ones(n,1),state.new_zeros(n,1)],0)
                    value=model(torch.cat([x,x]),torch.cat([tt,tt]),torch.cat([state,state]),
                                torch.cat([ctx,ctx]),torch.cat([base_z,base_z]),torch.cat([base_actions,base_actions]),present)
                    scores[j]=value[:n];unconditional[j]=value[n:]
                else:scores[j]=model(x,tt,state,ctx,base_z,base_actions)
            if self.method=='keepout':epsilon=scores[0]
            elif self.method=='waypoint':epsilon=scores[1]
            elif self.method=='mean':epsilon=.5*(scores[0]+scores[1])
            elif self.method=='sum':epsilon=scores[0]+scores[1]
            else:
                neutral=.5*(unconditional[0]+unconditional[1])
                epsilon=neutral+self.guidance*((scores[0]-unconditional[0])+(scores[1]-unconditional[1]))
            ab=self.models[0].alpha_bar[time]
            clean=((x-(1-ab).sqrt()*epsilon)/ab.sqrt()).clamp(-1,1)
            if index+1==len(times):x=clean
            else:
                next_ab=self.models[0].alpha_bar[times[index+1]]
                x=next_ab.sqrt()*clean+(1-next_ab).sqrt()*epsilon
        self.previous=x.detach()
        return x
