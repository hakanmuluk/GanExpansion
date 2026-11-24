"""Frozen two-scalar agreement search; native task verification is separate."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class SearchConfig:
    method:str='adam'
    family:str='keepout'
    objective:str='agreement'
    regularizer:str='gaussian'
    prior_weight:float=.01
    steps:int=15
    starts:int=4
    candidates:int=64
    lr:float=.15
    initialization:str='gaussian'
    temporal:str='warm'
    tail_weight:float=.1
    agreement_horizon:int=16
    agreement_mode:str='continuous'
    grid_size:int=16
    grid_bound:float=3.
    fixed_codes:tuple|None=None
    lookahead:int=16
    workers:int=6
    task_weight:float=1.
    agreement_weight:float=0.
    phase_only:bool=False
    action_sigma:float=.15
    action_deviation_weight:float=.01
    knot_steps:int=8
    native_terminal_bonus:float=50.
    common_margin:float=.005


class LatentSearch:
    def __init__(self,bank,config,priors=None,seed=41):
        self.bank=bank;self.config=config;self.priors=priors;self.device=next(bank.parameters()).device
        self.rng=torch.Generator(device=self.device).manual_seed(seed);self.previous=None;self.tail=None
        self.bank.common_margin=config.common_margin
        self.encoder_hint=None

    def prior(self,z):
        if self.config.regularizer=='none':return z.new_zeros(len(z))
        if self.config.regularizer=='gaussian':return .5*z.square().mean(-1)
        if self.priors is None:raise ValueError('Empirical prior requires individual physical-training encodings')
        values=[]
        for j,skill in enumerate(['keepout','waypoint']):
            p=self.priors[skill];mu=z.new_tensor(p['means']);sd=z.new_tensor(p['stds']);weights=z.new_tensor(p['weights'])
            logp=-.5*((z[:,j,None]-mu)/sd).square()-sd.log()-.9189385332046727+weights.log()
            values.append(-torch.logsumexp(logp,-1))
        return torch.stack(values,-1).mean(-1)

    def initial(self,n,warm=None):
        z=torch.randn((n,2),device=self.device,generator=self.rng);cfg=self.config
        if cfg.fixed_codes is not None:z[:]=z.new_tensor(cfg.fixed_codes)
        elif cfg.initialization=='structured':
            magnitudes=z.new_tensor([[.4,.4],[1.5,1.5],[.4,1.5],[1.5,.4]]);sectors=z.new_tensor([[-1,-1],[-1,1],[1,-1],[1,1]])
            grid=(magnitudes[:,None]*sectors[None]).reshape(16,2);z[:min(n,16)]=grid[:min(n,16)]
        elif cfg.initialization in ['encoder','encoder_prefix']:
            if self.priors is None:raise ValueError('Encoder-code bank is required')
            for j,skill in enumerate(['keepout','waypoint']):
                codes=z.new_tensor(self.priors[skill]['initial_codes']);z[:,j]=codes[torch.randint(len(codes),(n,),device=self.device,generator=self.rng)]
        if warm is not None:
            warm=torch.as_tensor(warm,device=self.device,dtype=z.dtype)
            z[0]=warm
            if n>4:z[1:4]=warm[None]+.2*torch.randn((3,2),device=self.device,generator=self.rng)
        return z

    def objective(self,z,state,ctx,bz,return_chunk=False):
        n=len(z);s=state.expand(n,-1);b=bz.expand(n,-1);c=tuple(v.expand(n,-1) for v in ctx);a,w=self.bank.pair(s,b,z,c)
        h=self.config.agreement_horizon;difference=(a[:,:h]-w[:,:h]).square()
        if self.config.agreement_mode=='binary_aware':
            difference=torch.cat([difference[:,:,:4],((5*a[:,:h,4]).tanh()-(5*w[:,:h,4]).tanh()).square()[:,:,None]],-1)
        disagreement=difference.mean((1,2));prior=self.prior(z);loss=disagreement+self.config.prior_weight*prior
        chunk=None
        if return_chunk or (self.tail is not None and self.config.temporal=='tail'):
            chunk=self.bank.execute(s,b,z,c,self.config.family,(a,w))
        tail=z.new_zeros(n)
        if self.tail is not None and self.config.temporal=='tail':
            tail=(chunk[:,:8]-self.tail).square().mean((1,2));loss=loss+self.config.tail_weight*tail
        return loss,{'disagreement':disagreement,'prior':prior,'tail_loss':tail,'executable_grip_disagreement':((a[:,:h,4]>=0)!=(w[:,:h,4]>=0)).float().mean(1),'chunk':chunk}

    def gradient(self,initial,state,ctx,bz,steps):
        sign=torch.where(initial>=0,1.,-1.);parameter=torch.nn.Parameter(torch.log(torch.expm1(initial.abs().clamp(.001,12))))
        optimizer=torch.optim.Adam([parameter],lr=self.config.lr);best=None;minimum=None
        for step in range(steps+1):
            z=sign*F.softplus(parameter);value,_=self.objective(z,state,ctx,bz)
            with torch.no_grad():
                i=value.argmin();candidate=z[i].clone();score=value[i].clone()
                if best is None:best=candidate;minimum=score
                else:best=torch.where(score<minimum,candidate,best);minimum=torch.minimum(minimum,score)
            if step==steps:break
            optimizer.zero_grad(set_to_none=True);grad,=torch.autograd.grad(value.sum(),parameter)
            if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite latent gradient')
            parameter.grad=grad*(20/grad.norm(dim=-1,keepdim=True).clamp_min(20));optimizer.step()
        return best.detach()

    def plan(self,state,ctx,bz,runtime=None):
        cfg=self.config
        encoder_seconds=0.;self.encoder_hint=None
        if cfg.initialization=='encoder_prefix' and runtime is not None and runtime.get('history') is not None:
            history=runtime['history'];length=len(history['actions'])
            if length>=8:
                started_encoder=time.perf_counter()
                with torch.inference_mode():
                    hs=torch.as_tensor(history['states'],device=self.device)[None];ha=torch.as_tensor(history['actions'],device=self.device)[None]
                    hc=torch.as_tensor(history['contexts'],device=self.device)[None];hb=torch.as_tensor(bz,device=self.device)[None];hl=torch.tensor([length],device=self.device)
                    hints=[encoder(hs,ha,hc[:,:,j],hb,hl)[0][0,0] for j,encoder in enumerate([self.bank.ek,self.bank.ew])]
                    self.encoder_hint=torch.stack(hints).detach()
                if self.device.type=='cuda':torch.cuda.synchronize()
                encoder_seconds=time.perf_counter()-started_encoder
        if cfg.objective!='agreement':
            from .native_search import NativeSearch
            if not hasattr(self,'native'):self.native=NativeSearch(self.bank,cfg,self.priors,self.rng)
            chunk,z,meta=self.native.plan(state,ctx,bz,runtime,self)
            meta.update(seconds=meta['seconds']+encoder_seconds,encoder_seconds=encoder_seconds,encoder_calls=2 if self.encoder_hint is not None else 0,
                encoder_hint=self.encoder_hint.cpu().tolist() if self.encoder_hint is not None else None)
            return chunk,z,meta
        if self.device.type=='cuda':torch.cuda.synchronize()
        started=time.perf_counter();self.bank.reset_counts();backward=iterations=0
        s=torch.as_tensor(state,device=self.device)[None];c=tuple(torch.as_tensor(v,device=self.device)[None] for v in ctx);b=torch.as_tensor(bz,device=self.device)[None]
        warm=self.previous if cfg.temporal in ['warm','tail'] else None
        if self.encoder_hint is not None:warm=self.encoder_hint
        if cfg.temporal=='fixed' and self.previous is not None:z=self.previous
        elif cfg.method=='fixed':z=self.initial(1,warm)[0]
        elif cfg.method=='grid':
            if cfg.temporal=='tail' or cfg.agreement_mode!='continuous':raise ValueError('Factorized grid is defined for continuous agreement without trajectory-tail cost')
            if cfg.grid_size%2:raise ValueError('Use an even grid to avoid the untrained zero-sign branch')
            values=torch.linspace(-cfg.grid_bound,cfg.grid_bound,cfg.grid_size,device=self.device);n=len(values);codes=values[:,None].expand(-1,2)
            with torch.no_grad():
                a,w=self.bank.pair(s.expand(n,-1),b.expand(n,-1),codes,tuple(v.expand(n,-1) for v in c));a=a[:,:cfg.agreement_horizon].flatten(1);w=w[:,:cfg.agreement_horizon].flatten(1)
                mse=(a.square().mean(-1)[:,None]+w.square().mean(-1)[None]-2*a@w.T/a.shape[-1]).clamp_min(0)
                pairs=torch.cartesian_prod(values,values);score=mse.flatten()+cfg.prior_weight*self.prior(pairs);z=pairs[score.argmin()]
            iterations=1
        elif cfg.method=='adam':
            initial=self.initial(max(4,cfg.starts),warm);sign=initial.new_tensor([[-1,-1],[-1,1],[1,-1],[1,1]]);initial[:4]=initial[:4].abs().clamp_min(.05)*sign
            if warm is not None:
                branch=int((warm[0]>=0).long()*2+(warm[1]>=0).long());initial[branch]=warm
            z=self.gradient(initial,s,c,b,cfg.steps);backward=iterations=cfg.steps
        elif cfg.method in ['random','cem','hybrid']:
            n=max(4,cfg.candidates);candidates=self.initial(n,warm);best=None;minimum=None
            sectors=candidates.new_tensor([[-1,-1],[-1,1],[1,-1],[1,1]]);mu=torch.zeros((4,2),device=self.device);sd=torch.ones_like(mu)
            if cfg.method=='cem':
                candidates=candidates.abs()*sectors[torch.arange(n,device=self.device)%4]
                if warm is not None:candidates[0]=warm
            for step in range(cfg.steps if cfg.method=='cem' else 1):
                if step and cfg.method=='cem':
                    per=(n+3)//4;u=mu[:,None]+sd[:,None]*torch.randn((4,per,2),device=self.device,generator=self.rng)
                    candidates=(sectors[:,None]*F.softplus(u)).flatten(0,1)[:n];candidates[0]=best
                with torch.no_grad():scores,_=self.objective(candidates,s,c,b)
                i=scores.argmin()
                if best is None or bool(scores[i]<minimum):best=candidates[i].clone();minimum=scores[i].clone()
                if cfg.method=='cem':
                    for j,sector in enumerate(sectors):
                        mask=(torch.sign(candidates)==sector).all(-1);ids=mask.nonzero().flatten()
                        if not len(ids):continue
                        k=max(1,len(ids)//4);elite=candidates[ids[scores[ids].topk(k,largest=False).indices]].abs().clamp_min(.001);u=torch.log(torch.expm1(elite))
                        mu[j]=u.mean(0);sd[j]=u.std(0,unbiased=False).clamp_min(.15)
                iterations+=1
            z=best
            if cfg.method=='hybrid':z=self.gradient(z[None],s,c,b,cfg.steps);backward=cfg.steps;iterations+=cfg.steps
        else:raise ValueError(cfg.method)
        with torch.no_grad():value,details=self.objective(z[None],s,c,b,return_chunk=True)
        chunk=details.pop('chunk')[0];self.previous=z.detach();self.tail=chunk[8:].detach()
        if self.device.type=='cuda':torch.cuda.synchronize()
        return chunk.cpu().numpy(),z.cpu().numpy(),{'seconds':time.perf_counter()-started+encoder_seconds,'encoder_seconds':encoder_seconds,'encoder_calls':2 if self.encoder_hint is not None else 0,
            'encoder_hint':self.encoder_hint.cpu().tolist() if self.encoder_hint is not None else None,'loss':float(value[0]),
            **{k:float(v[0]) for k,v in details.items()},'iterations':iterations,'backward_steps':backward,'calls':{k:v.copy() for k,v in self.bank.calls.items()}}

    def close(self):
        if hasattr(self,'native'):self.native.close()
