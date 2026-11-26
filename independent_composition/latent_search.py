"""Generic tensor-only re-encoding composition of frozen independent actors.

Inputs contain the CURRENT observation and each skill's observed context only.
There is no environment handle, dynamics model, geometry cost, teacher, witness,
future observation, or learned shared composer in this module.
"""
from dataclasses import dataclass
import time
import torch
from torch.nn import functional as F


@dataclass
class LatentConfig:
    family: str = "sum"
    cycle: str = "none"  # none, self, executed
    cycle_weight: float = 1.
    reconstruction_weight: float = 0.
    prior_weight: float = .01
    continuity_weight: float = 0.
    cycle_metric: str = "mae"
    steps: int = 15
    starts: int = 4
    lr: float = .15
    warm: bool = True


class LatentComposer:
    def __init__(self, bank, encoders, config, seed=41):
        if config.family not in ["sum", "keepout", "waypoint", "average"]:
            raise ValueError("Only neural execution families are allowed")
        if config.cycle not in ["none", "self", "executed"]:
            raise ValueError(config.cycle)
        if config.cycle_metric not in ["mae", "nll"]:
            raise ValueError(config.cycle_metric)
        if config.starts < 4 or config.starts % 4:
            raise ValueError("Enumerate all four hard-sign sectors equally")
        if config.cycle != "none" and (encoders is None or len(encoders) != 2):
            raise ValueError("Two independent validated window encoders required")
        self.bank=bank;self.encoders=encoders;self.config=config
        self.device=next(bank.parameters()).device
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        self.previous=None
        for module in [bank, *(encoders or [])]:module.eval().requires_grad_(False)
        if encoders:
            first={p.data_ptr() for p in encoders[0].parameters()}
            assert not first & {p.data_ptr() for p in encoders[1].parameters()}, "Shared encoder storage"

    def objective(self, states, base_z, codes, contexts, previous=None):
        c=self.config;k,w=self.bank.pair(states,base_z,codes,contexts)
        executed=self.bank.execute(states,base_z,codes,contexts,c.family,(k,w))
        disagreement=(k-w).square().mean((-1,-2))
        prior=.5*codes.square().mean(-1)
        cycle=torch.zeros_like(prior);reconstruction=torch.zeros_like(prior)
        if c.cycle != "none":
            for j,(encoder,actor,own) in enumerate(zip(self.encoders,[self.bank.k,self.bank.w],[k,w])):
                a=own if c.cycle=="self" else executed
                mu,lv=encoder(states,contexts[j],base_z,a)
                error=mu-codes[:,j:j+1]
                if c.cycle_metric=="mae":penalty=error.abs().mean(-1)
                else:penalty=.5*(error.square()*(-lv).exp()+lv).mean(-1)
                cycle=cycle+.5*penalty
                if c.reconstruction_weight:
                    decoded=actor(states,base_z,mu,contexts[j])
                    reconstruction=reconstruction+.5*(decoded-a).square().mean((-1,-2))
        continuity=torch.zeros_like(prior) if previous is None else (codes-previous).square().mean(-1)
        total=disagreement+c.prior_weight*prior+c.cycle_weight*cycle+c.reconstruction_weight*reconstruction+c.continuity_weight*continuity
        return total,{"disagreement":disagreement,"prior":prior,"cycle":cycle,
                      "reconstruction":reconstruction,"continuity":continuity},executed

    def optimize(self, states, base_z, contexts):
        c=self.config;b=len(states);n=c.starts;device=self.device
        if device.type=="cuda":torch.cuda.synchronize()
        started=time.perf_counter();self.bank.reset_counts()
        initial=.4+torch.rand((b,n,2),device=device,generator=self.rng)*1.1
        signs=states.new_tensor([[-1,-1],[-1,1],[1,-1],[1,1]]).repeat(n//4,1)
        initial=initial*signs[None]
        previous=self.previous if c.warm else None
        if previous is not None:
            sector=(previous[:,0]>=0).long()*2+(previous[:,1]>=0).long()
            initial[torch.arange(b,device=device),sector]=previous
        signs=torch.where(initial>=0,1.,-1.)
        parameter=torch.nn.Parameter(torch.log(torch.expm1(initial.abs().clamp(.02,3))))
        optimizer=torch.optim.Adam([parameter],lr=c.lr)
        s=states.repeat_interleave(n,0);bz=base_z.repeat_interleave(n,0)
        ctx=tuple(x.repeat_interleave(n,0) for x in contexts)
        prev=None if previous is None else previous.repeat_interleave(n,0)
        best=None;best_value=states.new_full((b,),float('inf'))
        for step in range(c.steps+1):
            z=signs*F.softplus(parameter).clamp_max(3.)
            values,_,_=self.objective(s,bz,z.flatten(0,1),ctx,prev)
            values=values.reshape(b,n)
            with torch.no_grad():
                value,index=values.min(-1)
                candidate=z[torch.arange(b,device=device),index].detach()
                if best is None:best=candidate.clone()
                update=value<best_value
                best=torch.where(update[:,None],candidate,best);best_value=torch.minimum(best_value,value)
            if step==c.steps:break
            gradient,=torch.autograd.grad(values.sum(),parameter)
            if not torch.isfinite(gradient).all():raise FloatingPointError("Non-finite code gradient")
            optimizer.zero_grad(set_to_none=True)
            parameter.grad=gradient*(20/gradient.norm(dim=-1,keepdim=True).clamp_min(20))
            optimizer.step()
        with torch.no_grad():value,details,action=self.objective(states,base_z,best,contexts,previous)
        self.previous=best.detach()
        if device.type=="cuda":torch.cuda.synchronize()
        assert all(p.grad is None and not p.requires_grad for p in self.bank.parameters())
        return action.detach(),best.detach(),{"loss":value.detach(),**{k:v.detach() for k,v in details.items()},
             "seconds":time.perf_counter()-started,"iterations":c.steps,"backward_steps":c.steps,
             "calls":{k:v.copy() for k,v in self.bank.calls.items()},"simulator_queries":0}
