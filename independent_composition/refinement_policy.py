"""Compose action-edit adapters, without routes, judges or simulator search.

The original base keeps its eight-control action stream. Each adapter edits
only the next control using the newly observed state and its own context.
"""
import torch
from .refinement_adapters import first_logits


class RefinementPolicy:
    trace_noise_name='fixed_operator_codes'

    def __init__(self,models,method='compose',sweeps=2,order='wk',base_period=8,seed=241):
        assert method in ['base','keepout','waypoint','compose','mean','residual']
        assert sweeps>=1 and order in ['wk','kw'] and 1<=base_period<=models[0].actor.base_config.horizon
        self.models=models;self.base=models[0].actor.base;self.method=method;self.sweeps=sweeps
        self.order=[1,0] if order=='wk' else [0,1];self.base_period=base_period
        self.device=next(self.base.parameters()).device;self.rng=torch.Generator(device=self.device).manual_seed(seed)
        self.noise=None;self.chunk=None;self.calls=0;self.last_plan={}
        for model in models:model.eval().requires_grad_(False)
        left={p.data_ptr() for p in models[0].parameters()};right={p.data_ptr() for p in models[1].parameters()}
        assert not left&right,'Learned skill parameters share storage'

    @torch.no_grad()
    def sample(self,state,contexts,base_z):
        if self.noise is None:self.noise=(.8*torch.randn(len(state),2,device=self.device,generator=self.rng)).clamp(-3,3)
        refresh=self.calls%self.base_period==0
        if refresh:self.chunk=self.base(state,base_z).detach()
        index=self.calls%self.base_period;candidate=self.chunk[:,index]
        self.last_plan={'simulator_queries':0,'base_control_period':self.base_period,'base_refreshed':refresh,
                        'base_plan_index':index,'neural_sweeps':0 if self.method=='base' else (self.sweeps if self.method=='compose' else 1),
                        'adapter_forward_calls':0 if self.method=='base' else (2*self.sweeps if self.method=='compose' else (1 if self.method in ['keepout','waypoint'] else 2))}
        self.calls+=1
        if self.method=='base':return candidate[:,None]
        reference=first_logits(self.models[0].actor,self.models[0].environment,state,base_z)
        def apply(j,value):return self.models[j](state,base_z,self.noise[:,j:j+1],contexts[j],value,reference)
        if self.method in ['keepout','waypoint']:value=apply(0 if self.method=='keepout' else 1,candidate)
        elif self.method in ['mean','residual']:
            k,w=apply(0,candidate),apply(1,candidate)
            value=.5*(k+w) if self.method=='mean' else (k+w-candidate).clamp(-1,1)
        else:
            value=candidate
            for _ in range(self.sweeps):
                for j in self.order:value=apply(j,value)
        if self.models[0].environment=='cube':assert torch.equal(value[:,3:],candidate[:,3:]),'Yaw/gripper were altered'
        return value[:,None]
