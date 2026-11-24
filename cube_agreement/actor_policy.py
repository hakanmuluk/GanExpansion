"""Frozen learned route manifolds driving the original StyleGAN action actor.

All executed controls are produced by the action network. The path cursor is
generic; no constraint predicate, teacher, geometric projection, or dynamics
search supplies controls or selects latent candidates during composition.
"""
import math,time
import numpy as np
import torch
from cube_addons.geometry import Observer,axis_frame
from cube_addons.contacts import force_holding
from .manifolds import geometry_cases,world_path,FREE
from .neural_experiment import load
from .agreement import solve,fingerprint
from .action_model import load as load_actor,path_context,world_path_tensor
from .action_agreement import refine


class Bank:
    def __init__(self,keepout,waypoint,actor,device='cuda'):
        self.device=device;self.models={s:load(p,device) for s,p in [('keepout',keepout),('waypoint',waypoint)]}
        self.actor=load_actor(actor,device=device)
        self.modules={**self.models,'StyleGAN_actor':self.actor}
        self.before={s:fingerprint(m) for s,m in self.modules.items()}
    def assert_frozen(self):
        for s,m in self.modules.items():
            assert fingerprint(m)==self.before[s],s
            assert all(not p.requires_grad and p.grad is None for p in m.parameters()),s


class ActorPolicy:
    def __init__(self,bank,case,method='agreement',execute_family='waypoint',steps=120,starts=12,seed=42,execute=4,action_steps=20,path_weight=1.,continuity_weight=.001,action_interval=1,action_starts=0):
        self.bank=bank;self.case=case;self.method=method;self.execute_family=execute_family;self.steps=steps;self.starts=starts;self.seed=seed;self.execute=execute
        self.path=None;self.index=0;self.done=False;self.previous_cube=None;self.chunk=None;self.chunk_index=0
        self.action_steps=action_steps;self.path_weight=path_weight;self.continuity_weight=continuity_weight;self.replans=0
        self.action_interval=action_interval;self.action_starts=action_starts or starts;self.next_action_replan=0
        z=np.asarray(case['base_z']);amp=(.035+.04*math.erf(abs(float(z[0]))/math.sqrt(2)))*(1 if z[0]>=0 else -1)
        start=case['initial_cube'];goal=case['goal'];_,_,_,perp,_=axis_frame(start,goal,start)
        ref=np.r_[(np.asarray(start[:2])+goal[:2])/2+(amp+np.sign(amp)*.060)*perp,.20,.022]
        self.reference=Observer(start,goal,ref,'waypoint',force_holding);self.codes=np.zeros(2*FREE+1,np.float32)

    def plan(self,state,ctx,bz,runtime):
        started=time.perf_counter();device=self.bank.device;planning={};info=runtime['info'];cube=np.asarray(info['privileged/block_0_pos']);held=force_holding(info)
        if self.path is None:
            self.geometry=geometry_cases([self.case],device)
            if self.method in ['agreement','multiscale']:
                codes,k,w,value=solve(self.bank.models,self.geometry,self.starts,self.steps,self.seed)
                self.codes=codes[0].detach().cpu().numpy();path=k if self.execute_family=='keepout' else w
                planning={'iterations':self.steps,'backward_steps':self.steps,'path_disagreement':float(((k-w)/.1).square().mean()),'agreement_loss':float(value[0])}
            elif self.method=='local_action':
                ref=torch.as_tensor(self.reference.features(info,runtime['time']),device=device)[None]
                codes,k,w,metrics=refine(self.bank,self.geometry,self.case,torch.as_tensor(state,device=device)[None],ref,torch.as_tensor(cube,device=device,dtype=torch.float32)[None],0,starts=self.starts,steps=self.steps,seed=self.seed)
                self.codes=codes[0].detach().cpu().numpy();path=k if self.execute_family=='keepout' else w
                planning={'iterations':self.steps,'backward_steps':self.steps,**metrics}
            else:
                family=self.method.removeprefix('single_');rng=torch.Generator(device=device).manual_seed(self.seed)
                z=.5*torch.randn((1,FREE+int(family=='keepout')),device=device,generator=rng)
                if family=='keepout':z[:,FREE]=2*torch.pi*torch.rand((1,),device=device,generator=rng)-torch.pi
                with torch.inference_mode():path=self.bank.models[family](z,self.geometry)
                if family=='keepout':self.codes[:FREE+1]=z[0].cpu().numpy()
                else:self.codes[FREE+1:]=z[0].cpu().numpy()
            self.path=world_path(path[0].detach().cpu().numpy(),self.case)
            self.path_tensor=torch.as_tensor(self.path,device=device,dtype=torch.float32)[None]
        if self.previous_cube is not None:self.reference.update(self.previous_cube,cube,info,np.stack([self.previous_cube,cube]))
        self.reference.previous_holding=held;self.reference.has_lifted|=bool(held and cube[2]>.10)
        while self.index<len(self.path)-1 and np.linalg.norm(cube-self.path[self.index])<.010:self.index+=1
        if self.index==len(self.path)-1 and np.linalg.norm(cube-self.path[-1])<.010:self.done=True
        self.reference.visited=self.done
        calls={}
        if self.method in ['local_action','multiscale'] and self.chunk_index>=self.execute and held and not self.done and runtime['time']>=self.next_action_replan:
            ref=torch.as_tensor(self.reference.features(info,runtime['time']),device=device)[None]
            previous=torch.as_tensor(self.codes,device=device)[None]
            codes,k,w,metrics=refine(self.bank,self.geometry,self.case,torch.as_tensor(state,device=device)[None],ref,torch.as_tensor(cube,device=device,dtype=torch.float32)[None],self.index,previous,self.action_starts,self.action_steps,self.seed+1009*self.replans,self.path_weight if self.method=='multiscale' else 0.,self.continuity_weight)
            self.codes=codes[0].detach().cpu().numpy();self.path=world_path((k if self.execute_family=='keepout' else w)[0].detach().cpu().numpy(),self.case);self.path_tensor=torch.as_tensor(self.path,device=device,dtype=torch.float32)[None];self.replans+=1
            self.next_action_replan=runtime['time']+self.action_interval
            planning={'iterations':self.action_steps,'backward_steps':self.action_steps,'disagreement':metrics['action_disagreement'],**metrics}
        with torch.inference_mode():
            if self.chunk is None or self.chunk_index>=self.execute:
                s=torch.as_tensor(state,device=device)[None];b=torch.as_tensor(bz,device=device)[None]
                ref=torch.as_tensor(self.reference.features(info,runtime['time']),device=device)[None]
                c=path_context(ref,torch.as_tensor(cube,device=device,dtype=torch.float32)[None],self.path_tensor,self.index)
                self.chunk=self.bank.actor(s,b,torch.tensor([[.8]],device=device),c)[0].cpu().numpy();self.chunk_index=0
                calls={'StyleGAN_actor':{'batched_calls':1,'candidate_equivalents':1}}
            action=self.chunk[self.chunk_index].copy();self.chunk_index+=1
        self.previous_cube=cube.copy()
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        return np.repeat(action[None],16,axis=0),self.codes.copy(),{'seconds':time.perf_counter()-started,'iterations':0,'backward_steps':0,'calls':calls,'disagreement':None,'path_index':self.index,'path_done':self.done,'control_phase':'StyleGAN_actor',**planning}

    def close(self):self.bank.assert_frozen()
