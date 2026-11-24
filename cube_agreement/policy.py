"""Frozen neural trajectory agreement with shared learned manipulation control.

No hand-coded path teacher, collision projection or native task search executes
at inference. A generic path cursor selects a reference; a learned steering head
and the preserved neural grasp/release backbone generate robot controls.
"""
import math,time,json
from pathlib import Path
import numpy as np
import torch
from cube_addons.evaluate import load_adapter
from cube_addons.geometry import Observer,axis_frame
from cube_addons.contacts import force_holding
from .manifolds import geometry_cases,world_path,FREE
from .neural_experiment import load
from .agreement import solve,fingerprint
from .steering import Steering


class Bank:
    def __init__(self,keepout,waypoint,steering,backbone='cube_addon_artifacts/final/waypoint_s42/adapter.pt',device='cuda'):
        self.device=device;self.models={s:load(p,device) for s,p in [('keepout',keepout),('waypoint',waypoint)]}
        self.backbone,self.encoder,_=load_adapter(backbone,'cube/artifacts/final/base_policy.pt',device)
        self.backbone.eval().requires_grad_(False);self.encoder.eval().requires_grad_(False)
        ck=torch.load(steering,map_location=device,weights_only=False);self.steering=Steering().to(device);self.steering.load_state_dict(ck['state']);self.steering.eval().requires_grad_(False)
        self.modules={**self.models,'shared_backbone':self.backbone,'backbone_encoder':self.encoder,'learned_steering':self.steering}
        self.before={s:fingerprint(m) for s,m in self.modules.items()}
    def assert_frozen(self):
        for s,m in self.modules.items():
            assert fingerprint(m)==self.before[s],s
            assert all(not p.requires_grad and p.grad is None for p in m.parameters()),s


class NeuralPolicy:
    def __init__(self,bank,case,method='agreement',execute_family='waypoint',steps=120,starts=12,seed=42,objective='path',tolerance=.010,threshold=.15,backbone_execute=1):
        self.bank=bank;self.case=case;self.method=method;self.execute_family=execute_family;self.steps=steps;self.starts=starts;self.seed=seed;self.objective=objective
        self.path=None;self.index=0;self.done=False;self.started=False;self.previous_cube=None;self.tolerance=tolerance;self.threshold=threshold
        self.backbone_execute=backbone_execute;self.base_chunk=None;self.base_index=0
        z=np.asarray(case['base_z']);p=.5*(1+math.erf(float(z[1])/math.sqrt(2)));self.speed=.65+2*p*.2 if p<.5 else .85+2*(p-.5)*.15
        amp=(.035+.04*math.erf(abs(float(z[0]))/math.sqrt(2)))*(1 if z[0]>=0 else -1)
        start=case['initial_cube'];goal=case['goal'];_,_,_,perp,_=axis_frame(start,goal,start)
        ref=np.r_[(np.asarray(start[:2])+goal[:2])/2+(amp+np.sign(amp)*.060)*perp,.20,.022]
        self.reference=Observer(start,goal,ref,'waypoint',force_holding);self.codes=np.zeros(FREE+1+FREE,np.float32)

    def plan(self,state,ctx,bz,runtime):
        started=time.perf_counter();device=self.bank.device;planning={};info=runtime['info'];cube=np.asarray(info['privileged/block_0_pos']);held=force_holding(info)
        if self.path is None:
            geometry=geometry_cases([self.case],device)
            if self.method=='agreement':
                codes,k,w,value=solve(self.bank.models,geometry,self.starts,self.steps,self.seed,objective=self.objective)
                self.codes=codes[0].detach().cpu().numpy();path=k if self.execute_family=='keepout' else w
                planning={'iterations':self.steps,'backward_steps':self.steps,'path_disagreement':float(((k-w)/.1).square().mean()),'agreement_loss':float(value[0])}
            else:
                family=self.method.removeprefix('single_');rng=torch.Generator(device=device).manual_seed(self.seed)
                z=.5*torch.randn((1,FREE+int(family=='keepout')),device=device,generator=rng)
                if family=='keepout':z[:,FREE]=2*torch.pi*torch.rand((1,),device=device,generator=rng)-torch.pi
                with torch.inference_mode():path=self.bank.models[family](z,geometry)
                if family=='keepout':self.codes[:FREE+1]=z[0].cpu().numpy()
                else:self.codes[FREE+1:]=z[0].cpu().numpy()
            self.path=world_path(path[0].detach().cpu().numpy(),self.case)
        if self.previous_cube is not None:self.reference.update(self.previous_cube,cube,info,np.stack([self.previous_cube,cube]))
        self.reference.previous_holding=held;self.reference.has_lifted|=bool(held and cube[2]>.10)
        self.started|=bool(held and cube[2]>self.threshold)
        if self.index==len(self.path)-1 and np.linalg.norm(cube-self.path[-1])<self.tolerance:self.done=True
        self.reference.visited=self.done
        with torch.inference_mode():
            if self.started and held and not self.done:
                self.base_chunk=None;self.base_index=0
                while self.index<len(self.path)-1 and np.linalg.norm(cube-self.path[self.index])<self.tolerance:self.index+=1
                inp=torch.as_tensor(self.speed*(self.path[self.index]-cube)*10,device=device,dtype=torch.float32)[None]
                action=self.bank.steering(inp)[0].cpu().numpy();phase='learned_steering';calls={'learned_steering':{'batched_calls':1,'candidate_equivalents':1}}
            else:
                calls={}
                if self.base_chunk is None or self.base_index>=self.backbone_execute:
                    s=torch.as_tensor(state,device=device)[None];b=torch.as_tensor(bz,device=device)[None];c=torch.as_tensor(self.reference.features(info,runtime['time']),device=device)[None]
                    self.base_chunk=self.bank.backbone(s,b,torch.tensor([[.8]],device=device),c)[0].cpu().numpy();self.base_index=0
                    calls={'shared_backbone':{'batched_calls':1,'candidate_equivalents':1}}
                action=self.base_chunk[self.base_index].copy();self.base_index+=1;phase='neural_grasp_release'
        self.previous_cube=cube.copy()
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        meta={'seconds':time.perf_counter()-started,'iterations':0,'backward_steps':0,'calls':calls,'disagreement':None,'path_index':self.index,'path_done':self.done,'control_phase':phase,**planning}
        return np.repeat(action[None],16,axis=0),self.codes.copy(),meta

    def close(self):self.bank.assert_frozen()
