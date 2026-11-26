"""Composition/inversion with ORIGINAL model inputs, weights and forwards.

Only output algebra and the original one-dimensional skill codes are changed.
History contains actually observed states/contexts and issued controls only.
No new encoder, judge, learned module, future state or simulator is used.
"""
from dataclasses import dataclass
import torch


@dataclass
class CompositionConfig:
    merge:str='residual'
    axes:str='xyz'
    alpha:float=.5
    k_code:float=.8
    w_code:float=.8
    optimize:bool=False
    optimize_waypoint:bool=True
    keepout_sign:int=0
    waypoint_sign:int=1
    preserve_mode:bool=False
    steps:int=20
    starts:int=4
    lr:float=.12
    prior:float=.005
    trust:float=.02
    continuity:float=.01
    history:float=.25
    minimum_magnitude:float=.25
    maximum_magnitude:float=1.5
    max_history:int=8
    agreement_horizon:int=16


def action_merge(k,w,base,config):
    if config.merge=='base':return base
    if config.merge=='keepout':return k
    if config.merge=='waypoint':return w
    if config.merge=='residual':combined=w+config.alpha*(k-base)
    elif config.merge=='max_delta':
        dk=config.alpha*(k-base);dw=w-base;same=dk*dw>=0
        dominant=torch.where(dk.abs()>=dw.abs(),dk,dw)
        combined=base+torch.where(same,dominant,dk+dw)
    elif config.merge=='blend':combined=(1-config.alpha)*w+config.alpha*k
    else:raise ValueError(config.merge)
    if k.shape[-1]==5:
        if config.axes=='xyz':combined=torch.cat([combined[:,:,:3],w[:,:,3:]],-1)
        elif config.axes=='z':combined=torch.cat([w[:,:,:2],combined[:,:,2:3],w[:,:,3:]],-1)
        elif config.axes!='all':raise ValueError(config.axes)
    return combined.clamp(-1,1)


class OriginalComposer:
    trace_noise_name='initial_original_skill_codes'

    def __init__(self,bank,config,seed=241):
        self.bank=bank;self.config=c=config;self.device=next(bank.parameters()).device
        assert c.starts>=2 and c.starts%2==0
        assert c.keepout_sign in [-1,0,1] and c.waypoint_sign in [-1,0,1]
        sector_count=(1 if c.keepout_sign else 2)*(1 if c.waypoint_sign else 2)
        assert c.starts%sector_count==0
        assert 0<c.minimum_magnitude<c.maximum_magnitude and c.steps>=0
        assert 1<=c.agreement_horizon<=16
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        self.previous=None;self.records=[];self.pending=None;self.noise=None;self.last_plan={}
        bank.eval().requires_grad_(False)
        from .inference_guard import forbidden
        for module in [bank.ek,bank.ew,bank.k.original_encoder,bank.w.original_encoder]:
            module.register_forward_pre_hook(forbidden)

    def coordinates(self,actions):
        if actions.shape[-1]==5:
            if self.config.axes=='all':return actions
            return actions[...,2:3] if self.config.axes=='z' else actions[...,:3]
        return actions

    def objective(self,state,bz,contexts,z,history=None,previous=None):
        c=self.config;k,w=self.bank.pair(state,bz,z,contexts)
        agreement=(self.coordinates(k[:,:c.agreement_horizon])-self.coordinates(w[:,:c.agreement_horizon])).square().mean((1,2))
        prior=z.square().mean(1);trust=(z.abs()-.8).square().mean(1)
        continuity=torch.zeros_like(prior) if previous is None else (z-previous).square().mean(1)
        reconstruction=torch.zeros_like(prior)
        if history is not None and c.history:
            hs,hbz,hctx,ha=history;n,t=hs.shape[:2]
            for j,actor in enumerate([self.bank.k,self.bank.w]):
                own=actor(hs.flatten(0,1),hbz.flatten(0,1),
                          z[:,None,j:j+1].expand(-1,t,-1).reshape(n*t,1),
                          hctx[j].flatten(0,1))[:,0].reshape(n,t,-1)
                reconstruction=reconstruction+.5*(self.coordinates(own)-self.coordinates(ha)).square().mean((1,2))
        loss=agreement+c.prior*prior+c.trust*trust+c.continuity*continuity+c.history*reconstruction
        return loss,k,w,{'agreement':agreement,'latent_prior':prior,'prior_trust':trust,
                         'past_action_reconstruction':reconstruction,'continuity':continuity}

    @torch.no_grad()
    def observe_action(self,action):
        if self.pending is None:return
        value=torch.as_tensor(action,device=self.device,dtype=self.pending[0].dtype)
        if value.ndim==1:value=value[None]
        self.records.append((*self.pending,value.detach().clone()));self.pending=None
        if len(self.records)>self.config.max_history:self.records.pop(1 if len(self.records)>1 else 0)

    def sample(self,state,contexts,base_z):
        c=self.config;n=len(state);self.bank.reset_counts()
        history=None
        if self.records and c.history:
            history=(torch.stack([r[0] for r in self.records],1),torch.stack([r[1] for r in self.records],1),
                tuple(torch.stack([r[2][j] for r in self.records],1) for j in [0,1]),
                torch.stack([r[3] for r in self.records],1))
        if not c.optimize:
            z=state.new_tensor([c.k_code,c.w_code])[None].expand(n,-1)
            self.noise=z.detach().clone()
            with torch.no_grad():k,w=self.bank.pair(state,base_z,z,contexts)
            self.last_plan={'optimizer_iterations':0,'starts':1,'simulator_queries':0,'history_observations':0}
        else:
            starts=c.starts
            sectors=[[k,w] for k in ([c.keepout_sign] if c.keepout_sign else [-1,1])
                     for w in ([c.waypoint_sign] if c.waypoint_sign else [-1,1])]
            signs=state.new_tensor(sectors).repeat(starts//len(sectors),1)[None].expand(n,-1,-1).clone()
            if c.preserve_mode and self.previous is not None:
                signs=self.previous.sign()[:,None].expand(n,starts,2).clone()
            magnitude=.4+.8*torch.rand(n,starts,2,device=self.device,generator=self.rng)
            initial=signs*magnitude
            if not c.optimize_waypoint:initial[:,:,1]=c.w_code
            if self.previous is not None:
                matches=(signs==self.previous.sign()[:,None]).all(-1)
                index=matches.to(torch.int64).argmax(1);initial[torch.arange(n,device=self.device),index]=self.previous
            self.noise=initial.detach().clone();signs=initial.sign().reshape(-1,2)
            fraction=((initial.abs()-c.minimum_magnitude)/(c.maximum_magnitude-c.minimum_magnitude)).clamp(.001,.999)
            parameter=torch.logit(fraction).reshape(-1,2).detach().requires_grad_(True)
            optimizer=torch.optim.Adam([parameter],lr=c.lr)
            s=state.repeat_interleave(starts,0);bz=base_z.repeat_interleave(starts,0)
            ctx=tuple(x.repeat_interleave(starts,0) for x in contexts)
            past=None if history is None else (history[0].repeat_interleave(starts,0),history[1].repeat_interleave(starts,0),
                tuple(x.repeat_interleave(starts,0) for x in history[2]),history[3].repeat_interleave(starts,0))
            prev=None if self.previous is None else self.previous.repeat_interleave(starts,0)
            best=state.new_full((n,starts),float('inf'));best_z=initial.clone()
            for step in range(c.steps+1):
                with torch.set_grad_enabled(step<c.steps):
                    z=signs*(c.minimum_magnitude+(c.maximum_magnitude-c.minimum_magnitude)*parameter.sigmoid())
                    if not c.optimize_waypoint:z=torch.cat([z[:,:1],state.new_full((len(z),1),c.w_code)],-1)
                    value=self.objective(s,bz,ctx,z,past,prev)[0].reshape(n,starts)
                with torch.no_grad():
                    improved=value<best;best=torch.minimum(best,value)
                    best_z=torch.where(improved[:,:,None],z.detach().reshape(n,starts,2),best_z)
                if step==c.steps:break
                gradient,=torch.autograd.grad(value.sum(),parameter)
                assert torch.isfinite(gradient).all()
                optimizer.zero_grad(set_to_none=True);parameter.grad=gradient;optimizer.step()
            with torch.no_grad():
                chosen=best.argmin(1);z=best_z[torch.arange(n,device=self.device),chosen]
                if c.keepout_sign:assert (z[:,0].sign()==c.keepout_sign).all()
                if c.preserve_mode and self.previous is not None:assert torch.equal(z.sign(),self.previous.sign())
                loss,k,w,details=self.objective(state,base_z,contexts,z,history,self.previous)
            self.last_plan={'optimizer_iterations':c.steps,'starts':starts,'simulator_queries':0,
                'agreement_horizon':c.agreement_horizon,
                'keepout_sign_constraint':c.keepout_sign,'preserve_mode':c.preserve_mode,
                'history_observations':len(self.records),'selected_codes':z.tolist(),
                'objective':loss.tolist(),**{key:value.tolist() for key,value in details.items()}}
        with torch.no_grad():
            base=self.bank.base(state,base_z)
            if c.merge=='sum':
                result=self.bank.execute(state,base_z,z,contexts,'sum')
                if result.shape[-1]==5:
                    if c.axes=='xyz':result=torch.cat([result[:,:,:3],w[:,:,3:]],-1)
                    elif c.axes=='z':result=torch.cat([w[:,:,:2],result[:,:,2:3],w[:,:,3:]],-1)
            else:result=action_merge(k,w,base,c)
        self.previous=z.detach().clone()
        self.pending=(state.detach().clone(),base_z.detach().clone(),tuple(x.detach().clone() for x in contexts))
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result.detach()
