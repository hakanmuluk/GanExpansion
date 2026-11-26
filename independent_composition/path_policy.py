"""Frozen private trajectory models and private action executors.

Policy inference accepts current state/context tensors only. The optimization
objective compares predicted trajectories and code priors. It never measures a
keepout/waypoint predicate, queries an environment, or simulates a candidate.
"""
import math
import torch
from .private_paths import RouteConfig,RouteDecoder,spline_proposal,world_path,path_context,load_controller,FREE,POINTS


def load_route(path,device='cuda'):
    saved=torch.load(path,map_location=device,weights_only=False)
    model=RouteDecoder(RouteConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['state'])
    return model.eval().requires_grad_(False),saved


def observed_task(state,contexts,base_z,active):
    center=state.new_tensor([.425,0.,0.])
    goal=state[:,28:31]/10+center;start=state[:,31:34]/10+center
    delta=goal[:,:2]-start[:,:2];length=delta.norm(dim=-1);axis=delta/length[:,None]
    perpendicular=torch.stack([-axis[:,1],axis[:,0]],-1)
    magnitude=.035+.04*torch.erf(base_z[:,0].abs()/math.sqrt(2))
    amplitude=torch.where(base_z[:,0]>=0,magnitude,-magnitude)
    own=[None,None];positions=[None,None]
    for j in active:
        ctx=contexts[j]
        spec=ctx[:,:3]*state.new_tensor([.2,.3,.2])+state.new_tensor([.425,0.,.2])
        relative=spec[:,:2]-start[:,:2]
        own[j]=torch.stack([length,amplitude,(relative*axis).sum(-1),
                            (relative*perpendicular).sum(-1),spec[:,2]-.2,.1*ctx[:,3]],-1)
        positions[j]=spec+.15*ctx[:,4:7]
    return start,goal,own,positions[active[0]]


class PrivatePathPolicy:
    trace_noise_name='initial_route_codes'

    def __init__(self,routes,controllers,method='compose',executor=0,steps=120,starts=8,
                 seed=41,prior=.0001,lr=.12,tolerance=.010):
        assert method in ['base','keepout','waypoint','compose','fixed']
        self.routes=routes;self.controllers=controllers;self.method=method;self.executor=executor
        self.steps=steps;self.starts=starts;self.prior=prior;self.lr=lr;self.tolerance=tolerance
        self.active=[0] if method in ['base','keepout'] else [1] if method=='waypoint' else [0,1]
        self.path=None;self.index=None;self.noise=None
        self.last_plan={'optimizer_iterations':0,'starts':0,'simulator_queries':0}
        self.device=next(routes[self.active[0]].parameters()).device
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        for module in [*routes,*controllers]:
            if module is not None:module.eval().requires_grad_(False)
        if len(self.active)==2:
            left={p.data_ptr() for m in [routes[0],controllers[0]] for p in m.parameters()}
            right={p.data_ptr() for m in [routes[1],controllers[1]] for p in m.parameters()}
            assert not left&right,'Learned parameters are shared across skills'

    def decode(self,codes,own):
        zk,zw=codes[:,:11],codes[:,11:]
        return [model(z,context,spline_proposal(z,context))
                for model,z,context in zip(self.routes,[zk,zw],own)]

    def objective(self,codes,own):
        pk,pw=self.decode(codes,own)
        disagreement=((pk-pw)/.1).square().mean((1,2))
        prior=codes[:,:FREE].square().mean(1)+codes[:,11:].square().mean(1)
        return disagreement+self.prior*prior,pk,pw,disagreement

    def initialize(self,state,contexts,base_z):
        start,goal,own,_=observed_task(state,contexts,base_z,self.active)
        n=len(state)
        with torch.no_grad():
            common=.5*torch.randn(n,self.starts,FREE,device=self.device,generator=self.rng)
            angles=torch.linspace(-math.pi,math.pi,self.starts+1,device=self.device)[:-1]
            angles=angles[None,:,None].expand(n,-1,-1)
            initial=torch.cat([common,angles,common.clone()],-1)
            self.noise=initial.clone()
        if len(self.active)==1:
            j=self.active[0];z=initial[:,0,:11] if j==0 else initial[:,0,11:]
            with torch.no_grad():relative=self.routes[j](z,own[j],spline_proposal(z,own[j]))
            self.executor=j
        elif self.method=='fixed':
            with torch.no_grad():
                values=self.objective(initial[:,0],own)
                relative=.5*(values[1]+values[2])
                self.relative_paths=[values[1],values[2]];self.selected_codes=initial[:,0]
                self.last_plan={'optimizer_iterations':0,'starts':1,'path_disagreement':values[3].tolist(),
                                'codes':initial[:,0].tolist(),'simulator_queries':0}
        else:
            expanded=[x.repeat_interleave(self.starts,0) for x in own]
            codes=initial.reshape(-1,21).clone().requires_grad_(True)
            optimizer=torch.optim.Adam([codes],lr=self.lr)
            with torch.no_grad():
                best=self.objective(codes,expanded)[0].clone();best_codes=codes.detach().clone()
                first=best.clone()
            for _ in range(self.steps):
                with torch.enable_grad():
                    loss=self.objective(codes,expanded)[0]
                    optimizer.zero_grad(set_to_none=True);loss.sum().backward();optimizer.step()
                with torch.no_grad():
                    codes[:,:FREE].clamp_(-3,3);codes[:,11:].clamp_(-3,3);codes[:,10].clamp_(-math.pi,math.pi)
                    value=self.objective(codes,expanded)[0];improved=value<best
                    best=torch.minimum(best,value);best_codes=torch.where(improved[:,None],codes,best_codes)
            with torch.no_grad():
                chosen=best.reshape(n,self.starts).argmin(1)+torch.arange(n,device=self.device)*self.starts
                selected=best_codes[chosen]
                value,pk,pw,disagreement=self.objective(selected,own)
                assert bool((value<=first[chosen]+1e-5).all())
                relative=pk if self.executor==0 else pw
                self.relative_paths=[pk,pw];self.selected_codes=selected
                self.last_plan={'optimizer_iterations':self.steps,'starts':self.starts,
                                'initial_objective':first[chosen].tolist(),'objective':value.tolist(),
                                'path_disagreement':disagreement.tolist(),'codes':selected.tolist(),'simulator_queries':0}
        with torch.no_grad():
            self.path=world_path(relative,start,goal).detach()
            self.start=start;self.goal=goal
            self.index=torch.zeros(n,device=self.device,dtype=torch.long)
            self.done=torch.zeros(n,device=self.device,dtype=torch.bool)

    def trajectory_fields(self):
        if self.path is None:return {}
        result={'planned_path':self.path[0].cpu().numpy()}
        if hasattr(self,'relative_paths'):
            for name,path in zip(['path_keepout','path_waypoint'],self.relative_paths):
                result[name]=world_path(path,self.start,self.goal)[0].cpu().numpy()
            result['selected_codes']=self.selected_codes[0].cpu().numpy()
        return result

    def sample(self,state,contexts,base_z):
        if self.method=='base':
            self.noise=state.new_zeros(len(state),0)
            self.last_plan={'optimizer_iterations':0,'starts':0,'simulator_queries':0}
            with torch.no_grad():return self.controllers[self.executor](state,base_z,gate=0)
        if self.path is None:self.initialize(state,contexts,base_z)
        else:self.last_plan={'optimizer_iterations':0,'starts':0,'simulator_queries':0}
        with torch.no_grad():
            j=self.executor
            _,_,_,position=observed_task(state,contexts,base_z,[j])
            row=torch.arange(len(state),device=self.device)
            for _ in range(POINTS):
                close=(position-self.path[row,self.index]).norm(dim=-1)<self.tolerance
                move=close&(self.index<POINTS-1)
                if not bool(move.any()):break
                self.index=torch.minimum(self.index+move,torch.full_like(self.index,POINTS-1))
            reached=(self.index==POINTS-1)&((position-self.path[:,-1]).norm(dim=-1)<self.tolerance)
            self.done|=reached
            context=path_context(contexts[j],position,self.path,self.index,self.done)
            actions=self.controllers[j](state,base_z,state.new_full((len(state),1),.8),context)
            self.last_plan.update(path_index=self.index.tolist(),path_done=self.done.tolist(),executor=j)
            return actions
