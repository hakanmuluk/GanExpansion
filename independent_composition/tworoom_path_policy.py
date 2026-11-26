"""TwoRoom full-route agreement with separate frozen neural executors.

Only code variables are optimized. No environment handle, transition model,
geometric teacher, constraint score, or learned shared executor is used here.
"""
import torch
from .tworoom_paths import condition,canonical,proposal,path_context


class TwoRoomPathPolicy:
    trace_noise_name='initial_route_codes'

    def __init__(self,routes,controllers,method='compose',executor=0,steps=120,starts=8,
                 seed=41,prior=.0001,lr=.12,tolerance=1.5,agreement='pointwise',
                 encoders=None,cycle_mode='none',cycle_weight=0.,code_bound=3.,
                 supports=None,support_weight=0.):
        self.routes=routes;self.controllers=controllers;self.method=method;self.executor=executor
        self.steps=steps;self.starts=starts;self.prior=prior;self.lr=lr;self.tolerance=tolerance
        assert agreement in ['pointwise','chamfer','hybrid']
        self.agreement=agreement
        assert cycle_mode in ['none','self','executed','projection']
        assert 0<code_bound<=3 and cycle_weight>=0
        assert cycle_mode=='none' or (encoders is not None and len(encoders)==2)
        self.encoders=encoders or [];self.cycle_mode=cycle_mode;self.cycle_weight=cycle_weight
        self.code_bound=code_bound
        self.supports=supports or [];self.support_weight=support_weight
        assert support_weight>=0 and (not support_weight or len(self.supports)==2)
        self.path=None;self.index=None;self.noise=None;self.last_plan={}
        self.device=next(routes[0].parameters()).device
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        for m in [*routes,*controllers,*self.encoders,*self.supports]:m.eval().requires_grad_(False)
        left={p.data_ptr() for m in [routes[0],controllers[0]] for p in m.parameters()}
        right={p.data_ptr() for m in [routes[1],controllers[1]] for p in m.parameters()}
        if self.encoders:
            left|={p.data_ptr() for p in self.encoders[0].parameters()}
            right|={p.data_ptr() for p in self.encoders[1].parameters()}
        if self.supports:
            left|={p.data_ptr() for p in self.supports[0].parameters()}
            right|={p.data_ptr() for p in self.supports[1].parameters()}
        assert not left&right,'Learned parameters shared between skills'

    def decode(self,codes,own):
        return [m(z,c,proposal(z,c)) for m,z,c in zip(self.routes,codes.split(10,-1),own)]

    def objective(self,codes,own):
        pk,pw=self.decode(codes,own)
        pointwise=(pk-pw).square().mean((1,2))
        if self.agreement=='pointwise':agreement=pointwise
        else:
            squared=(pk[:,:,None]-pw[:,None]).square().sum(-1)
            # Compare complete spatial curves without querying a constraint or
            # transition. Both directions prevent matching only a small subset.
            chamfer=.5*(squared.amin(1).mean(1)+squared.amin(2).mean(1))/pk.shape[-1]
            agreement=chamfer if self.agreement=='chamfer' else .1*pointwise+.9*chamfer
        cycle=torch.zeros_like(agreement)
        if self.cycle_weight and self.cycle_mode!='none':
            executed=pk if self.executor==0 else pw
            paths=[pk,pw] if self.cycle_mode=='self' else [executed,executed]
            recovered=[encoder(path,c) for encoder,path,c in zip(self.encoders,paths,own)]
            if self.cycle_mode in ['self','executed']:
                cycle=(torch.cat(recovered,-1)-codes).square().mean(1)
            else:
                projections=[m(z,c,proposal(z,c)) for m,z,c in zip(self.routes,recovered,own)]
                cycle=.5*sum((path-executed).square().mean((1,2)) for path in projections)
        support=torch.zeros_like(agreement)
        if self.support_weight:
            executed=pk if self.executor==0 else pw
            logits=torch.stack([model(executed,c)[:,0] for model,c in zip(self.supports,own)],-1)
            support=torch.nn.functional.softplus(-logits).sum(-1)
        value=agreement+self.prior*codes.square().mean(1)+self.cycle_weight*cycle+self.support_weight*support
        return value,pk,pw,agreement,cycle,support

    def initialize(self,state,contexts,base_z):
        n=len(state);own=[condition(state,c[:,:12],base_z) for c in contexts]
        common=(.5*torch.randn(n,self.starts,10,device=self.device,generator=self.rng)).clamp(-self.code_bound,self.code_bound)
        initial=torch.cat([common,common.clone()],-1);self.noise=initial.detach().clone()
        if self.method in ['keepout','waypoint']:
            self.executor=0 if self.method=='keepout' else 1;j=self.executor
            z=initial[:,0,j*10:(j+1)*10]
            with torch.no_grad():path=self.routes[j](z,own[j],proposal(z,own[j]))
            self.last_plan={'optimizer_iterations':0,'starts':1,'simulator_queries':0}
        elif self.method=='fixed':
            with torch.no_grad():
                values=self.objective(initial[:,0],own);path=.5*(values[1]+values[2])
                self.route_paths=values[1:3];self.selected_codes=initial[:,0].detach()
                self.last_plan={'optimizer_iterations':0,'starts':1,'path_disagreement':values[3].tolist(),'simulator_queries':0}
        else:
            expanded=[x.repeat_interleave(self.starts,0) for x in own]
            codes=initial.reshape(-1,20).detach().clone().requires_grad_(True)
            optimizer=torch.optim.Adam([codes],lr=self.lr)
            with torch.no_grad():
                best=self.objective(codes,expanded)[0].clone();best_codes=codes.detach().clone();first=best.clone()
            for _ in range(self.steps):
                with torch.enable_grad():
                    loss=self.objective(codes,expanded)[0]
                    optimizer.zero_grad(set_to_none=True);loss.sum().backward();optimizer.step()
                with torch.no_grad():
                    codes.clamp_(-self.code_bound,self.code_bound);value=self.objective(codes,expanded)[0];improved=value<best
                    best=torch.minimum(best,value);best_codes=torch.where(improved[:,None],codes,best_codes)
            with torch.no_grad():
                chosen=best.reshape(n,self.starts).argmin(1)+torch.arange(n,device=self.device)*self.starts
                selected=best_codes[chosen];value,pk,pw,agreement,cycle,support=self.objective(selected,own)
                assert bool((value<=first[chosen]+1e-5).all())
                path=pk if self.executor==0 else pw
                self.route_paths=[pk,pw];self.selected_codes=selected.detach()
                self.last_plan={'optimizer_iterations':self.steps,'starts':self.starts,
                    'agreement_loss':self.agreement,
                    'cycle_mode':self.cycle_mode,'cycle_loss':cycle.tolist(),
                    'learned_support_energy':support.tolist(),
                    'initial_objective':first[chosen].tolist(),'objective':value.tolist(),
                    'path_disagreement':agreement.tolist(),'simulator_queries':0}
        self.path=((path+1)*112).detach();self.index=torch.zeros(n,device=self.device,dtype=torch.long)
        self.done=torch.zeros(n,device=self.device,dtype=torch.bool)

    def trajectory_fields(self):
        result={'planned_path':self.path.cpu().numpy()} if self.path is not None else {}
        if hasattr(self,'route_paths'):
            result.update(predicted_keepout_path=(self.route_paths[0].detach().cpu().numpy()+1)*112,
                          predicted_waypoint_path=(self.route_paths[1].detach().cpu().numpy()+1)*112,
                          selected_codes=self.selected_codes.cpu().numpy())
        return result

    def sample(self,state,contexts,base_z):
        if self.method=='base':
            self.noise=state.new_zeros(len(state),0);self.last_plan={'optimizer_iterations':0,'simulator_queries':0}
            with torch.no_grad():return self.controllers[self.executor](state,base_z,gate=0)[:,:1]
        if self.path is None:self.initialize(state,contexts,base_z)
        else:self.last_plan={'optimizer_iterations':0,'starts':0,'simulator_queries':0}
        with torch.no_grad():
            cs,_=canonical(state);position=(cs[:,:2]+1)*112;row=torch.arange(len(state),device=self.device)
            for _ in range(65):
                close=(position-self.path[row,self.index]).norm(dim=-1)<self.tolerance
                advance=close&(self.index<64)
                if not bool(advance.any()):break
                self.index=(self.index+advance).clamp_max(64)
            self.done|=(self.index==64)&((position-self.path[:,-1]).norm(dim=-1)<self.tolerance)
            ctx=path_context(contexts[self.executor][:,:12],position,self.path,self.index,self.done)
            actions=self.controllers[self.executor](state,base_z,state.new_full((len(state),1),.8),ctx)
            self.last_plan.update(path_index=self.index.tolist(),executor=self.executor)
            # These adapters supervise the first control only. Re-observe after
            # that control instead of exposing the unsupervised future heads.
            return actions[:,:1]
