"""One Gaussian pair per episode; original bounded Adam, no candidate search."""
from dataclasses import replace
import time
import torch
from independent_composition.regularized_latent_composition import RegularizedComposer
from independent_composition.unchanged_composition import action_merge


class SingleGaussianComposer(RegularizedComposer):
    trace_noise_name='single_initial_projected_existing_codes'

    def __init__(self,bank,config,seed=241,record=True):
        assert config.starts==1 and config.cycle_weight==0
        # The legacy constructor assumes multiple sign-sector starts. Reuse its
        # frozen-model/encoder guards, then set the actual one-start config.
        # Construction performs no candidate generation or neural evaluation.
        super().__init__(bank,replace(config,starts=4),seed)
        self.config=config;self.record=record;self._plan={};self._trace=[]
        self.gaussian=None;self.gaussian_draw_calls=0;self.replans=0
        self.initial_gaussian_override=None

    def initial(self,state,base_z):
        c=self.config;n=len(state)
        if self.initial_gaussian_override is None:
            raw=torch.randn(n,2,device=state.device,dtype=state.dtype,generator=self.rng)
            self.gaussian_draw_calls+=1
        else:
            raw=self.initial_gaussian_override.to(state).detach().clone()
            assert raw.shape==(n,2)
        self.gaussian=raw.detach().clone()
        signs=torch.where(raw>=0,1.,-1.)
        for j,sign in enumerate([c.keepout_sign,c.waypoint_sign]):
            if sign:signs[:,j]=float(sign)
        private=signs*raw.abs().clamp(c.minimum_magnitude,c.maximum_magnitude)
        amp=base_z[:,:1].abs().clamp(c.amplitude_minimum,c.amplitude_bound).expand(-1,2)
        initial=torch.cat([private,amp],-1)
        self.noise=initial[:,None].detach().clone()
        return initial

    def parameterize(self,initial):
        c=self.config
        private=(initial[:,:2].abs()-c.minimum_magnitude)/(c.maximum_magnitude-c.minimum_magnitude)
        amplitude=(initial[:,2:]-c.amplitude_minimum)/(c.amplitude_bound-c.amplitude_minimum)
        return torch.cat([torch.logit(private.clamp(.001,.999)),torch.logit(amplitude.clamp(.001,.999))],-1)

    def decode(self,parameter,signs):
        c=self.config
        private=signs*(c.minimum_magnitude+(c.maximum_magnitude-c.minimum_magnitude)*parameter[:,:2].sigmoid())
        amplitude=c.amplitude_minimum+(c.amplitude_bound-c.amplitude_minimum)*parameter[:,2:].sigmoid()
        return torch.cat([private,amplitude],-1)

    def sample(self,state,contexts,base_z):
        c=self.config;n=len(state)
        assert c.optimize and c.optimize_waypoint and not c.preserve_mode
        assert c.history==c.prior==c.continuity==c.trust==0 and c.starts==1
        assert c.agreement_horizon==8 and (base_z[:,0]!=0).all()
        self.bank.reset_counts();cold=self.previous is None
        before_draws=self.gaussian_draw_calls
        initial=self.initial(state,base_z) if cold else self.previous.detach().clone()
        self._warm_input=None if cold else initial.detach().clone()
        signs=initial[:,:2].sign()
        parameter=self.parameterize(initial).detach().requires_grad_(True)
        optimizer=torch.optim.Adam([parameter],lr=c.lr)
        best=state.new_full((n,),float('inf'));best_z=initial.clone()
        trace=[];events=None
        if self.record:
            assert state.is_cuda
            events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record();started=time.perf_counter()
        # Same Adam parameterization, sum gradient, defaults, and best-iterate
        # retention as AmplitudeComposer.sample, with exactly one trajectory.
        for step in range(c.steps+1):
            with torch.set_grad_enabled(step<c.steps):
                z=self.decode(parameter,signs)
                cost,k,w,details=self.objective(state,base_z,contexts,z)
            with torch.no_grad():
                better=cost<best;best=torch.minimum(best,cost)
                best_z=torch.where(better[:,None],z.detach(),best_z)
            if self.record:
                trace.append({'z':z.detach(),'objective':cost.detach(),
                    'raw':details['agreement'].detach(),'best':best.detach().clone()})
            if step==c.steps:break
            gradient,=torch.autograd.grad(cost.sum(),parameter)
            assert torch.isfinite(gradient).all()
            optimizer.zero_grad(set_to_none=True);parameter.grad=gradient;optimizer.step()
        if self.record:events[1].record()
        with torch.no_grad():
            loss,k,w,details=self.objective(state,base_z,contexts,best_z)
            result=action_merge(k,w,self.bank.base(state,base_z),c)
        self.previous=best_z.detach().clone();self.replans+=1
        self._draws_this_call=self.gaussian_draw_calls-before_draws
        self._cold=cold
        if self.record:
            events[2].record();events[2].synchronize()
            self._wall=time.perf_counter()-started;self._cuda=events[0].elapsed_time(events[1])/1000
            self._trace=trace;self._details={key:value.detach() for key,value in details.items()}
            self._loss=loss.detach();self._pair=(k.detach(),w.detach());self._issued=result.detach()
            self._counts={key:dict(value) for key,value in self.bank.calls.items()}
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result.detach()

    @property
    def last_plan(self):
        if not self.record or not self._trace:return self._plan
        c=self.config;t=self._trace;k,w=self._pair
        raw=(self.coordinates(k[:,:8])-self.coordinates(w[:,:8])).square().mean((1,2))
        return {'optimizer_iterations':c.steps,'starts':1,'coarse_network_candidates':0,
            'random_network_candidates':0,'gaussian_pairs_drawn_this_call':self._draws_this_call,
            'gaussian_pair_draw_calls_total':self.gaussian_draw_calls,
            'sampled_once_at_episode_start':True,'warm_start':not self._cold,
            'previous_selected_inputs':None if self._warm_input is None else self._warm_input.tolist(),
            'gaussian_initial_sample':self.gaussian.tolist() if self._cold else None,
            'initial_projected_inputs':self.noise[:,0].tolist() if self._cold else None,
            'initial_existing_inputs':t[0]['z'].tolist(),
            'selected_existing_inputs':self.previous.tolist(),'objective':self._loss.tolist(),
            **{key:value.tolist() for key,value in self._details.items()},
            'initial_raw_agreement':t[0]['raw'].tolist(),'initial_objective':t[0]['objective'].tolist(),
            'refinement_raw_per_iteration':torch.stack([x['raw'] for x in t],1).tolist(),
            'refinement_objective_per_iteration':torch.stack([x['objective'] for x in t],1).tolist(),
            'refinement_best_objective_per_iteration':torch.stack([x['best'] for x in t],1).tolist(),
            'refinement_existing_inputs_per_iteration':torch.stack([x['z'] for x in t],1).tolist(),
            'final_raw_agreement_common8':raw.tolist(),
            'executed_vs_keepout_common8':(self._issued[:,:8]-k[:,:8]).square().mean((1,2)).tolist(),
            'executed_vs_waypoint_common8':(self._issued[:,:8]-w[:,:8]).square().mean((1,2)).tolist(),
            'optimizer_loop_and_final_merge_wall_seconds':self._wall,'optimizer_loop_cuda_seconds':self._cuda,
            'model_evaluation_counts':self._counts,'backward_candidate_equivalents':len(k)*c.steps,
            'simulator_queries':0,'agreement_horizon':8,'logging_serialized_after_planning_timer':True}

    @last_plan.setter
    def last_plan(self,value):self._plan=value

    def trajectory_fields(self):
        return {'single_initial_gaussian':self.gaussian.detach().cpu().numpy()}
