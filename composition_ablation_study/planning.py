"""Observe the original planner; only H, Adam steps, or search family varies.

Positive-step Adam calls the original sample() verbatim. Detached logging never
changes its arithmetic or gradient. Zero steps implements the requested exact
coarse-grid selection, without the original logit round-trip at latent bounds.
"""
from dataclasses import replace
import math
import time
import torch
from independent_composition.regularized_latent_composition import RegularizedComposer
from independent_composition.unchanged_composition import action_merge


def grid_candidates(state, base_z, config, previous):
    """Literal candidate construction from the existing AmplitudeComposer."""
    c=config;n=len(state)
    q=state.new_tensor(c.seed_quantiles)
    v=math.sqrt(2)*torch.erfinv(2*q-1)
    values=v[v.abs()>1e-6]
    if (v.abs()<=1e-6).any():
        values=torch.cat([values,state.new_tensor([-c.minimum_magnitude,c.minimum_magnitude])])
    values=values.sign()*values.abs().clamp(c.minimum_magnitude,c.maximum_magnitude)
    private=torch.cartesian_prod(values,values)
    if c.keepout_sign:private=private[private[:,0].sign()==c.keepout_sign]
    if c.waypoint_sign:private=private[private[:,1].sign()==c.waypoint_sign]
    amplitudes=torch.cartesian_prod(state.new_tensor([.25,1.,2.]),state.new_tensor([.25,1.,2.]))
    grid=torch.cat([private.repeat_interleave(len(amplitudes),0),amplitudes.repeat(len(private),1)],-1)
    candidates=grid[None].expand(n,-1,-1).clone()
    original_amplitude=base_z[:,:1].abs().clamp(c.amplitude_minimum,c.amplitude_bound)
    unchanged=torch.cat([private[None].expand(n,-1,-1),original_amplitude[:,None].expand(-1,len(private),2)],-1)
    candidates=torch.cat([candidates,unchanged],1)
    if previous is not None:candidates=torch.cat([candidates,previous[:,None]],1)
    return candidates


def expanded(state,bz,contexts,count):
    return (state.repeat_interleave(count,0),bz.repeat_interleave(count,0),
            tuple(x.repeat_interleave(count,0) for x in contexts))


class ObservedComposer(RegularizedComposer):
    def __init__(self,bank,config,seed=241,search='adam',random_candidates=0,record=True,zero_mode='exact_grid'):
        self._plan={};self._trace=[];self._recording=False
        super().__init__(bank,config,seed)
        assert search in ['adam','random']
        self.search=search;self.random_candidates=random_candidates;self.record=record
        assert zero_mode in ['exact_grid','original_parameterization']
        self.zero_mode=zero_mode

    def uses_original_sample(self):
        return self.search=='adam' and (self.config.steps>0 or self.zero_mode=='original_parameterization')

    def objective(self,state,bz,contexts,z,history=None,previous=None):
        if self._recording:
            call=len(self._trace)
            if call==1:self._refine_start.record()
            final_call=(self.config.steps+2 if self.uses_original_sample()
                        else 2 if self.search=='random' else 1)
            if call==final_call:self._refine_end.record()
        result=super().objective(state,bz,contexts,z,history,previous)
        if self._recording:
            loss,k,w,details=result
            self._trace.append({'loss':loss.detach(),'z':z.detach(),
                                'details':{key:value.detach() for key,value in details.items()},
                                'gradient_enabled':bool(torch.is_grad_enabled() and loss.requires_grad)})
            # Retain only final proposal tensors; no new model forwards.
            self._last_pair=(k.detach(),w.detach())
            if len(self._trace)==1:self._grid_end.record()
        return result

    def _select_without_adam(self,state,contexts,bz):
        c=self.config;n=len(state);self.bank.reset_counts()
        candidates=grid_candidates(state,bz,c,self.previous);count=candidates.shape[1]
        s,b,ctx=expanded(state,bz,contexts,count)
        with torch.no_grad():
            costs=self.objective(s,b,ctx,candidates.reshape(-1,4))[0].reshape(n,count)
            # Keep the unchanged four best coarse starts in the trace.
            choice=costs.topk(min(c.starts,count),largest=False,dim=1).indices
            self.noise=candidates.gather(1,choice[:,:,None].expand(-1,-1,4)).detach().clone()
            best_cost,index=costs.min(1)
            best_z=candidates[torch.arange(n,device=state.device),index]
            if self.search=='random':
                assert self.random_candidates>0
                m=self.random_candidates
                random=torch.rand(n,m,4,device=state.device,generator=self.rng)
                private=c.minimum_magnitude+(c.maximum_magnitude-c.minimum_magnitude)*random[:,:,:2]
                signs=[]
                for sign in [c.keepout_sign,c.waypoint_sign]:
                    if sign:signs.append(torch.full((n,m),float(sign),device=state.device))
                    else:signs.append(torch.where(torch.rand(n,m,device=state.device,generator=self.rng)<.5,-1.,1.))
                private=private*torch.stack(signs,-1)
                amplitude=c.amplitude_minimum+(c.amplitude_bound-c.amplitude_minimum)*random[:,:,2:]
                random_z=torch.cat([private,amplitude],-1)
                s,b,ctx=expanded(state,bz,contexts,m)
                values=self.objective(s,b,ctx,random_z.reshape(-1,4))[0].reshape(n,m)
                value,index=values.min(1)
                winner=random_z[torch.arange(n,device=state.device),index]
                best_z=torch.where((value<best_cost)[:,None],winner,best_z)
            loss,k,w,details=self.objective(state,bz,contexts,best_z)
            result=action_merge(k,w,self.bank.base(state,bz),c)
        self.previous=best_z.detach().clone()
        self.last_plan={'optimizer_iterations':0,'starts':c.starts,'coarse_network_candidates':count,
            'random_network_candidates':self.random_candidates if self.search=='random' else 0,
            'simulator_queries':0,'agreement_horizon':c.agreement_horizon,
            'selected_existing_inputs':best_z.tolist(),'objective':loss.tolist(),
            **{key:value.tolist() for key,value in details.items()},
            'zero_step_semantics':'Exact best original grid candidate, no logit-bound round-trip.'}
        assert all(not p.requires_grad and p.grad is None for p in self.bank.parameters())
        return result.detach()

    def sample(self,state,contexts,base_z):
        assert state.is_cuda,'All study planning measurements use the authorized GPU'
        self._n=len(state);self._trace=[];self._recording=self.record
        self._prior_previous=None if self.previous is None else self.previous.detach().clone()
        self._events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
        self._start,self._grid_end,self._refine_start,self._refine_end,self._end=self._events
        self._start.record();started=time.perf_counter()
        if self.uses_original_sample():
            result=super().sample(state,contexts,base_z)
        else:result=self._select_without_adam(state,contexts,base_z)
        self._executed=result.detach()
        self._end.record();self._end.synchronize()
        self._wall_seconds=time.perf_counter()-started
        self._recording=False
        return result

    @property
    def last_plan(self):
        plan=dict(self._plan)
        if not self.record or not self._trace:return plan
        n=self._n;c=self.config;trace=self._trace
        grid=trace[0];cost=grid['loss'].reshape(n,-1);index=cost.argmin(1)
        coarse={key:value.reshape(n,-1)[torch.arange(n,device=value.device),index].tolist()
                for key,value in grid['details'].items()}
        initial=cost.min(1).values
        refine=trace[1:-1]
        if self.uses_original_sample():
            losses=torch.stack([r['loss'].reshape(n,-1).min(1).values for r in refine])
            best=losses.cummin(0).values
            agreement=torch.stack([r['details']['agreement'].reshape(n,-1).min(1).values for r in refine])
            # Logging is intentionally serialized only when the evaluator reads
            # last_plan AFTER its synchronized planning timer has stopped.
            plan.update(refinement_objective_min_per_iteration=losses.T.tolist(),
                        refinement_best_objective_per_iteration=best.T.tolist(),
                        refinement_min_raw_agreement_per_iteration=agreement.T.tolist(),
                        all_recorded_refinement_values_finite=bool(torch.isfinite(losses).all()),
                        gradient_objective_evaluations=sum(r['gradient_enabled'] for r in trace))
        k,w=self._last_pair
        per_action=(self.coordinates(k)-self.coordinates(w)).square().mean(2)
        if k.shape[-1]==5:
            from independent_composition.regularized_latent_composition import issued_controls
            issued=(issued_controls(k)-issued_controls(w)).square().mean(2)
            plan.update(final_issued_control_disagreement_common8=issued[:,:8].mean(1).tolist(),
                        gripper_sign_disagreement_common8=((k[:,:8,4]>=0)!=(w[:,:8,4]>=0)).float().mean(1).tolist())
        final=trace[-1]['loss']
        plan.update(search_family=self.search,initial_grid_objective=initial.tolist(),
                    initial_grid_details=coarse,objective_improvement_from_exact_grid=(initial-final).tolist(),
                    final_raw_agreement_common8=per_action[:,:8].mean(1).tolist(),
                    final_raw_agreement_per_action=per_action.tolist(),
                    executed_vs_keepout_common8=(self.coordinates(self._executed[:,:8])-self.coordinates(k[:,:8])).square().mean((1,2)).tolist(),
                    executed_vs_waypoint_common8=(self.coordinates(self._executed[:,:8])-self.coordinates(w[:,:8])).square().mean((1,2)).tolist(),
                    planning_wall_seconds=self._wall_seconds,
                    planning_cuda_seconds=self._start.elapsed_time(self._end)/1000,
                    coarse_objective_cuda_seconds=self._start.elapsed_time(self._grid_end)/1000,
                    refinement_cuda_seconds=self._refine_start.elapsed_time(self._refine_end)/1000,
                    model_evaluation_counts={key:dict(value) for key,value in self.bank.calls.items()},
                    backward_candidate_equivalents=n*c.starts*c.steps if self.search=='adam' else 0,
                    backward_batched_passes=c.steps if self.search=='adam' else 0,
                    previous_candidate_used=self._prior_previous is not None,
                    logging_serialized_after_planning_timer=True)
        return plan

    @last_plan.setter
    def last_plan(self,value):self._plan=value


class SingleSkillControl:
    """Diagnostic only: unchanged release single-adapter path with u=-0.8."""
    trace_noise_name='fixed_original_private_code'
    def __init__(self,bank,skill,code=-.8):
        self.bank=bank;self.skill=skill;self.code=code;self.noise=None;self.last_plan={}

    def sample(self,state,contexts,base_z):
        self.bank.reset_counts();self.noise=state.new_full((len(state),1),self.code)
        index=0 if self.skill=='keepout' else 1;actor=[self.bank.k,self.bank.w][index]
        with torch.no_grad():result=actor(state,base_z,self.noise,contexts[index],gate=1)
        self.last_plan={'diagnostic_only':True,'single_skill':self.skill,'fixed_private_code':self.code,
                        'optimizer_iterations':0,'simulator_queries':0}
        return result.detach()
