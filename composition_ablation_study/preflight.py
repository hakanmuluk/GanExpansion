"""Actual-GPU parity, zero-step semantics and outcome-blind compute matching."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from independent_composition.regularization_models import load_bank
from independent_composition.regularized_latent_composition import RegularizedConfig,RegularizedComposer
from independent_composition.inference_guard import tripwires
from .planning import ObservedComposer,grid_candidates,expanded
from .probes import initial
from .evaluate import verify

def linear_flops(function):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True,record_shapes=True) as profile:
        function();torch.cuda.synchronize()
    rows={r.key:int(r.flops or 0) for r in profile.key_averages() if r.key in ['aten::mm','aten::addmm','aten::bmm']}
    assert sum(rows.values())>0,rows
    return sum(rows.values()),rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
    path=Path(a.study);study=json.loads(path.read_text());verify(study);torch.set_num_threads(2)
    proof={'passed':False,'environments':{},'scope':'Actual GPU probes and arithmetic-work profiling only. No success outcomes used to choose random-search budget.'}
    for env,e in study['environments'].items():
        bank=load_bank(env,**e['models'],device='cuda');c=RegularizedConfig(**e['reference_config'])
        checks=[];zero_differences=[]
        for kind in ['supported_easy','supported_conflict','coupled_ood']:
            s,ctx,bz=initial(bank,env,e['tasks'][kind],64 if env=='tworoom' else 4)
            original=RegularizedComposer(bank,c,241);observed=ObservedComposer(bank,c,241)
            for warm in [False,True]:
                with tripwires():
                    reference=original.sample(s,ctx,bz);actual=observed.sample(s,ctx,bz)
                assert torch.equal(reference,actual) and torch.equal(original.previous,observed.previous)
                assert torch.equal(original.noise,observed.noise)
                plan=observed.last_plan
                assert plan['gradient_objective_evaluations']==20
                assert all(p==20 for p in [plan['backward_batched_passes']])
                grid=grid_candidates(s,bz,c,observed._prior_previous)
                assert torch.equal(grid,observed._trace[0]['z'].reshape(len(s),-1,4))
                checks.append({'category':kind,'batch':len(s),'warm_previous_candidate':warm,'actions_codes_initial_grid_bitwise_equal':True})
            # Changing the unused Adam RNG seed is not an independent trial.
            other=RegularizedComposer(bank,c,1241);first=RegularizedComposer(bank,c,241)
            with tripwires():assert torch.equal(first.sample(s,ctx,bz),other.sample(s,ctx,bz))
            zero=ObservedComposer(bank,replace(c,steps=0),241)
            original_zero=ObservedComposer(bank,replace(c,steps=0),241,zero_mode='original_parameterization')
            with tripwires():
                exact_actions=zero.sample(s,ctx,bz);mapped_actions=original_zero.sample(s,ctx,bz)
            candidates=grid_candidates(s,bz,c,None);ss,bb,cc=expanded(s,bz,ctx,candidates.shape[1])
            with torch.no_grad():
                costs=RegularizedComposer.objective(zero,ss,bb,cc,candidates.reshape(-1,4))[0].reshape(len(s),-1)
                best=candidates[torch.arange(len(s),device=s.device),costs.argmin(1)]
                assert torch.equal(zero.previous,best)
            zero_differences.append({'category':kind,'batch':len(s),'maximum_action_difference_from_legacy_zero':float((exact_actions-mapped_actions).abs().max()),
                'maximum_code_difference_from_legacy_zero':float((zero.previous-original_zero.previous).abs().max())})
            for horizon in [4,8,12,16]:
                h=ObservedComposer(bank,replace(c,agreement_horizon=horizon,steps=1),241)
                with tripwires():h.sample(s,ctx,bz)
                plan=h.last_plan
                raw=torch.tensor(plan['final_raw_agreement_per_action'])[:,:horizon].mean(1)
                assert torch.allclose(raw,torch.tensor(plan['agreement']),atol=1e-7,rtol=1e-6)
        # Work matching uses one measured reset state, never task outcomes.
        s,ctx,bz=initial(bank,env,e['tasks']['supported_easy'],1)
        def original20():
            with tripwires():RegularizedComposer(bank,c,241).sample(s,ctx,bz)
        def grid0():
            with tripwires():ObservedComposer(bank,replace(c,steps=0),241,record=False).sample(s,ctx,bz)
        original20();grid0()
        f20,ops20=linear_flops(original20);f0,ops0=linear_flops(grid0)
        z=s.new_tensor([[-.8,-.8,1.,1.]]).expand(4,-1).clone()
        ss,bb,cc=expanded(s,bz,ctx,4)
        base=RegularizedComposer(bank,c,241)
        def forward():
            with torch.no_grad(),tripwires():base.objective(ss,bb,cc,z)
        fforward,opforward=linear_flops(forward);per_candidate=fforward/4
        m=max(1,round((f20-f0)/per_candidate))
        def random():
            with tripwires():ObservedComposer(bank,c,241,search='random',random_candidates=m,record=False).sample(s,ctx,bz)
        frandom,opsrandom=linear_flops(random)
        relative=abs(frandom-f20)/f20;assert relative<.01,(env,m,relative)
        budget={'random_candidates_after_unchanged_grid':m,'adam_steps_matched':20,
            'profiled_linear_flops_adam20':f20,'profiled_linear_flops_exact_grid':f0,
            'profiled_linear_flops_random':frandom,'relative_work_difference':relative,
            'forward_pair_linear_flops_per_candidate':per_candidate,
            'operator_totals':{'adam20':ops20,'grid0':ops0,'forward_batch4':opforward,'random':opsrandom},
            'scope':'Matched matrix multiply arithmetic proxy including coarse grid, frozen forward and input-gradient backward work. Not an assertion of equal wall time or all hardware operations.'}
        proof['environments'][env]={'reference_parity_checks':checks,'seed_invariance_actual_probes_passed':True,
            'zero_exact_grid_and_all_four_agreement_horizons_checked':True,'zero_parameterization_differences':zero_differences,
            'compute_budget':budget}
        e['random_compute_budget']=budget
        for seed in study['random_search_seeds']:
            setting=f'random_budget20_s{seed}'
            for kind in ['supported_easy','supported_conflict','coupled_ood']:
                study['jobs'].append({'id':f'{env}_{setting}_{kind}','environment':env,'setting':setting,'category':kind,
                    'search':'random','seed':seed,'config':study['configurations'][env+'/h8_n20']['path'],
                    'random_candidates':m,'out':str(path.parent/'runs'/env/setting/kind),'reference_control':False})
        bank.assert_frozen()
    proof['passed']=True;verify(study)
    assert len(study['jobs'])==84
    (path.parent/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
    path.write_text(json.dumps(study,indent=2)+'\n')
    print(json.dumps({'passed':True,'cells':len(study['jobs']),'random_budgets':{k:e['random_compute_budget']['random_candidates_after_unchanged_grid'] for k,e in study['environments'].items()}}))

if __name__=='__main__':main()
