"""GPU correctness probes before any new closed-loop comparison."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from independent_composition.regularization_models import load_bank
from independent_composition.regularized_latent_composition import RegularizedConfig,RegularizedComposer
from independent_composition.inference_guard import tripwires
from .freeze import read,write,verify,sha
from .planning import SingleGaussianComposer

def initial_inputs(s,env,kind,n=4):
    old=read(s['reference_study']);j=next(j for j in old['jobs'] if j['environment']==env and j['category']==kind and j['setting']=='h8_n20');folder=Path(j['out'])
    if env=='tworoom':
        from tworoom_gan.env import observation
        from tworoom_composition.geometry import contexts
        from tworoom_composition.evaluate import load_cases
        case=load_cases(s['environments'][env]['tasks'][kind],'cuda',64);case={k:v[:n] for k,v in case.items()}
        state=observation(case['start'],case['goal']);ctx=contexts(case['start'],case,torch.zeros(n,device='cuda',dtype=torch.bool),torch.zeros(n,device='cuda'),torch.zeros(n,2,device='cuda'))
        return state,ctx,case['base_z']
    states=[];contexts=[];bz=[]
    for i in range(n):
        with np.load(folder/f'rollout_{i:04d}.npz') as d:states.append(d['states'][0]);contexts.append(d['contexts'][0]);bz.append(d['base_z'])
    tensor=lambda x:torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float32)
    context=tensor(contexts);return tensor(states),(context[:,0],context[:,1]),tensor(bz)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();s=read(a.study);verify(s);torch.set_num_threads(2);result={}
    for env,e in s['environments'].items():
        bank=load_bank(env,**e['models'],device='cuda');checks=0;maximum=0.;matched=0
        for kind in ['supported_easy','supported_conflict','coupled_ood']:
            state,ctx,bz=initial_inputs(s,env,kind)
            initial=[];hist=[]
            for n in s['steps']:
                c=replace(RegularizedConfig(**e['reference_config']),steps=n,starts=1)
                actor=SingleGaussianComposer(bank,c,241)
                with tripwires():actor.sample(state,ctx,bz)
                plan=actor.last_plan;initial.append(plan['gaussian_initial_sample']);hist.append(plan)
                assert plan['starts']==1 and plan['coarse_network_candidates']==0
                first=actor.previous.clone();rng=actor.rng.get_state().clone()
                with tripwires():actor.sample(state,ctx,bz)
                assert torch.equal(actor.rng.get_state(),rng) and actor.last_plan['gaussian_pairs_drawn_this_call']==0
                assert actor.last_plan['previous_selected_inputs']==first.tolist();checks+=1
                # Technical parity only: legacy grid chooses ONE known start;
                # compare its unchanged Adam loop from that exact input. This
                # never supplies initialization to a new evaluation episode.
                old=RegularizedComposer(bank,replace(c,starts=4),241);old.config=c
                with tripwires():a0=old.sample(state,ctx,bz)
                probe=SingleGaussianComposer(bank,c,241);probe.previous=old.noise[:,0].clone()
                with tripwires():a1=probe.sample(state,ctx,bz)
                error=float((a0-a1).abs().max());maximum=max(maximum,error)
                assert torch.equal(a0,a1) and torch.equal(old.previous,probe.previous);matched+=1
            assert all(x==initial[0] for x in initial)
            for x in hist:
                for key in ['refinement_existing_inputs_per_iteration','refinement_objective_per_iteration','refinement_raw_per_iteration']:
                    assert x[key]==np.asarray(hist[-1][key])[:,:x['optimizer_iterations']+1].tolist()
        bank.assert_frozen();result[env]={'gaussian_matching_and_warm_start_checks':checks,'one_start_legacy_Adam_parity_checks':matched,
            'maximum_action_difference':maximum,'bitwise_actions_and_codes_equal':True,'no_new_episode_grid_or_restarts':True}
    verify(s);write(Path(a.study).parent/'preflight.json',{'passed':True,'environments':result,'study_sha256':sha(a.study),
        'scope':'Actual GPU neural probes only. Legacy-grid calls are correctness controls for its Adam loop; they do not initialize any new evaluated episode.'})
    print(json.dumps({'passed':True,'environments':result}))

if __name__=='__main__':main()
