"""Inspect native records, matched Gaussian starts, sole warm starts and work."""
import argparse
from pathlib import Path
import numpy as np
from independent_composition.regularized_latent_composition import RegularizedConfig
from .freeze import read,write,sha,verify

OUTCOMES=['success','goal_success','keepout_satisfied','waypoint_satisfied','keepout_full_success','waypoint_full_success']

def plans_by_episode(folder,env):
    p=read(Path(folder)/'planning.json')
    return [[dict(x,**{k:v[i:i+1] for k,v in x.items() if isinstance(v,list) and len(v)==64}) for x in p] for i in range(64)] if env=='tworoom' else p

def audit_cell(s,j,require_done=True):
    out=Path(j['out']);env=j['environment'];e=s['environments'][env];n=64 if env=='tworoom' else 1
    config=read(out/'config.json');c=__import__('json').loads(__import__('json').dumps(vars(RegularizedConfig(**read(j['config'])))))
    rows=read(out/'records.json');summary=read(out/'summary.json')
    if require_done:assert read(out/'DONE.json')['single_gaussian_initialization_and_warm_start_audited']
    assert config['composition']==c and c['starts']==1 and c['steps']==j['steps'] and c['agreement_horizon']==8
    original=e['reference_config'];assert all(c[k]==v for k,v in original.items() if k not in ['starts','steps'])
    assert config['models']=={p:sha(p) for p in e['models'].values()}
    assert config['task_sha256']==e['bank_sha256'][j['category']]
    assert len(rows)==64 and [r['context'] for r in rows]==list(range(64))
    for k in OUTCOMES:assert abs(summary[k]-np.mean([r[k] for r in rows]))<1e-12
    plans=read(out/'planning.json');streams=[plans] if env=='tworoom' else plans
    if env=='tworoom':
        with np.load(out/'trajectories.npz') as d:base_codes=d['base_z']
    else:
        base_codes=[]
        for i in range(64):
            with np.load(out/f'rollout_{i:04d}.npz') as d:base_codes.append(d['base_z'])
        base_codes=np.asarray(base_codes)
    total=0;maxwarm=0.
    for si,stream in enumerate(streams):
        assert [x['step'] for x in stream]==list(range(0,8*len(stream),8))
        assert len(stream)>0;branch=np.sign(np.asarray(stream[0]['initial_existing_inputs'])[:,:2])
        for pi,p in enumerate(stream):
            assert p['optimizer_iterations']==j['steps'] and p['starts']==1
            assert p['coarse_network_candidates']==p['random_network_candidates']==0 and p['simulator_queries']==0
            assert p['gaussian_pairs_drawn_this_call']==int(pi==0) and p['gaussian_pair_draw_calls_total']==1
            assert p['warm_start']==(pi>0) and p['agreement_horizon']==8
            z=np.asarray(p['refinement_existing_inputs_per_iteration']);obj=np.asarray(p['refinement_objective_per_iteration']);raw=np.asarray(p['refinement_raw_per_iteration'])
            assert z.shape==(n,j['steps']+1,4) and obj.shape==raw.shape==(n,j['steps']+1)
            assert all(np.isfinite(x).all() for x in [z,obj,raw])
            assert np.array_equal(np.sign(z[:,:,:2]),np.broadcast_to(branch[:,None,:],z[:,:,:2].shape))
            assert (np.abs(z[:,:,:2])>=c['minimum_magnitude']-1e-7).all() and (np.abs(z[:,:,:2])<=c['maximum_magnitude']+1e-7).all()
            assert (z[:,:,2:]>=c['amplitude_minimum']-1e-7).all() and (z[:,:,2:]<=c['amplitude_bound']+1e-7).all()
            selected=np.asarray(p['selected_existing_inputs']);loss=np.asarray(p['objective']);agree=np.asarray(p['agreement'])
            assert np.allclose(loss,obj.min(1),atol=3e-7,rtol=2e-6)
            best_index=obj.argmin(1);assert np.allclose(selected,z[np.arange(n),best_index],atol=1e-7,rtol=1e-6)
            total_term=agree+c['amplitude_tether']*np.asarray(p['base_amplitude_tether'])+c['private_prior_weight']*np.asarray(p['private_gaussian_nll'])
            assert np.allclose(loss,total_term,atol=3e-7,rtol=2e-6)
            assert np.allclose(agree,p['final_raw_agreement_common8'],atol=1e-7,rtol=2e-6)
            assert np.allclose(p['initial_raw_agreement'],raw[:,0],atol=1e-7)
            assert np.allclose(p['initial_objective'],obj[:,0],atol=1e-7)
            if pi:
                previous=np.asarray(stream[pi-1]['selected_existing_inputs'])
                error=float(np.max(np.abs(previous-np.asarray(p['previous_selected_inputs']))));maxwarm=max(maxwarm,error);assert error==0
                # Original bounded reparameterization, including its fraction clamp.
                lo=np.array([c['minimum_magnitude']]*2+[c['amplitude_minimum']]*2)
                hi=np.array([c['maximum_magnitude']]*2+[c['amplitude_bound']]*2)
                absolute=np.c_[np.abs(previous[:,:2]),previous[:,2:]]
                expected=lo+(hi-lo)*np.clip((absolute-lo)/(hi-lo),.001,.999)
                expected[:,:2]*=np.sign(previous[:,:2]);assert np.allclose(z[:,0],expected,atol=4e-7,rtol=2e-6)
            else:
                gaussian=np.asarray(p['gaussian_initial_sample']);initial=np.asarray(p['initial_projected_inputs'])
                assert gaussian.shape==(n,2) and np.isfinite(gaussian).all()
                sg=np.where(gaussian>=0,1.,-1.)
                for k,sign in enumerate([c['keepout_sign'],c['waypoint_sign']]):
                    if sign:sg[:,k]=sign
                assert np.allclose(initial[:,:2],sg*np.clip(np.abs(gaussian),c['minimum_magnitude'],c['maximum_magnitude']),atol=1e-7)
                bz=base_codes if env=='tworoom' else base_codes[si:si+1]
                expected_amp=np.clip(np.abs(bz[:,:1]),c['amplitude_minimum'],c['amplitude_bound'])
                assert np.allclose(initial[:,2:],np.repeat(expected_amp,2,axis=1),atol=1e-7)
            calls=p['model_evaluation_counts'];assert calls['independent']['candidate_equivalents']==2*n*(j['steps']+2)
            assert calls['independent']['batched_calls']==2*(j['steps']+2)
            assert calls['base']['candidate_equivalents']==n and calls['base']['batched_calls']==1
            assert p['backward_candidate_equivalents']==n*j['steps']
            total+=1
    baseline=next(x for x in s['jobs'] if x['environment']==env and x['category']==j['category'] and x['seed']==j['seed'] and x['steps']==0)
    bf=Path(baseline['out']);initial_error=0.
    if j['steps']!=0:
        other=read(bf/'planning.json');other_streams=[other] if env=='tworoom' else other
        for current,reference in zip(streams,other_streams):
            for key in ['gaussian_initial_sample','initial_projected_inputs','initial_existing_inputs','initial_raw_agreement','initial_objective']:
                assert np.array_equal(current[0][key],reference[0][key]),(j['id'],key)
        if env=='tworoom':
            with np.load(out/'trajectories.npz') as x,np.load(bf/'trajectories.npz') as y:
                for key in ['start','goal','keepout','waypoint','base_z','single_initial_gaussian','positions']:
                    left=x[key][:,0] if key=='positions' else x[key];right=y[key][:,0] if key=='positions' else y[key]
                    assert np.array_equal(left,right)
        else:
            for i in range(64):
                with np.load(out/f'rollout_{i:04d}.npz') as x,np.load(bf/f'rollout_{i:04d}.npz') as y:
                    for key in ['states','contexts','poses']:
                        error=float(np.max(np.abs(x[key][0]-y[key][0])));initial_error=max(initial_error,error);assert error<=2e-6
                    assert np.array_equal(x['single_initial_gaussian'],y['single_initial_gaussian'])
    if env=='tworoom':assert read(out/'native_replay_aggregate.json')['all_positions_and_outcomes_match']
    else:assert len(list(out.glob('rollout_*.npz')))==64
    return {'episodes':64,'planning_calls':total,'one_gaussian_pair_and_no_search_verified':True,
        'initialization_matched_to_zero_steps':True,'sole_previous_code_warm_start_verified':True,
        'maximum_warm_start_difference':maxwarm,'maximum_initial_observation_difference':initial_error,
        'fixed_branches_bounds_objective_counts_and_finite_values_verified':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    s=read(a.study);verify(s);rows=[];pending=[]
    for j in s['jobs']:
        if not (Path(j['out'])/'DONE.json').exists():pending.append(j['id']);continue
        rows.append({'job':j['id'],**audit_cell(s,j)})
    # Shared first-replan optimizer prefixes across all N, not only equal starts.
    for env in s['environments']:
        for seed in s['seeds']:
            for kind in ['supported_easy','supported_conflict','coupled_ood']:
                group=[j for j in s['jobs'] if j['environment']==env and j['seed']==seed and j['category']==kind and (Path(j['out'])/'DONE.json').exists()]
                if not group:continue
                longest=max(group,key=lambda j:j['steps']);ref=read(Path(longest['out'])/'planning.json');rr=[ref[0]] if env=='tworoom' else [x[0] for x in ref]
                for j in group:
                    cur=read(Path(j['out'])/'planning.json');cc=[cur[0]] if env=='tworoom' else [x[0] for x in cur]
                    for x,y in zip(cc,rr):
                        for key in ['refinement_existing_inputs_per_iteration','refinement_objective_per_iteration','refinement_raw_per_iteration']:
                            assert np.array_equal(x[key],np.asarray(y[key])[:,:j['steps']+1].tolist()),(j['id'],key)
    complete=len(rows)==90 and not pending
    if a.require_complete:assert complete,pending
    record={'complete':complete,'cells':len(rows),'episodes':64*len(rows),'expected_cells':90,'independent_contexts':384,
        'initialization_seeds':3,'planning_calls':sum(x['planning_calls'] for x in rows),'checks':rows,'pending':pending,
        'shared_first_replan_gradient_prefixes_verified':True,'study_sha256':sha(a.study),'audit_source_sha256':sha(__file__),'raw_episode_data_exported':False}
    write(a.out,record);print(json.dumps({k:v for k,v in record.items() if k not in ['checks','pending']}))

if __name__=='__main__':main()
