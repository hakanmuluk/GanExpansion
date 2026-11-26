"""Read actual results and enforce the controlled-study invariants."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .analyze import read,sha
from .evaluate import verify
from independent_composition.regularized_latent_composition import RegularizedConfig

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();source_sha256=sha(__file__);study=read(a.study);verify(study);checks=[];missing=[];allplans=0;extra_originals={}
    for e in study['environments'].values():
        frozen=read(e['old_freeze'])
        for name,digest in frozen['original_sources'].items():
            path=Path(name) if Path(name).is_absolute() else Path(study['original_root'])/name
            assert sha(path)==digest,path
            extra_originals[str(path)]=digest
    for job in study['jobs']:
        folder=Path(job['out'])
        if not (folder/'DONE.json').exists():missing.append(job['id']);continue
        done=read(folder/'DONE.json');config=read(folder/'config.json')
        c=json.loads(json.dumps(asdict(RegularizedConfig(**read(job['config'])))))
        e=study['environments'][job['environment']]
        assert config['composition']==c and config['task_sha256']==e['bank_sha256'][job['category']]
        assert config['models']=={path:sha(path) for path in e['models'].values()}
        assert config['study_sha256']==sha(a.study)
        assert done['episodes']==64 and done['all_weights_frozen'] and done['original_sources_models_banks_unchanged']
        assert done['inference_tripwires_enabled'] and done['inference_environment_queries']==0
        rows=read(folder/'records.json');summary=read(folder/'summary.json')
        assert len(rows)==64 and [r['context'] for r in rows]==list(range(64))
        for key in ['success','goal_success','keepout_satisfied','waypoint_satisfied','keepout_full_success','waypoint_full_success']:
            assert abs(np.mean([r[key] for r in rows])-summary[key])<1e-12,(job['id'],key)
        plans=read(folder/'planning.json');tt=job['environment']=='tworoom'
        reference_job=next(j for j in study['jobs'] if j['environment']==job['environment'] and j['category']==job['category'] and j['setting']=='h8_n20')
        reference=Path(reference_job['out']);initial_error=0.
        if tt:
            with np.load(folder/'trajectories.npz') as actual,np.load(reference/'trajectories.npz') as old:
                for key in ['start','goal','keepout','waypoint','base_z']:assert np.array_equal(actual[key],old[key])
                assert np.array_equal(actual['positions'][:,0],old['positions'][:,0])
        else:
            for i in range(64):
                with np.load(folder/f'rollout_{i:04d}.npz') as actual,np.load(reference/f'rollout_{i:04d}.npz') as old:
                    for key in ['states','contexts','poses']:
                        error=float(np.max(np.abs(actual[key][0]-old[key][0])));initial_error=max(initial_error,error)
                        assert error<=2e-6,(job['id'],key,error)
                    assert np.array_equal(actual['base_z'],old['base_z'])
        if job['search']!='single' and c['agreement_horizon']==8:
            reference_plans=read(reference/'planning.json')
            first=[plans[0]] if tt else [r[0] for r in plans]
            previous=[reference_plans[0]] if tt else [r[0] for r in reference_plans]
            for current,ref in zip(first,previous):
                assert np.allclose(current['initial_grid_objective'],ref['initial_grid_objective'],atol=1e-7,rtol=1e-6)
                if job['search']=='adam' and 'refinement_best_objective_per_iteration' in current:
                    x=np.asarray(current['refinement_best_objective_per_iteration']);y=np.asarray(ref['refinement_best_objective_per_iteration'])
                    length=min(x.shape[1],y.shape[1]);assert np.allclose(x[:,:length],y[:,:length],atol=2e-7,rtol=2e-6)
        episodes=[plans] if tt else plans
        assert len(episodes)==(1 if tt else 64)
        n=64 if tt else 1;plans_checked=0
        for episode in episodes:
            assert [p['step'] for p in episode]==list(range(0,len(episode)*8,8))
            for pi,plan in enumerate(episode):
                assert plan['simulator_queries']==0
                if job['search']=='single':
                    assert plan['fixed_private_code']==-.8 and plan['single_skill']==job['skill'];continue
                assert plan['agreement_horizon']==c['agreement_horizon']
                expected_grid=(90 if tt else 360)+int(pi>0)
                assert plan['coarse_network_candidates']==expected_grid
                assert plan['previous_candidate_used']==(pi>0)
                oldzero=job.get('zero_mode')=='original_parameterization'
                if job['search']=='random':
                    m=e['random_compute_budget']['random_candidates_after_unchanged_grid']
                    assert plan['random_network_candidates']==m
                    candidates=expected_grid+m+1;objective_calls=3;backward=0
                elif c['steps']>0 or oldzero:
                    candidates=expected_grid+4*(c['steps']+1)+1;objective_calls=c['steps']+3;backward=c['steps']
                else:candidates=expected_grid+1;objective_calls=2;backward=0
                assert plan['optimizer_iterations']==backward
                calls=plan['model_evaluation_counts']
                assert calls['independent']['candidate_equivalents']==2*n*candidates
                assert calls['independent']['batched_calls']==2*objective_calls
                assert calls['base']['candidate_equivalents']==n and calls['base']['batched_calls']==1
                assert plan['backward_candidate_equivalents']==n*4*backward and plan['backward_batched_passes']==backward
                if 'gradient_objective_evaluations' in plan:assert plan['gradient_objective_evaluations']==backward
                raw=np.asarray(plan['agreement']);per_action=np.asarray(plan['final_raw_agreement_per_action'])
                assert raw.shape==(n,) and per_action.shape==(n,16) and np.isfinite(per_action).all()
                assert np.allclose(raw,per_action[:,:c['agreement_horizon']].mean(1),rtol=2e-6,atol=1e-7)
                total=raw+c['amplitude_tether']*np.asarray(plan['base_amplitude_tether'])+c['private_prior_weight']*np.asarray(plan['private_gaussian_nll'])
                assert np.allclose(total,plan['objective'],rtol=2e-6,atol=2e-7)
                assert plan.get('all_recorded_refinement_values_finite',True)
                assert plan['logging_serialized_after_planning_timer']
                plans_checked+=1
        if tt:
            native=read(folder/'native_replay_aggregate.json')
            assert native['episodes']==64 and native['all_positions_and_outcomes_match'] and native['maximum_position_error_pixels']<5e-5
        else:assert len(list(folder.glob('rollout_*.npz')))==64
        if job.get('reference_control'):
            r=read(folder/'original_reference_replay.json')
            assert r['episodes']==64 and r['outcomes_and_lengths_exact'] and max(r['maximum_absolute_differences'].values())<=1e-6
        checks.append({'job':job['id'],'episodes':64,'planning_calls_checked':plans_checked,
                       'maximum_initial_observation_difference':initial_error,'matched_initial_conditions_and_shared_gradient_prefix_verified':True,
                       'source_models_task_config_and_neural_work_verified':True})
        allplans+=plans_checked
    complete=not missing and len(checks)==84
    if a.require_complete:assert complete,missing
    record={'complete':complete,'cells':len(checks),'expected_cells':84,'episodes':64*len(checks),'pending':missing,
        'original_files_verified':len(study['original_hashes']),'frozen_core_source_files_verified':len(study['source_hashes']),
        'additional_original_model_environment_sources_verified':len(extra_originals),
        'planning_calls_checked':allplans,'checks':checks,'study_sha256':sha(a.study),
        'audit_source_sha256':source_sha256,
        'raw_episode_data_exported':False,'scope':'Every completed cell: exact configurations/banks, all outcomes reaggregated, all replanning horizons/steps/regularizers/forward-backward counts checked, original reference parity, native TwoRoom replay.'}
    assert sha(__file__)==source_sha256,'Audit source changed while executing'
    Path(a.out).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['checks','pending']}))

if __name__=='__main__':main()
