"""Aggregate paired held-out contexts without treating model seeds as new tasks."""
import argparse,json
from pathlib import Path
import numpy as np


def bootstrap_indices(rng,n,repeats=10000):
    assert n==256,'Primary design has 128 contexts in each supported stratum'
    return np.concatenate([rng.integers(0,128,(repeats,128)),128+rng.integers(0,128,(repeats,128))],1)


def rows_for(root,name):
    rows=[]
    for bank in ['test_supported_easy','test_supported_conflict']:
        p=root/bank/name/'records.json'
        if not p.exists():return None
        rows.extend(json.loads(p.read_text()))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='composition_artifacts');args=p.parse_args();root=Path(args.root);rng=np.random.default_rng(90317)
    roots=[root/f'main_seed{s}' for s in [42,123,2026]];methods=sorted(set(p.name for p in (roots[0]/'test_supported_easy').iterdir() if p.is_dir()))
    results=[];arrays={}
    for name in methods:
        rows=[rows_for(r,name) for r in roots];present=[r for r in rows if r is not None]
        values=np.array([[r['success'] for r in rows] for rows in present],float);arrays[name]=values
        modelrates=values.mean(1);contextmeans=values.mean(0);idx=bootstrap_indices(rng,len(contextmeans))
        item={'method':name,'unique_heldout_contexts':len(contextmeans),'model_seeds':[seed for seed,r in zip([42,123,2026],rows) if r is not None],
            'per_model_seed_success':modelrates.tolist(),'mean':float(modelrates.mean()),'model_seed_std':float(modelrates.std(ddof=1)) if len(modelrates)>1 else None,
            'shared_context_bootstrap95_conditional_on_available_model_seeds':np.quantile(contextmeans[idx].mean(1),[.025,.975]).tolist(),
            'per_model_seed_success_count':[int(v.sum()) for v in values]}
        flat=[r for trial in present for r in trial]
        item['failure_rates_over_all_rollouts_nonexclusive']={k:float(np.mean([r[k] for r in flat])) for k in ['keepout_violation','strict_wall_violation','early_goal_entry','terminal_stall','stalled']}
        item['goal_success']=float(np.mean([r['goal_success'] for r in flat]));item['waypoint_visited']=float(np.mean([r['visited'] for r in flat]))
        item['original_door_preserved']=float(np.mean([r['original_door_preserved'] for r in flat]));item['mean_controls']=float(np.mean([r['steps'] for r in flat]))
        item['style_preserving_joint_success']=float(np.mean([r['success'] and r['original_door_preserved'] for r in flat]))
        item['behavior']={k:float(np.mean([r[k] for r in flat])) for k in ['latent_total_variation','keepout_strategy_switch','waypoint_strategy_switch']}
        item['behavior']['latent_sign_flips_mean']=np.mean([r['latent_sign_flips'] for r in flat],axis=0).tolist()
        for skill in ['keepout','waypoint']:
            valid=[r for r in flat if r[skill+'_strategy']!=0]
            item['behavior'][skill+'_observable_strategy_fraction']=len(valid)/len(flat)
            item['behavior'][skill+'_code_match_given_observable_strategy']=float(np.mean([r[skill+'_strategy_matches_code'] for r in valid])) if valid else None
        results.append(item)
    paired=[]
    for other in ['naive_sum','agreement_adam_k','task_cem16x2_h64','task_adam4x4_h32','action_adam10_h64']:
        a=arrays['structured_task_h64'];b=arrays[other];seeds=min(len(a),len(b));delta=(a[:seeds]-b[:seeds]).mean(0)
        idx=bootstrap_indices(rng,len(delta));paired.append({'primary_minus':other,'model_seeds_compared':seeds,'contexts':len(delta),
            'success_difference':float(delta.mean()),'shared_context_paired_bootstrap95':np.quantile(delta[idx].mean(1),[.025,.975]).tolist()})
    # Equal-budget clean agreement: every independent generator gets 65 candidate-equivalent queries per active plan.
    budget=[]
    for name in ['agreement_adam_k','agreement_random_k64','agreement_cem_k2x32','agreement_hybrid_k57x6']:
        files=list((roots[0]).glob('*/'+name+'/planning.json'));queries=[];backwards=[]
        for file in files:
            info=json.loads(file.read_text());n=info['batch_size']
            for plan in info['planning']:
                queries.append(plan['calls']['independent']['candidate_equivalents']/(2*n));backwards.append(plan['backward_steps'])
        assert set(queries)=={65.},(name,set(queries))
        budget.append({'method':name,'queries_per_independent_generator_per_plan':65,'all_saved_plans_verified':len(queries),'backward_steps_per_plan':sorted(set(backwards))})
    result={'methods':results,'paired_differences':paired,'matched_query_budget_audit':budget,
        'uncertainty_note':'Model seeds share the same 256 test contexts. Bootstrap resamples contexts within each 128-context stratum jointly across model seeds; it is conditional on these three trained model pairs. Model-seed standard deviation is reported separately.'}
    (root/'aggregate_statistics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
