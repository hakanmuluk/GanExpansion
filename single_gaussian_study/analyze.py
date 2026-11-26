"""Aggregate-only paired context/seed analysis; raw records stay on GPU."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .freeze import read,write,sha,verify,STEPS,KINDS
from .audit import OUTCOMES
from composition_ablation_study.analyze import wilson

BOOT=2000;BOOT_SEED=20261127

def csvfile(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def extract(job):
    out=Path(job['out']);records=read(out/'records.json');allplans=read(out/'planning.json');tt=job['environment']=='tworoom';result=[]
    for i,r in enumerate(records):
        plans=[p for p in allplans if p['step']<r['steps']] if tt else allplans[i];index=i if tt else 0
        arr=lambda key:np.array([p[key][index] for p in plans],float)
        raw0=arr('initial_raw_agreement');raw1=arr('agreement');obj0=arr('initial_objective');obj1=arr('objective')
        normal=np.array(plans[0]['gaussian_initial_sample'][index]);usable=np.array(plans[0]['initial_existing_inputs'][index])
        result.append({'context':i,**{k:float(r[k]) for k in OUTCOMES},'first_raw_before':raw0[0],
            'first_raw_after':raw1[0],'episode_raw_before':raw0.mean(),'episode_raw_after':raw1.mean(),
            'first_J_before':obj0[0],'first_J_after':obj1[0],'episode_J_before':obj0.mean(),'episode_J_after':obj1.mean(),
            'first_raw_improvement':raw0[0]-raw1[0],'first_J_improvement':obj0[0]-obj1[0],
            'first_raw_improved':float(raw0[0]-raw1[0]>1e-7),'first_J_improved':float(obj0[0]-obj1[0]>1e-7),
            'first_raw_improved_but_joint_failed':float(raw0[0]-raw1[0]>1e-7 and not r['success']),
            'first_J_improved_but_joint_failed':float(obj0[0]-obj1[0]>1e-7 and not r['success']),
            'episode_steps':r['steps'],'replans':len(plans),'native_planning_seconds':r['planning_seconds'],
            'native_replan_seconds':np.mean([p['seconds'] for p in plans]),
            'optimizer_loop_cuda_seconds':np.mean([p['optimizer_loop_cuda_seconds'] for p in plans]),
            'initial_gaussian_norm':float(np.linalg.norm(normal)),'initial_abs_u_mean':float(np.abs(normal).mean()),
            'initial_private_bound_projection_fraction':float(np.mean((np.abs(normal)<.01)|(np.abs(normal)>2.5))),
            'initial_sign_branch':''.join('+' if v>0 else '-' for v in usable[:2]),
            'wall_violation':float(r.get('wall_violation',False)),'early_goal_entry':float(r.get('early_goal_entry',False)),
            'dropped':float(r.get('dropped',False)),'table_violation':float(r.get('table_violation',False)),
            'robot_table_contact_in_transport':float(r.get('robot_table_contact_in_transport',False))})
    return result

def boot_indices(contexts=64,seeds=3):
    r=np.random.default_rng(BOOT_SEED)
    return r.integers(contexts,size=(BOOT,contexts)),r.integers(seeds,size=(BOOT,seeds))

def estimate(values,ci,si):
    values=np.asarray(values,float);assert values.ndim==2
    both=values[ci[:,:,None],si[:,None,:]].mean((1,2))
    context_only=values.mean(1)[ci].mean(1)
    return {'mean':float(values.mean()),'context_seed_bootstrap95':np.quantile(both,[.025,.975]).tolist(),
        'context_only_bootstrap95':np.quantile(context_only,[.025,.975]).tolist(),
        'per_seed_means':values.mean(0).tolist(),'seed_mean_min':float(values.mean(0).min()),
        'seed_mean_max':float(values.mean(0).max()),'seed_mean_sample_std':float(values.mean(0).std(ddof=1))}

def conditional(numerator,denominator,ci,si):
    numerator=np.asarray(numerator,float);denominator=np.asarray(denominator,float)
    num=numerator[ci[:,:,None],si[:,None,:]].sum((1,2));den=denominator[ci[:,:,None],si[:,None,:]].sum((1,2));valid=den>0
    return {'value':float(numerator.sum()/denominator.sum()) if denominator.sum() else None,
        'bootstrap95':np.quantile(num[valid]/den[valid],[.025,.975]).tolist() if valid.any() else None,
        'bootstrap_valid':int(valid.sum()),'bootstrap_attempts':BOOT}

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
    source_hash=sha(__file__);s=read(a.study);verify(s);out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    jobs=[j for j in s['jobs'] if (Path(j['out'])/'DONE.json').exists()]
    if not a.partial:assert len(jobs)==90 and read(Path(a.study).parent/'final_audit.json')['complete']
    internal={};seed_rows=[];cell_rows=[]
    for j in jobs:
        rows=extract(j);key=(j['environment'],j['category'],j['steps'],j['seed']);internal[key]=rows
        numerical=[k for k,v in rows[0].items() if isinstance(v,(float,int,np.floating)) and k!='context']
        r={'environment':j['environment'],'category':j['category'],'steps':j['steps'],'seed':j['seed'],'contexts':64}
        r.update({k:float(np.mean([x[k] for x in rows])) for k in numerical})
        r.update({k+'_count':int(sum(x[k] for x in rows)) for k in OUTCOMES})
        r.update({k+'_wilson95_conditional_on_seed':wilson(int(sum(x[k] for x in rows)),64) for k in OUTCOMES})
        assert r['steps']==j['steps'] and r['steps'] in STEPS
        seed_rows.append(r)
        native=read(Path(j['out'])/'summary.json');validation=read(Path(j['out'])/'DONE.json')['validation']
        ps=read(Path(j['out'])/'planning.json');ps=ps if j['environment']=='tworoom' else [p for episode in ps for p in episode]
        cell_rows.append({**r,'native_batch_size':64 if j['environment']=='tworoom' else 1,
            'total_evaluation_seconds':native['total_seconds'],'planning_calls_audited':validation['planning_calls'],
            'actor_forward_candidate_equivalents':sum(p['model_evaluation_counts']['independent']['candidate_equivalents'] for p in ps),
            'base_forward_candidate_equivalents':sum(p['model_evaluation_counts']['base']['candidate_equivalents'] for p in ps),
            'input_backward_candidate_equivalents':sum(p['backward_candidate_equivalents'] for p in ps),
            'gaussian_scalar_samples':128})
    summaries=[];paired=[];sensitivity=[];convergence=[];failure=[];ci,si=boot_indices()
    for env,e in s['environments'].items():
        for kind in KINDS:
            def matrix(n,key):return np.array([[internal[env,kind,n,seed][i][key] for seed in s['seeds']] for i in range(64)])
            if not all((env,kind,0,seed) in internal for seed in s['seeds']):continue
            zero=matrix(0,'success');norm=matrix(0,'initial_gaussian_norm');cuts=np.quantile(norm,[.25,.5,.75]);quartile=np.searchsorted(cuts,norm,side='right')
            branches=np.array([[internal[env,kind,0,seed][i]['initial_sign_branch'] for seed in s['seeds']] for i in range(64)])
            for n in STEPS:
                if not all((env,kind,n,seed) in internal for seed in s['seeds']):continue
                fields=[k for k,v in internal[env,kind,n,s['seeds'][0]][0].items() if isinstance(v,(float,int,np.floating)) and k!='context']
                estimates={key:estimate(matrix(n,key),ci,si) for key in fields}
                current=matrix(n,'success');recover=(zero==0)&(current==1);regress=(zero==1)&(current==0)
                diff=estimate(current-zero,ci,si)
                pair={'environment':env,'category':kind,'steps':n,'contexts':64,'initialization_seeds':3,'matched_rollouts':192,
                    'N0_fail_refined_success_count':int(recover.sum()),'N0_success_refined_fail_count':int(regress.sum()),
                    'N0_fail_count':int((zero==0).sum()),'N0_success_count':int((zero==1).sum()),
                    'success_difference':diff['mean'],'difference_context_seed_bootstrap95':diff['context_seed_bootstrap95'],
                    'difference_context_only_bootstrap95':diff['context_only_bootstrap95'],
                    'recoveries_per_seed':recover.sum(0).tolist(),'regressions_per_seed':regress.sum(0).tolist(),
                    'conditional_recovery':conditional(recover,zero==0,ci,si),'conditional_regression':conditional(regress,zero==1,ci,si)}
                paired.append(pair)
                summaries.append({'environment':env,'category':kind,'steps':n,'contexts':64,'initialization_seeds':3,
                    'evaluated_rollouts':192,'metrics':estimates,'paired_with_zero':pair})
                for q in range(4):
                    mask=quartile==q
                    sensitivity.append({'environment':env,'category':kind,'steps':n,'group':'raw_Gaussian_norm_quartile'+str(q+1),
                        'matched_context_seed_pairs':int(mask.sum()),'success':float(current[mask].mean()),
                        'first_raw_after':float(matrix(n,'first_raw_after')[mask].mean()),'cutpoints':cuts.tolist(),
                        'scope':'Descriptive groups of matched context-seed pairs; not independent episodes.'})
                for branch in sorted(set(branches.flatten())):
                    mask=branches==branch
                    sensitivity.append({'environment':env,'category':kind,'steps':n,'group':'fixed_private_signs'+branch,
                        'matched_context_seed_pairs':int(mask.sum()),'success':float(current[mask].mean()),
                        'first_raw_after':float(matrix(n,'first_raw_after')[mask].mean()),'scope':'Descriptive fixed sampled branches; no branch search or reselection.'})
                # Retain first-replan optimization curves as aggregate curves.
                curves=[]
                for seed in s['seeds']:
                    j=next(j for j in jobs if (j['environment'],j['category'],j['steps'],j['seed'])==(env,kind,n,seed));ps=read(Path(j['out'])/'planning.json')
                    curves.append(ps[0]['refinement_objective_per_iteration'] if env=='tworoom' else [p[0]['refinement_objective_per_iteration'][0] for p in ps])
                values=np.stack(curves,1) #context,seed,iteration
                bootstrap=values[ci[:,:,None],si[:,None,:]].mean((1,2));q=np.quantile(bootstrap,[.025,.975],axis=0)
                convergence.append({'environment':env,'category':kind,'steps':n,'mean_first_replan_J_curve':values.mean((0,1)).tolist(),
                    'context_seed_bootstrap95':q.tolist(),'scope':'All starts and observed first states matched across N. Total regularized J, not raw MSE.'})
                failed=current==0
                failure.append({'environment':env,'category':kind,'steps':n,'failed_context_seed_pairs':int(failed.sum()),
                    **{k+'_among_failures':int(matrix(n,k)[failed].sum()) for k in ['wall_violation','early_goal_entry','dropped','table_violation','robot_table_contact_in_transport']},
                    **{k+'_false_among_failures':int((1-matrix(n,k))[failed].sum()) for k in ['keepout_satisfied','waypoint_satisfied','goal_success']}})
    reference=[]
    for env,e in s['environments'].items():
        for key,item in e['previous_results'].items():
            assert sha(item['path'])==item['sha256'];x=read(item['path']);n,kind=key.split('/')
            reference.append({'environment':env,'category':kind,'steps':int(n[1:]),**{k:x[k] for k in OUTCOMES},
                'contexts':64,'reference':'Previously completed original grid/four-start method; deterministic, no new evaluations.','source_sha256':item['sha256']})
    flat=[]
    for r in summaries:
        x={k:v for k,v in r.items() if k not in ['metrics','paired_with_zero']}
        for key,val in r['metrics'].items():
            x[key]=val['mean'];x[key+'_context_seed_low'],x[key+'_context_seed_high']=val['context_seed_bootstrap95']
            x[key+'_per_seed']=val['per_seed_means'];x[key+'_context_only95']=val['context_only_bootstrap95']
        flat.append(x)
    for name,rows in [('results.csv',flat),('seed_results.csv',seed_rows),('native_cell_timing_and_work.csv',cell_rows),
        ('paired_recovery_regression.csv',paired),('initialization_sensitivity.csv',sensitivity),('failure_taxonomy.csv',failure),('historical_grid_reference.csv',reference)]:csvfile(out/name,rows)
    write(out/'paired_recovery_regression.json',paired);write(out/'optimization_convergence.json',convergence)
    record={'complete':len(jobs)==90,'partial':len(jobs)!=90,'completed_cells':len(jobs),'evaluations':64*len(jobs),
        'independent_contexts':384,'contexts_per_category':64,'initialization_seeds':s['seeds'],'rows':summaries,
        'historical_grid_reference':reference,'bootstrap_attempts':BOOT,'bootstrap_rng_seed':BOOT_SEED,
        'uncertainty':'Two-way task-context and initialization-seed resampling, paired across methods; context-only intervals and all seed-specific rates also supplied. Only3 seeds, reused research banks and one frozen model pair; descriptive uncertainty.',
        'study_sha256':sha(a.study),'analyzer_source_sha256':source_hash,'raw_episode_data_exported':False}
    assert sha(__file__)==source_hash;write(out/'summary.json',record)
    print(json.dumps({'complete':record['complete'],'cells':len(jobs),'rows':len(summaries),'out':str(out)}))

if __name__=='__main__':main()
