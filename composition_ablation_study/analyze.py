"""Aggregate only: scene-level inference, no raw data/episode record export.

All records/trajectories are read on the execution host. Outputs contain only
category/method summaries, uncertainty and curve/histogram aggregates.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr

KINDS=['supported_easy','supported_conflict','coupled_ood']
MAIN=['h4_n20','h8_n20','h12_n20','h16_n20','h8_n0','h8_n5','h8_n10','h8_n40']
BOOT=2000
SEED=20261126

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def nullable(value):return float(value) if value is not None and np.isfinite(value) else None

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,mid-half),min(1.,mid+half)]

def bootstrap_mean(values):
    values=np.asarray(values,float)
    if not len(values):return [None,None]
    rng=np.random.default_rng(SEED)
    x=values[rng.integers(len(values),size=(BOOT,len(values)))].mean(1)
    return np.quantile(x,[.025,.975]).tolist()

def auc(labels,scores):
    y=np.asarray(labels,bool);score=np.asarray(scores,float);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return None
    return float((rankdata(score,method='average')[y].sum()-n1*(n1+1)/2)/(n1*n0))

def roc(labels,scores):
    y=np.asarray(labels,bool);score=np.asarray(scores,float)
    if not y.any() or y.all():return None
    order=np.argsort(-score,kind='stable');ys=y[order];ss=score[order]
    end=np.r_[np.flatnonzero(np.diff(ss)!=0),len(y)-1]
    tp=np.r_[0,np.cumsum(ys)[end]]/y.sum();fp=np.r_[0,np.cumsum(~ys)[end]]/(~y).sum()
    area=float(np.sum(np.diff(fp)*(tp[1:]+tp[:-1])*.5))
    assert abs(area-auc(y,score))<1e-12
    return {'false_positive_rate':fp.tolist(),'true_positive_rate':tp.tolist()}

def classifier(rows,metric='first_agreement',outcome='success'):
    y=np.array([r[outcome] for r in rows],bool);x=np.array([r[metric] for r in rows],float)
    assert np.isfinite(x).all()
    value=auc(y,-x);rng=np.random.default_rng(SEED);boot=[]
    groups=[np.array([i for i,r in enumerate(rows) if r['category']==kind]) for kind in KINDS]
    groups=[g for g in groups if len(g)]
    if value is not None:
        for _ in range(BOOT):
            idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);a=auc(y[idx],-x[idx])
            if a is not None:boot.append(a)
    cuts=np.quantile(x,[.25,.5,.75]);bins=np.searchsorted(cuts,x,side='right');quantiles=[]
    for j in range(4):
        sel=bins==j;n=int(sel.sum());k=int(y[sel].sum())
        quantiles.append({'quantile':j+1,'contexts':n,'successes':k,'success':k/n if n else None,
            'wilson95':wilson(k,n),'mean_raw_disagreement':float(x[sel].mean()) if n else None,
            'failures':n-k})
    correlation=nullable(spearmanr(x,y.astype(float)).statistic) if np.ptp(x)>0 and not (y.all() or not y.any()) else None
    return {'contexts':len(rows),'outcome':outcome,'successes':int(y.sum()),'auroc_negative_raw_disagreement':value,
        'sparse_outcome_class':min(int(y.sum()),int((~y).sum()))<5,
        'sparse_class_note':'Bootstrap intervals cannot represent variation in unobserved outcome-class examples; sparse cells remain descriptive.' if min(int(y.sum()),int((~y).sum()))<5 else None,
        'bootstrap95':np.quantile(boot,[.025,.975]).tolist() if boot else [None,None],
        'bootstrap_valid':len(boot),'bootstrap_attempts':BOOT if value is not None else 0,
        'undefined_reason':'Only one outcome class' if value is None else None,
        'spearman_raw_disagreement_vs_success':correlation,'quantile_cutpoints':cuts.tolist(),
        'quantiles':quantiles,'roc':roc(y,-x),'score':'negative raw action-MSE; never total objective',
        'uncertainty':'Context bootstrap, stratified by difficulty for pooled rows; ties use midranks.'}

def adjusted_auc(rows):
    groups=[[r for r in rows if r['category']==kind] for kind in KINDS]
    groups=[g for g in groups if auc([r['success'] for r in g],[-r['first_agreement'] for r in g]) is not None]
    if not groups:return {'auroc':None,'included_categories':[]}
    value=float(np.mean([auc([r['success'] for r in g],[-r['first_agreement'] for r in g]) for g in groups]))
    rng=np.random.default_rng(SEED);values=[]
    for _ in range(BOOT):
        estimates=[]
        for g in groups:
            idx=rng.integers(len(g),size=len(g))
            estimates.append(auc([g[i]['success'] for i in idx],[-g[i]['first_agreement'] for i in idx]))
        if all(v is not None for v in estimates):values.append(np.mean(estimates))
    return {'auroc':value,'bootstrap95':np.quantile(values,[.025,.975]).tolist() if values else [None,None],
            'bootstrap_valid':len(values),'bootstrap_attempts':BOOT,'included_categories':[g[0]['category'] for g in groups],
            'scope':'Equal-weight mean of within-category joint-success AUROCs. Fixed eligible strata; bootstrap replicate omitted if any becomes single-class.'}

def get_plan_value(plan,key,index):
    value=plan[key]
    return float(value[index]) if isinstance(value,list) else float(value)

def load_cell(job):
    folder=Path(job['out']);rows=read(folder/'records.json');planning=read(folder/'planning.json')
    assert len(rows)==64
    is_tworoom=job['environment']=='tworoom'
    per_episode=[];calls=planning if is_tworoom else [p for episode in planning for p in episode]
    for i,row in enumerate(rows):
        assert row['context']==i
        plans=planning if is_tworoom else planning[i]
        active=[p for p in plans if p['step']<row['steps']];assert active
        r={**row,'category':job['category'],'context_key':job['category']+'/'+str(i)}
        if job['search']!='single':
            j=i if is_tworoom else 0
            def values(key):return np.array([get_plan_value(p,key,j) for p in active],float)
            agreement=values('agreement');objective=values('objective');initial=values('initial_grid_objective')
            raw_initial=np.array([p['initial_grid_details']['agreement'][j] for p in active])
            assert np.isfinite(agreement).all() and np.isfinite(objective).all()
            r.update(first_agreement=float(agreement[0]),middle_agreement=float(agreement[len(agreement)//2]),
                last_agreement=float(agreement[-1]),mean_agreement=float(agreement.mean()),
                first_objective=float(objective[0]),mean_objective=float(objective.mean()),
                initial_first_objective=float(initial[0]),initial_first_agreement=float(raw_initial[0]),
                first_objective_improvement=float(initial[0]-objective[0]),
                mean_objective_improvement=float((initial-objective).mean()),
                first_raw_improvement=float(raw_initial[0]-agreement[0]),
                mean_raw_improvement=float((raw_initial-agreement).mean()),
                any_replan_raw_improvement=bool((raw_initial-agreement>1e-9).any()),
                any_replan_total_improvement=bool((initial-objective>1e-9).any()),
                common8_first=get_plan_value(active[0],'final_raw_agreement_common8',j),
                mean_common8=float(values('final_raw_agreement_common8').mean()),
                mean_executed_vs_waypoint=float(values('executed_vs_waypoint_common8').mean()),
                mean_executed_vs_keepout=float(values('executed_vs_keepout_common8').mean()),
                first_per_action=np.array(active[0]['final_raw_agreement_per_action'][j],float),
                first_selected=np.array(active[0]['selected_existing_inputs'][j],float),
                active_replans=len(active),mean_refinement_cuda_seconds=float(np.mean([p['refinement_cuda_seconds'] for p in active])),
                mean_replan_seconds=float(np.mean([p['seconds'] for p in active])))
            r['landmarks']={str(p['step']):get_plan_value(p,'agreement',j) for p in active if p['step'] in [0,32,64,96,128]}
            r['landmark_curves']={str(p['step']):np.asarray(p['refinement_best_objective_per_iteration'][j])/max(get_plan_value(p,'initial_grid_objective',j),1e-12)
                for p in active if p['step'] in [0,32,64,96,128] and 'refinement_best_objective_per_iteration' in p}
            codes=np.array([p['selected_existing_inputs'][j] for p in active],float)
            assert np.isfinite(codes).all() and (np.abs(codes)>=.01-1e-6).all() and (np.abs(codes)<=2.5+1e-6).all()
            r['maximum_selected_magnitude']=float(np.abs(codes).max())
            r['minimum_selected_magnitude']=float(np.abs(codes).min())
            r['near_bound_coordinate_fraction']=float(((np.abs(codes)<.0349)|(np.abs(codes)>2.4751)).mean())
            r['private_sign_flips']=int((np.diff(np.sign(codes[:,:2]),axis=0)!=0).sum())
            r['latent_total_variation']=float(np.abs(np.diff(codes,axis=0)).sum())
            if 'final_issued_control_disagreement_common8' in active[0]:
                r['first_issued_disagreement']=get_plan_value(active[0],'final_issued_control_disagreement_common8',j)
                r['first_gripper_sign_disagreement']=get_plan_value(active[0],'gripper_sign_disagreement_common8',j)
            if 'refinement_best_objective_per_iteration' in active[0]:
                curve=np.array(active[0]['refinement_best_objective_per_iteration'][j],float)
                assert np.isfinite(curve).all() and (np.diff(curve)<=1e-8).all()
                r['first_convergence_curve']=curve
        per_episode.append(r)
    cfg=read(folder/'config.json')['composition']
    summary={'environment':job['environment'],'category':job['category'],'setting':job['setting'],
        'search':job['search'],'seed':job['seed'],'H':cfg['agreement_horizon'],'N':cfg['steps'] if job['search']=='adam' else 0,
        'diagnostic':job['search']=='single' or job['setting'].endswith('parameterized'),
        'contexts':64,'native_batch_size':64 if is_tworoom else 1,
        'task_sha256':read(folder/'config.json')['task_sha256']}
    for key in ['success','goal_success','keepout_satisfied','waypoint_satisfied','keepout_full_success','waypoint_full_success']:
        k=sum(bool(r[key]) for r in rows);summary[key]=k/64
        summary[key+'_count']=k;summary[key+'_wilson95']=wilson(k,64)
    for key in ['first_agreement','mean_agreement','first_objective','mean_objective','initial_first_objective',
                'initial_first_agreement','first_objective_improvement','mean_objective_improvement','first_raw_improvement',
                'mean_raw_improvement',
                'common8_first','mean_common8','mean_executed_vs_waypoint','mean_executed_vs_keepout',
                'mean_refinement_cuda_seconds','first_issued_disagreement','first_gripper_sign_disagreement',
                'private_sign_flips','latent_total_variation']:
        values=[r[key] for r in per_episode if key in r]
        if values:summary[key]=float(np.mean(values));summary[key+'_bootstrap95']=bootstrap_mean(values)
    summary['steps_mean']=float(np.mean([r['steps'] for r in rows]))
    summary['native_rollout_wall_seconds_including_recording_and_physics']=float(read(folder/'summary.json')['total_seconds'])
    summary['native_total_planning_wall_seconds']=float(sum(p['seconds'] for p in calls))
    summary['native_replan_wall_seconds_mean']=float(np.mean([p['seconds'] for p in calls]))
    summary['native_replan_wall_seconds_median']=float(np.median([p['seconds'] for p in calls]))
    summary['native_replan_wall_seconds_p95']=float(np.quantile([p['seconds'] for p in calls],.95))
    summary['native_first_call_wall_seconds']=float(calls[0]['seconds'])
    summary['planning_wall_seconds_amortized_per_episode']=summary['native_total_planning_wall_seconds']/64
    summary['planning_calls']=len(calls)
    summary['backward_candidate_equivalents']=sum(p.get('backward_candidate_equivalents',0) for p in calls)
    summary['backward_pair_batches']=sum(p.get('backward_batched_passes',0) for p in calls)
    for name in ['independent','base']:
        for key in ['batched_calls','candidate_equivalents']:
            summary[name+'_'+key]=sum(p.get('model_evaluation_counts',{}).get(name,{}).get(key,0) for p in calls)
    if job['search']!='single':
        summary['share_first_raw_agreement_improved']=float(np.mean([r['first_raw_improvement']>0 for r in per_episode]))
        summary['share_first_total_objective_improved']=float(np.mean([r['first_objective_improvement']>0 for r in per_episode]))
        summary['mean_first_per_action_disagreement']=np.mean([r['first_per_action'] for r in per_episode],axis=0).tolist()
        summary['all_recorded_optimization_values_finite']=all(p.get('all_recorded_refinement_values_finite',True) for p in calls)
        summary['maximum_selected_magnitude']=max(r['maximum_selected_magnitude'] for r in per_episode)
        summary['minimum_selected_magnitude']=min(r['minimum_selected_magnitude'] for r in per_episode)
        summary['near_bound_coordinate_fraction']=float(np.mean([r['near_bound_coordinate_fraction'] for r in per_episode]))
        summary['contexts_with_first_raw_improvement_but_joint_failure']=sum(r['first_raw_improvement']>0 and not r['success'] for r in per_episode)
        summary['contexts_with_first_total_improvement_but_joint_failure']=sum(r['first_objective_improvement']>0 and not r['success'] for r in per_episode)
        summary['contexts_with_any_replan_raw_improvement_but_joint_failure']=sum(r['any_replan_raw_improvement'] and not r['success'] for r in per_episode)
        summary['contexts_with_any_replan_total_improvement_but_joint_failure']=sum(r['any_replan_total_improvement'] and not r['success'] for r in per_episode)
        summary['share_first_raw_agreement_exactly_zero']=float(np.mean([r['first_agreement']==0 for r in per_episode]))
    return summary,per_episode

def flat_csv(path,rows):
    expanded=[]
    for r in rows:
        entry={}
        for key,value in r.items():
            if isinstance(value,list) and len(value)==2 and all(v is None or isinstance(v,(int,float)) for v in value):
                entry[key+'_low'],entry[key+'_high']=value
            elif not isinstance(value,(list,dict)):entry[key]=value
        expanded.append(entry)
    fields=list(dict.fromkeys(k for r in expanded for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in expanded)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();source_sha256=sha(__file__);study=read(a.study);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    summaries=[];internal={};missing=[]
    for job in study['jobs']:
        if not (Path(job['out'])/'DONE.json').exists():missing.append(job['id']);continue
        s,rows=load_cell(job);summaries.append(s);internal[(job['environment'],job['setting'],job['category'])]=rows
    if not a.partial:assert not missing and len(summaries)==84,missing
    flat_csv(out/'all_cell_results.csv',summaries)
    flat_csv(out/'experiment_A_horizon.csv',[s for s in summaries if s['setting'] in [f'h{h}_n20' for h in [4,8,12,16]]])
    flat_csv(out/'experiment_C_steps.csv',[s for s in summaries if s['setting'] in [f'h8_n{n}' for n in [0,5,10,20,40]]])
    flat_csv(out/'additional_controls.csv',[s for s in summaries if s['setting'] not in MAIN])
    agreement={};auc_rows=[];paired=[];failures=[];random_aggregates=[];convergence=[]
    for env in ['tworoom','cube']:
        reference=[]
        for kind in KINDS:
            rows=internal.get((env,'h8_n20',kind))
            if not rows:continue
            reference+=rows;r=classifier(rows);agreement[env+'/'+kind]=r
            auc_rows.append({'environment':env,'category':kind,'bootstrap95':r['bootstrap95'],
                             **{k:v for k,v in r.items() if not isinstance(v,(list,dict))}})
            for outcome in ['goal_success','keepout_satisfied','waypoint_satisfied']:
                other=classifier(rows,outcome=outcome);agreement[env+'/'+kind+'/'+outcome]=other
                auc_rows.append({'environment':env,'category':kind,'bootstrap95':other['bootstrap95'],
                    **{k:v for k,v in other.items() if not isinstance(v,(list,dict))}})
            for landmark in [0,32,64,96,128]:
                eligible=[{**r,'landmark_score':r['landmarks'][str(landmark)]} for r in rows if str(landmark) in r['landmarks']]
                if eligible:
                    v=classifier(eligible,metric='landmark_score');v['eligibility']='Still active at fixed elapsed-control landmark; descriptive conditional final-outcome association.'
                    agreement[f'{env}/{kind}/landmark{landmark}']=v
            for setting in MAIN+['h8_n0_parameterized']:
                alt=internal.get((env,setting,kind))
                if not alt:continue
                assert [r['context_key'] for r in rows]==[r['context_key'] for r in alt]
                delta=np.array([int(x['success'])-int(y['success']) for x,y in zip(alt,rows)],float)
                paired.append({'environment':env,'category':kind,'setting':setting,'reference':'h8_n20',
                    'contexts':64,'success_difference':float(delta.mean()),'bootstrap95':bootstrap_mean(delta),
                    'reference_fail_alternative_success':int((delta==1).sum()),'reference_success_alternative_fail':int((delta==-1).sum())})
            grid=internal.get((env,'h8_n0',kind))
            if grid:
                for setting in [f'h8_n{n}' for n in [5,10,20,40]]+['h8_n0_parameterized']:
                    alt=internal.get((env,setting,kind))
                    if alt:
                        delta=np.array([int(x['success'])-int(y['success']) for x,y in zip(alt,grid)],float)
                        paired.append({'environment':env,'category':kind,'setting':setting,'reference':'h8_n0',
                            'contexts':64,'success_difference':float(delta.mean()),'bootstrap95':bootstrap_mean(delta),
                            'grid_fail_alternative_success':int((delta==1).sum()),'grid_success_alternative_fail':int((delta==-1).sum())})
            mapped=internal.get((env,'h8_n0_parameterized',kind))
            if mapped:
                for setting in [f'h8_n{n}' for n in [5,10,20,40]]:
                    alt=internal.get((env,setting,kind))
                    if alt:
                        delta=np.array([int(x['success'])-int(y['success']) for x,y in zip(alt,mapped)],float)
                        paired.append({'environment':env,'category':kind,'setting':setting,'reference':'h8_n0_parameterized',
                            'contexts':64,'success_difference':float(delta.mean()),'bootstrap95':bootstrap_mean(delta),
                            'zero_mapped_fail_alternative_success':int((delta==1).sum()),'zero_mapped_success_alternative_fail':int((delta==-1).sum())})
            for setting in MAIN+['h8_n0_parameterized']:
                selected=internal.get((env,setting,kind),[])
                if selected and all('first_convergence_curve' in r for r in selected):
                    curves=np.array([r['first_convergence_curve']/max(r['initial_first_objective'],1e-12) for r in selected])
                    rng=np.random.default_rng(SEED);idx=rng.integers(64,size=(BOOT,64));means=curves[idx].mean(1)
                    convergence.append({'environment':env,'category':kind,'setting':setting,'contexts':64,'landmark_controls':0,
                        'mean_best_objective_over_exact_grid_initial':curves.mean(0).tolist(),
                        'context_bootstrap95':np.quantile(means,[.025,.975],axis=0).tolist(),
                        'note':'Iteration0 evaluates transformed coarse starts; exact-grid objective is the denominator. Best objective includes selection across starts and prior/tether.'})
                if selected:
                    for landmark in [32,64,96,128]:
                        eligible=[r['landmark_curves'][str(landmark)] for r in selected if str(landmark) in r['landmark_curves']]
                        if not eligible:continue
                        curves=np.array(eligible);count=len(curves);rng=np.random.default_rng(SEED)
                        means=curves[rng.integers(count,size=(BOOT,count))].mean(1)
                        convergence.append({'environment':env,'category':kind,'setting':setting,'contexts':count,'landmark_controls':landmark,
                            'mean_best_objective_over_exact_grid_initial':curves.mean(0).tolist(),
                            'context_bootstrap95':np.quantile(means,[.025,.975],axis=0).tolist(),
                            'note':'Still-active landmark cohort; selected independently of outcome, but conditional on survival to this elapsed step.'})
            seeds=study['random_search_seeds']
            repeats=[internal.get((env,f'random_budget20_s{seed}',kind)) for seed in seeds]
            if all(repeats):
                assert all(len(r)==64 for r in repeats)
                combined={'environment':env,'category':kind,'setting':'random_budget20_mean3seeds',
                    'independent_contexts':64,'optimization_seeds':3,'individual_rollouts':192}
                for metric in ['success','goal_success','keepout_satisfied','waypoint_satisfied',
                               'first_agreement','mean_agreement','first_objective_improvement','mean_refinement_cuda_seconds']:
                    per_context=np.mean([[r[metric] for r in repeat] for repeat in repeats],axis=0)
                    combined[metric]=float(per_context.mean());combined[metric+'_context_bootstrap95']=bootstrap_mean(per_context)
                random_aggregates.append(combined)
                if grid:
                    delta=np.mean([[r['success'] for r in repeat] for repeat in repeats],axis=0)-np.array([r['success'] for r in grid])
                    paired.append({'environment':env,'category':kind,'setting':'random_budget20_mean3seeds','reference':'h8_n0',
                        'contexts':64,'optimization_seeds':3,'success_difference':float(delta.mean()),'bootstrap95':bootstrap_mean(delta)})
            threshold=float(np.quantile([r['first_agreement'] for r in rows],.25))
            low=[r for r in rows if r['first_agreement']<=threshold];bad=[r for r in low if not r['success']]
            failure={'environment':env,'category':kind,'low_agreement_definition':'At or below first-score within-category quartile1 cutpoint; ties retained.',
                'low_agreement_contexts':len(low),'low_agreement_failures':len(bad),'raw_agreement_cutpoint':threshold}
            for key in ['goal_success','keepout_satisfied','waypoint_satisfied']:
                failure['low_agreement_failure_'+key+'_false']=sum(not r[key] for r in bad)
            for key in ['wall_violation','early_goal_entry','dropped','robot_table_contact_in_transport','table_violation','keepout_violation']:
                failure['low_agreement_failure_'+key+'_true']=sum(bool(r.get(key,False)) for r in bad)
            for skill in ['keepout','waypoint']:
                control=internal.get((env,'diagnostic_single_'+skill,kind))
                if control:
                    byid={r['context_key']:r for r in control};key=skill+'_full_success'
                    failure['low_agreement_failures_where_fixed_single_'+skill+'_succeeds']=sum(byid[r['context_key']][key] for r in bad)
            failures.append(failure)
        if len(reference)==192:
            agreement[env+'/pooled']=classifier(reference)
            r=agreement[env+'/pooled'];auc_rows.append({'environment':env,'category':'pooled','bootstrap95':r['bootstrap95'],
                **{k:v for k,v in r.items() if not isinstance(v,(list,dict))}})
            agreement[env+'/category_adjusted_joint']=adjusted_auc(reference)
            for landmark in [0,32,64,96,128]:
                eligible=[{**r,'landmark_score':r['landmarks'][str(landmark)]} for r in reference if str(landmark) in r['landmarks']]
                if eligible:
                    v=classifier(eligible,metric='landmark_score');v['eligibility']='Still active at fixed elapsed-control landmark; descriptive conditional final-outcome association.'
                    agreement[f'{env}/pooled/landmark{landmark}']=v
            for outcome in ['goal_success','keepout_satisfied','waypoint_satisfied']:
                other=classifier(reference,outcome=outcome);agreement[env+'/pooled/'+outcome]=other
                auc_rows.append({'environment':env,'category':'pooled','bootstrap95':other['bootstrap95'],
                    **{k:v for k,v in other.items() if not isinstance(v,(list,dict))}})
            for name in ['middle_agreement','last_agreement','mean_agreement']:
                agreement[env+'/retrospective_'+name]=classifier(reference,name)
    (out/'agreement_predictiveness.json').write_text(json.dumps(agreement,indent=2,allow_nan=False)+'\n')
    flat_csv(out/'experiment_B_predictiveness.csv',auc_rows)
    flat_csv(out/'agreement_landmarks.csv',[{'environment':key.split('/')[0],'category':key.split('/')[1],
        'landmark_controls':int(key.rsplit('landmark',1)[1]),**value} for key,value in agreement.items() if '/landmark' in key])
    (out/'paired_context_differences.json').write_text(json.dumps(paired,indent=2,allow_nan=False)+'\n')
    flat_csv(out/'paired_context_differences.csv',paired)
    flat_csv(out/'random_search_context_aggregates.csv',random_aggregates)
    (out/'optimization_convergence.json').write_text(json.dumps(convergence,indent=2,allow_nan=False)+'\n')
    flat_csv(out/'low_agreement_failure_taxonomy.csv',failures)
    result={'partial':bool(missing),'completed_cells':len(summaries),'expected_cells':84,'missing':missing,
        'contexts_per_environment':192,'contexts_per_category':64,'rows':summaries,'random_search_aggregates':random_aggregates,
        'study_sha256':sha(a.study),'raw_episode_data_exported':False,
        'analysis_source_sha256':source_sha256,'context_bootstrap_replicates':BOOT,'context_bootstrap_seed':SEED,
        'uncertainty':'Wilson success intervals; context-bootstrap paired differences and AUROC. Reused existing banks; one frozen model pair.',
        'timing_scope':'Native TwoRoom batch64, Cube batch1. Amortized per-episode total is batchwork/64, not batch1 latency.',
        'random_compute_budget':{env:e['random_compute_budget'] for env,e in study['environments'].items()}}
    assert sha(__file__)==source_sha256,'Analysis source changed while executing'
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed_cells':len(summaries),'missing':len(missing),'out':str(out)}))

if __name__=='__main__':main()
