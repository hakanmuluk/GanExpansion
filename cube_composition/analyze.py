"""Aggregate completed empirical runs; model seeds and contexts stay explicit."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def bounded_mean_interval(mean,n,alpha=.05):
    """Conservative Chernoff-KL interval for independent context rates in[0,1]."""
    c=np.log(2/alpha)/n;p=float(mean)
    if p==0:return [0.,float(-np.expm1(-c))]
    if p==1:return [float(np.exp(-c)),1.]
    def kl(q):return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
    left,right=np.finfo(float).tiny,p
    for _ in range(70):
        mid=(left+right)/2
        if kl(mid)>c:left=mid
        else:right=mid
    lower=right;left,right=p,1-np.finfo(float).eps
    for _ in range(70):
        mid=(left+right)/2
        if kl(mid)>c:right=mid
        else:left=mid
    return [float(lower),float(left)]


def auc(score,label):
    a=np.asarray(score)[np.asarray(label,dtype=bool)];b=np.asarray(score)[~np.asarray(label,dtype=bool)]
    return float(((a[:,None]>b[None]).mean()+.5*(a[:,None]==b[None]).mean())) if len(a) and len(b) else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',nargs='+',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    roots=[Path(v) for v in args.root];out=Path(args.out);out.mkdir(parents=True,exist_ok=True);runs=[];groups=defaultdict(list)
    paths=sorted({p for root in roots for p in root.rglob('summary.json')})
    for path in paths:
        folder=path.parent
        if not(folder/'DONE.json').exists() or not(folder/'config.json').exists():continue
        summary=json.loads(path.read_text());config=json.loads((folder/'config.json').read_text());rows=json.loads((folder/'records.json').read_text());plans=json.loads((folder/'planning.json').read_text())
        method=folder.name;kind=rows[0]['kind'];record={'method':method,'kind':kind,'model_seed':config['args']['model_seed'],'search_seed':config['args']['seed'],'path':str(folder),**summary}
        runs.append(record);groups[(kind,method)].append((record,rows,plans))
    if not runs:raise ValueError('No completed evaluations')
    fields=['method','kind','model_seed','search_seed','episodes','success','released_success','keepout_satisfied','waypoint_satisfied','dropped','grasp_losses',
        'robot_table_contact_in_transport','supported_transport_path_fraction','ee_region_violation_diagnostic','planning_seconds','planning_calls',
        'planning_seconds_p50','planning_seconds_p95','initial_planning_seconds_p50','initial_planning_seconds_p95','planning_seconds_max',
        'native_controls','native_forks','backward_steps','prior_mean','latent_total_variation','action_disagreement_mean','stationary','native_forecast_prefix_max_error_m']
    with (out/'runs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(runs)
    aggregates=[];rng=np.random.default_rng(18585)
    metrics=['success','released_success','keepout_satisfied','waypoint_satisfied','dropped','grasped','grasp_losses','supported_transport_path_fraction',
        'ee_region_violation_diagnostic','latent_total_variation','stationary','planning_seconds_p50','planning_seconds_p95','native_controls']
    for (kind,method),items in sorted(groups.items()):
        records=[v[0] for v in items];rows=[r for _,rr,_ in items for r in rr];plans=[plan for _,_,episodes in items for episode in episodes for plan in episode]
        seeds=sorted({r['seed'] for r in rows});context_rates=np.array([np.mean([r['success'] for r in rows if r['seed']==s]) for s in seeds])
        samples=context_rates[rng.integers(len(context_rates),size=(2000,len(context_rates)))].mean(1)
        a={'kind':kind,'method':method,'evaluation_runs':len(items),'model_seeds':sorted({r['model_seed'] for r in records}),
            'search_seeds':sorted({r['search_seed'] for r in records}),'episodes':len(rows),'unique_contexts':len(seeds),
            'success_context_bootstrap_95ci':np.quantile(samples,[.025,.975]).tolist(),
            'success_context_bounded_mean_95ci':bounded_mean_interval(context_rates.mean(),len(context_rates)),
            'bootstrap_scope':'Resample unique physical contexts, keeping all observed model/search seeds for that context. Conditional on evaluated models, not a confidence interval over unseen training seeds.'}
        a['bounded_interval_scope']='Chernoff-KL bound for independent bounded context-average success rates, conditional on the evaluated model/search seeds. Conservative and non-degenerate for all-zero/all-one samples.'
        for skill in ['keepout','waypoint']:
            a['own_'+skill+'_released_success_mean']=float(np.mean([r['released_success'] and r[skill+'_satisfied'] for r in rows]))
        a['both_constraints_without_goal_mean']=float(np.mean([r['keepout_satisfied'] and r['waypoint_satisfied'] for r in rows]))
        for key in metrics:
            values=[r[key] for r in records if r.get(key) is not None];a[key+'_mean']=float(np.mean(values)) if values else None
            a[key+'_run_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        a['amortized_plan_seconds']=float(np.sum([r['planning_seconds'] for r in records])/np.sum([r['planning_calls'] for r in records]))
        x=[r['action_disagreement_initial'] for r in rows if r['action_disagreement_initial'] is not None];y=[r['success'] for r in rows if r['action_disagreement_initial'] is not None]
        a['initial_low_disagreement_success_auc']=auc(-np.asarray(x),y) if x else None
        a['action_disagreement_initial_mean']=float(np.mean(x)) if x else None
        predictions=[(episode[0]['native_predicted_success'],r['success']) for _,rr,episodes in items for r,episode in zip(rr,episodes) if episode[0].get('native_predicted_success') is not None and episode[0].get('native_horizon')==240]
        a['full_native_prediction_count']=len(predictions);a['full_native_prediction_accuracy']=float(np.mean([p==s for p,s in predictions])) if predictions else None
        a['real_prefix_max_error_m']=max([p['actual_prefix_max_error_m'] for p in plans if p.get('actual_prefix_max_error_m') is not None],default=None)
        aggregates.append(a)
    (out/'aggregate.json').write_text(json.dumps({'runs':runs,'groups':aggregates},indent=2)+'\n')
    kinds=sorted({g['kind'] for g in aggregates});fig,axes=plt.subplots(1,len(kinds),figsize=(max(7,6*len(kinds)),7),squeeze=False)
    for ax,kind in zip(axes[0],kinds):
        records=sorted([g for g in aggregates if g['kind']==kind],key=lambda g:g['success_mean']);yy=np.arange(len(records))
        means=np.array([g['success_mean'] for g in records]);low=np.array([g['success_context_bounded_mean_95ci'][0] for g in records]);high=np.array([g['success_context_bounded_mean_95ci'][1] for g in records])
        ax.barh(yy,100*means,color='#367fa4');ax.errorbar(100*means,yy,xerr=100*np.stack([np.maximum(0,means-low),np.maximum(0,high-means)]),fmt='none',ecolor='#26333b',capsize=2)
        ax.set(yticks=yy,yticklabels=[g['method'] for g in records],xlim=(0,102),xlabel='Native joint success (%)',title=kind.replace('_',' '));ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Completed physical evaluations; conservative bounded-mean context intervals');fig.tight_layout();fig.savefig(out/'success.png',dpi=180);fig.savefig(out/'success.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for kind in kinds:
        rr=[g for g in aggregates if g['kind']==kind];ax.scatter([g['amortized_plan_seconds']*1000 for g in rr],[g['success_mean']*100 for g in rr],label=kind)
        for g in rr:ax.annotate(g['method'],(g['amortized_plan_seconds']*1000,g['success_mean']*100),fontsize=6,xytext=(2,3),textcoords='offset points')
    ax.set(xscale='log',xlabel='Wall time per replan, including amortized initial solve (ms)',ylabel='Native joint success (%)',ylim=(-3,105));ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'runtime_tradeoff.png',dpi=180);plt.close(fig)
    print(json.dumps({'completed_runs':len(runs),'groups':len(aggregates),'episodes':sum(r['episodes'] for r in runs)},indent=2))


if __name__=='__main__':main()
