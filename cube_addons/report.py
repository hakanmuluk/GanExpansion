"""Aggregate stored physical evaluations; cluster uncertainty by task context."""
import argparse
import json
from pathlib import Path
import numpy as np


def bootstrap(values, seed=585, repetitions=4000):
    values=np.asarray(values,float);rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(repetitions,len(values)))].mean(1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def clustered(rows,key):
    groups={}
    for row in rows:groups.setdefault(row['context'],[]).append(float(row[key]))
    values=[np.mean(v) for v in groups.values()]
    return {'mean':float(np.mean(values)),'context_bootstrap_95ci':bootstrap(values),'contexts':len(values)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='cube_addon_artifacts');args=p.parse_args()
    root=Path(args.root);out=root/'report';out.mkdir(exist_ok=True)
    summaries={};records={}
    for folder in sorted((root/'test').glob('*')):
        if not (folder/'summary.json').is_file():continue
        summary=json.loads((folder/'summary.json').read_text());rows=json.loads((folder/'records.json').read_text())
        summary['uncertainty']={key:clustered(rows,key) for key in ['success','released_success','constraint_success']}
        summaries[folder.name]=summary;records[folder.name]=rows
    aggregate={}
    for skill in ['keepout','waypoint']:
        names=[f'{skill}_s{seed}' for seed in [42,43,44]]
        if all(name in summaries for name in names):
            values=[summaries[name]['success'] for name in names]
            aggregate[skill]={'seeds':[42,43,44],'full_success_seed_mean':float(np.mean(values)),
                              'full_success_seed_sample_sd':float(np.std(values,ddof=1)),
                              'seed_successes':values,'episodes_per_seed':summaries[names[0]]['episodes']}
    temporal={}
    for folder in sorted((root/'temporal').glob('*')):
        if folder.name not in records or not (folder/'records.json').is_file():continue
        moving=json.loads((folder/'records.json').read_text())
        ids={r['context'] for r in moving};fixed=[r for r in records[folder.name] if r['context'] in ids]
        from .evaluate import summarize
        temporal[folder.name]={'fixed':summarize(fixed),'resampled':summarize(moving),
                               'comparison':'Identical first 64 test contexts, paired initial codes, original code fixed; modifier resampled after time zero at every replan.'}
    result={'models':summaries,'across_training_seeds':aggregate,'temporal':temporal,
            'uncertainty':'Percentile bootstrap of context means, 4000 replicates; codes in a context are clustered. Seed SD is separate and uses three training initializations.',
            'test_selection':'All 128 contexts retained; eight paired Gaussian codes per context. Selection used validation only.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Held-out native MuJoCo results','',result['test_selection'],'',
           '| Model | Episodes | Full skill | 95% context CI | Released goal | Strategy/code | Both strategies/context | Physical code RMSE |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        ci=s['uncertainty']['success']['context_bootstrap_95ci'];rmse=s.get('physical_latent_rmse')
        lines.append(f"| {name} | {s['episodes']} | {100*s['success']:.2f}% | {100*ci[0]:.2f}–{100*ci[1]:.2f}% | {100*s['released_success']:.2f}% | {100*s['strategy_matches_code']:.2f}% | {100*s['both_successful_strategies_per_context']:.2f}% | {rmse:.3f} |" if rmse is not None else
                     f"| {name} | {s['episodes']} | {100*s['success']:.2f}% | {100*ci[0]:.2f}–{100*ci[1]:.2f}% | {100*s['released_success']:.2f}% | {100*s['strategy_matches_code']:.2f}% | {100*s['both_successful_strategies_per_context']:.2f}% | — |")
    lines+=['',result['uncertainty'],'','Original and deterministic models receive the paired code labels for common evaluation contexts, but cannot condition on them. Their code-match rate is not learned code control.']
    for skill,s in aggregate.items():lines+=['',f"{skill}: mean full success {100*s['full_success_seed_mean']:.2f}%, sample SD {100*s['full_success_seed_sample_sd']:.2f} percentage points across seeds."]
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    plot(root,out,summaries,records,temporal)
    print(json.dumps({'models':list(summaries),'seed_summary':aggregate},indent=2))


def plot(root,out,summaries,records,temporal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,skill in zip(axes,['keepout','waypoint']):
        names=[n for n in summaries if n.startswith(skill+'_')]
        if not names:continue
        values=[100*summaries[n]['success'] for n in names]
        errors=np.array([[v-100*summaries[n]['uncertainty']['success']['context_bootstrap_95ci'][0],
                          100*summaries[n]['uncertainty']['success']['context_bootstrap_95ci'][1]-v] for n,v in zip(names,values)]).T
        ax.bar(np.arange(len(names)),values,color=['#287ab2' if '_s4' in n else '#939ba2' for n in names],yerr=errors,capsize=3)
        ax.set(xticks=np.arange(len(names)),xticklabels=[n[len(skill)+1:] for n in names],ylim=(0,105),ylabel='Full skill success (%)',title=skill)
        ax.tick_params(axis='x',rotation=35)
    fig.suptitle('Held-out native MuJoCo: original goal AND added constraint');fig.savefig(out/'success_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    for col,skill in enumerate(['keepout','waypoint']):
        rows=records.get(skill+'_s42',[])
        if not rows:continue
        x=np.array([r['code'] for r in rows]);y=np.array([r.get('recovered_code',np.nan) for r in rows]);valid=np.isfinite(y)
        axes[0,col].scatter(x[valid],y[valid],s=6,alpha=.25,color='#287ab2');axes[0,col].plot([-3,3],[-3,3],color='black',lw=1)
        axes[0,col].set(title=skill,xlabel='Sampled modifier',ylabel='New encoder, physical trajectory')
        axes[1,col].scatter(np.abs(x),[r['height_excursion']*100 for r in rows],s=6,alpha=.25,color='#db8131')
        axes[1,col].set(xlabel='Absolute modifier magnitude',ylabel='Executed vertical excursion (cm)')
    fig.savefig(out/'physical_latent_recovery.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,skill in zip(axes,['keepout','waypoint']):
        runs=sorted((root/'runs').glob(skill+'*'))
        for run in runs:
            path=run/'validation.jsonl'
            if not path.exists():continue
            rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            ax.plot([r['step'] for r in rows],[r['success']*100 for r in rows],marker='.',label=run.name[len(skill)+1:])
        ax.set(title=skill,xlabel='Training step within run',ylabel='Validation full skill (%)',ylim=(0,102));ax.legend(fontsize=6,ncol=2)
    fig.suptitle('Development runs; validation banks and protocols differ, see configurations');fig.savefig(out/'validation_experiments.png',dpi=180);plt.close(fig)
    if temporal:
        fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
        for ax,(name,values) in zip(axes,temporal.items()):
            ax.bar(['Fixed modifier','Resampled each replan'],[100*values[k]['success'] for k in ['fixed','resampled']],color=['#287ab2','#db8131'])
            ax.set(title=name,ylabel='Full skill success (%)',ylim=(0,102))
        fig.savefig(out/'temporal_consistency.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
