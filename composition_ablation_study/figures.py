"""Vector/PNG presentation figures from final aggregate evidence only."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyze import KINDS,read,sha

COLORS=['#3274A1','#E1812C','#B3444A']
LABELS=['Easy','Conflict','Coupled/OOD']

def main():
    p=argparse.ArgumentParser();p.add_argument('--aggregate',required=True);p.add_argument('--latency',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.aggregate);s=read(root/'summary.json');timing=read(a.latency)
    assert not s['partial'] and s['completed_cells']==84 and timing['complete']
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);files=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,
        'legend.fontsize':9,'pdf.fonttype':42,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    def row(env,setting,kind):
        r=[r for r in s['rows'] if r['environment']==env and r['setting']==setting and r['category']==kind];assert len(r)==1;return r[0]
    def save(fig,name):
        for suffix in ['pdf','png']:
            path=out/(name+'.'+suffix);fig.savefig(path,bbox_inches='tight');files.append(path)
        plt.close(fig)
    def axes_pair(title):
        fig,axes=plt.subplots(1,2,figsize=(11.6,3.8),layout='constrained',sharey=True)
        fig.suptitle(title,fontsize=14)
        for ax,name in zip(axes,['TwoRoom','Cube']):ax.set_title(name);ax.grid(alpha=.18);ax.set_ylim(-3,103)
        return fig,axes
    def interval_points(ax,x,v,low,high,**kwargs):
        # Percentile intervals can exclude the point estimate. Draw the actual
        # interval around its midpoint instead of clipping a negative yerr.
        low=np.asarray(low);high=np.asarray(high);middle=(low+high)/2
        color=kwargs.get('color',COLORS[0])
        ax.errorbar(x,middle,yerr=[middle-low,high-middle],fmt='none',color=color,capsize=3)
        ax.plot(x,v,**kwargs)
    def success_curve(ax,env,settings,x):
        for kind,color,label in zip(KINDS,COLORS,LABELS):
            records=[row(env,key,kind) for key in settings];v=np.array([r['success']*100 for r in records])
            lo=np.array([r['success_wilson95'][0]*100 for r in records]);hi=np.array([r['success_wilson95'][1]*100 for r in records])
            ax.errorbar(x,v,yerr=[v-lo,hi-v],color=color,marker='o',capsize=3,label=label)
    fig,axes=axes_pair('Joint success versus agreement horizon (20 Adam steps; execute 8)')
    for ax,env in zip(axes,['tworoom','cube']):
        x=[4,8,12,16];success_curve(ax,env,[f'h{h}_n20' for h in x],x);ax.set_xticks(x);ax.set_xlabel('Agreement horizon H (actions)')
    axes[0].set_ylabel('Strict joint success (%)');axes[1].legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.supxlabel('64 existing contexts per difficulty; bars: 95% Wilson intervals. Exploratory benchmark reuse.',fontsize=9)
    save(fig,'horizon_success')
    fig,axes=axes_pair('Joint success versus Adam refinement (agreement horizon 8)')
    for ax,env in zip(axes,['tworoom','cube']):
        x=[0,5,10,20,40];success_curve(ax,env,[f'h8_n{n}' for n in x],x);ax.set_xticks(x);ax.set_xlabel('Adam steps')
        for kind,color in zip(KINDS,COLORS):
            r=row(env,'h8_n0_parameterized',kind);ax.plot(0,r['success']*100,marker='D',mfc='white',mec=color,ms=6)
    axes[0].set_ylabel('Strict joint success (%)');axes[1].legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.supxlabel('N=0: exact grid selection. Hollow diamonds: legacy parameterized-zero control. Bars: 95% Wilson intervals.',fontsize=9)
    save(fig,'steps_success')
    fig,axes=plt.subplots(2,2,figsize=(11.6,6.5),layout='constrained',sharex=True)
    fig.suptitle('Final raw disagreement versus Adam steps (not the regularized objective)',fontsize=14)
    for col,env in enumerate(['tworoom','cube']):
        for ri,metric in enumerate(['first_agreement','mean_agreement']):
            ax=axes[ri,col];x=[0,5,10,20,40]
            for kind,color,label in zip(KINDS,COLORS,LABELS):
                records=[row(env,f'h8_n{n}',kind) for n in x];v=np.array([r[metric] for r in records])
                ci=np.array([r[metric+'_bootstrap95'] for r in records]);interval_points(ax,x,v,ci[:,0],ci[:,1],color=color,marker='o',label=label)
            ax.grid(alpha=.18);ax.set_xticks(x)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.set_title(('TwoRoom' if env=='tworoom' else 'Cube')+(' | first replan' if ri==0 else ' | episode-mean active replans'))
            ax.set_ylabel('Raw action MSE');ax.set_xlabel('Adam steps')
    axes[0,1].legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False);fig.supxlabel('Context-bootstrap 95% intervals; every episode gets equal weight.',fontsize=9)
    save(fig,'agreement_by_steps')
    b=read(root/'agreement_predictiveness.json')
    fig,axes=plt.subplots(2,2,figsize=(11.6,7),layout='constrained')
    fig.suptitle('Does lower initial disagreement predict strict joint success?',fontsize=14)
    for col,env in enumerate(['tworoom','cube']):
        r=b[env+'/pooled'];ax=axes[0,col];curve=r['roc'];name='TwoRoom' if env=='tworoom' else 'Cube'
        if curve:
            ax.plot(curve['false_positive_rate'],curve['true_positive_rate'],color=COLORS[0],lw=2)
        ax.plot([0,1],[0,1],'--',color='.6');ax.set(xlim=(0,1),ylim=(0,1),xlabel='False positive rate',ylabel='True positive rate')
        ci=r['bootstrap95'];ax.set_title(f'{name}: AUROC {r["auroc_negative_raw_disagreement"]:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]')
        ax.grid(alpha=.15);ax=axes[1,col]
        q=r['quantiles'];v=np.array([100*(x['success'] or 0) for x in q]);lo=np.array([100*(x['wilson95'][0] or 0) for x in q]);hi=np.array([100*(x['wilson95'][1] or 0) for x in q])
        ax.bar(np.arange(4),v,color=COLORS[0],alpha=.85)
        ax.errorbar(np.arange(4),v,yerr=[v-lo,hi-v],fmt='none',c='black',capsize=4)
        for i,x in enumerate(q):ax.text(i,8,f'n={x["contexts"]}',ha='center',color='white',fontsize=9)
        ax.set(ylim=(0,105),ylabel='Joint success (%)',xlabel='Initial raw-disagreement quantile',xticks=np.arange(4),xticklabels=['Q1: lowest','Q2','Q3','Q4: highest']);ax.grid(axis='y',alpha=.15)
    fig.supxlabel('192 reused contexts/environment. Predictor: negative raw MSE before any controls. ROC CI: stratified context bootstrap; bars: Wilson.',fontsize=9)
    save(fig,'agreement_predictiveness')
    fig,axes=plt.subplots(1,2,figsize=(11.6,3.8),layout='constrained',sharey=True)
    for ax,env in zip(axes,['tworoom','cube']):
        x=[];v=[];low=[];high=[];n=[]
        for t in [0,32,64,96,128]:
            r=b.get(f'{env}/pooled/landmark{t}')
            if r and r['auroc_negative_raw_disagreement'] is not None:
                x.append(t);v.append(r['auroc_negative_raw_disagreement']);low.append(r['bootstrap95'][0]);high.append(r['bootstrap95'][1]);n.append(r['contexts'])
        v=np.array(v);interval_points(ax,x,v,low,high,marker='o',color=COLORS[0])
        for xx,yy,nn in zip(x,v,n):ax.annotate(f'n={nn}',(xx,yy),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        ax.axhline(.5,color='.6',ls='--');ax.set(xlabel='Elapsed controls at observed replan',ylabel='AUROC for joint success',ylim=(-.03,1.03),title='TwoRoom' if env=='tworoom' else 'Cube');ax.grid(alpha=.15)
    fig.suptitle('Fixed-landmark diagnostic: changing still-active cohorts',fontsize=14)
    fig.supxlabel('Descriptive final-outcome association: earlier violations may already have happened. One-class AUROCs omitted; cohorts are not interchangeable.',fontsize=9)
    save(fig,'agreement_landmarks')
    fig,axes=axes_pair('Joint success versus measured batch1 planning latency')
    times={(r['environment'],r['setting']):r for r in timing['rows']}
    for ax,env in zip(axes,['tworoom','cube']):
        settings=[f'h8_n{n}' for n in [0,5,10,20,40]];x=[1000*times[env,key]['median_wall_seconds'] for key in settings]
        success_curve(ax,env,settings,x)
        xt=1000*times[env,'random_budget20']['median_wall_seconds']
        for kind,color in zip(KINDS,COLORS):
            r=next(r for r in s['random_search_aggregates'] if r['environment']==env and r['category']==kind)
            v=r['success']*100;ci=r['success_context_bootstrap95'];interval_points(ax,[xt],[v],[ci[0]*100],[ci[1]*100],ls='',marker='*',ms=13,color=color)
        ax.set_xlabel('Median batch-one planner latency (ms)')
    axes[0].set_ylabel('Strict joint success (%)');axes[1].legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.supxlabel('Line points: N=0,5,10,20,40. Stars: random search matched to N20 matrix work (3 seeds, context-clustered CI). Common observed-state probes.',fontsize=9)
    save(fig,'success_vs_latency')
    convergence=read(root/'optimization_convergence.json')
    fig,axes=plt.subplots(2,2,figsize=(11.6,6.2),layout='constrained')
    for col,env in enumerate(['tworoom','cube']):
        for ri,landmark in enumerate([0,64]):
            ax=axes[ri,col]
            for kind,color,label in zip(KINDS,COLORS,LABELS):
                matches=[r for r in convergence if r['environment']==env and r['category']==kind and r['setting']=='h8_n40' and r['landmark_controls']==landmark]
                if not matches:continue
                r=matches[0];v=np.array(r['mean_best_objective_over_exact_grid_initial']);ci=np.array(r['context_bootstrap95']);xx=np.arange(len(v))
                ax.plot(xx,v,color=color,label=f'{label} (n={r["contexts"]})');ax.fill_between(xx,ci[0],ci[1],color=color,alpha=.12)
            ax.axhline(1,color='.5',ls='--');ax.grid(alpha=.15);ax.set(xlabel='Adam updates',ylabel='Objective / best grid objective',title=('TwoRoom' if env=='tworoom' else 'Cube')+f' | elapsed {landmark}');ax.legend(fontsize=8)
    fig.suptitle('Optimization convergence: first and fixed 64-control replans',fontsize=14)
    fig.supxlabel('H8/N40; context-bootstrap 95% bands. Iteration 0 includes original bound parameterization. Later cohort includes only still-active contexts.',fontsize=9)
    save(fig,'optimization_convergence')
    fig,axes=plt.subplots(2,3,figsize=(12.8,6),layout='constrained',sharex=True)
    for ri,env in enumerate(['tworoom','cube']):
        for col,(kind,label) in enumerate(zip(KINDS,LABELS)):
            ax=axes[ri,col]
            for h,color in zip([4,8,12,16],['#4C78A8','#F58518','#54A24B','#B279A2']):
                r=row(env,f'h{h}_n20',kind);ax.plot(np.arange(1,17),r['mean_first_per_action_disagreement'],color=color,label=f'H={h}')
            ax.axvspan(.5,8.5,color='.7',alpha=.12);ax.set(xlim=(1,16),xlabel='Predicted action index',ylabel='Mean raw disagreement',title=('TwoRoom' if env=='tworoom' else 'Cube')+' | '+label);ax.grid(alpha=.15)
    axes[0,2].legend();fig.suptitle('Which parts of the proposed chunk agree? First observed replan',fontsize=14)
    fig.supxlabel('Shaded region: eight actions actually executed before re-observation. All generators still predict 16 actions.',fontsize=9)
    save(fig,'horizon_action_disagreement')
    record={'complete':True,'render_source_sha256':sha(__file__),'source_summary_sha256':sha(root/'summary.json'),'latency_summary_sha256':sha(a.latency),
        'files':{p.name:sha(p) for p in files},'raw_episode_data_exported':False,
        'uncertainty':'Context units; never treat replans or random seeds as independent contexts. Native TwoRoom batch64 timings are not plotted as batch1 latency.'}
    (out/'figure_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'complete':True,'figures':len(files)//2,'out':str(out)}))

if __name__=='__main__':main()
