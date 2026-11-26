"""Three required publication figures from completed aggregate evidence."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .freeze import read,write,sha,STEPS,KINDS
COLORS=['#3274A1','#E1812C','#B3444A'];LABELS=['Easy','Conflict','Coupled/OOD']

def main():
    p=argparse.ArgumentParser();p.add_argument('--aggregate',required=True);p.add_argument('--latency',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.aggregate);s=read(root/'summary.json');t=read(a.latency);assert s['complete'] and t['complete'] and len(s['rows'])==30
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);files=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,
        'legend.fontsize':9,'pdf.fonttype':42,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    def row(env,kind,n):return next(x for x in s['rows'] if x['environment']==env and x['category']==kind and x['steps']==n)
    def pair(title,ylabel,success=False):
        fig,axes=plt.subplots(1,2,figsize=(11.6,3.9),layout='constrained',sharey=success)
        fig.suptitle(title,fontsize=14)
        for ax,name in zip(axes,['TwoRoom','Cube']):
            ax.set_title(name);ax.grid(alpha=.16);ax.set_xlabel('Adam updates');ax.set_xticks(STEPS)
            if success:ax.set_ylim(-3,103)
        axes[0].set_ylabel(ylabel)
        if not success:axes[1].set_ylabel(ylabel)
        return fig,axes
    def interval(ax,x,value,ci,color,label):
        lo=np.array(ci)[:,0];hi=np.array(ci)[:,1];mid=(lo+hi)/2
        ax.errorbar(x,mid,yerr=[mid-lo,hi-mid],fmt='none',capsize=3,color=color)
        ax.plot(x,value,'-o',c=color,label=label)
    def save(fig,name):
        for ext in ['pdf','png']:
            p=out/(name+'.'+ext);fig.savefig(p,bbox_inches='tight');files.append(p)
        plt.close(fig)
    fig,axes=pair('Can one Gaussian start discover a successful composition?','Strict joint success (%)',True)
    for ax,env in zip(axes,['tworoom','cube']):
        for kind,color,label in zip(KINDS,COLORS,LABELS):
            values=[row(env,kind,n)['metrics']['success'] for n in STEPS]
            interval(ax,STEPS,[100*x['mean'] for x in values],[[100*v for v in x['context_seed_bootstrap95']] for x in values],color,label)
            old=next(x for x in s['historical_grid_reference'] if x['environment']==env and x['category']==kind and x['steps']==20)
            ax.plot(20,100*old['success'],marker='s',ms=8,mfc='white',mec=color,ls='')
    axes[1].legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.supxlabel('64 reused contexts/category; 3 Gaussian seeds, all retained. Bars: paired context/seed bootstrap95. Hollow squares: historical grid Adam20.',fontsize=9)
    save(fig,'success_vs_steps')
    fig,axes=pair('Raw action disagreement after refining the same Gaussian start','First-replan raw action MSE')
    for ax,env in zip(axes,['tworoom','cube']):
        for kind,color,label in zip(KINDS,COLORS,LABELS):
            values=[row(env,kind,n)['metrics']['first_raw_after'] for n in STEPS]
            interval(ax,STEPS,[x['mean'] for x in values],[x['context_seed_bootstrap95'] for x in values],color,label)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[1].legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.supxlabel('At N=0 the plotted value equals pre-optimization disagreement. All N share that same initial point. Raw MSE is separate from regularized J.',fontsize=9)
    save(fig,'disagreement_vs_steps')
    fig,axes=pair('Joint success versus measured planning latency','Strict joint success (%)',True)
    for ax,env in zip(axes,['tworoom','cube']):
        times=[next(x for x in t['rows'] if x['environment']==env and x['steps']==n)['median_wall_seconds']*1000 for n in STEPS]
        for kind,color,label in zip(KINDS,COLORS,LABELS):
            values=[row(env,kind,n)['metrics']['success'] for n in STEPS]
            interval(ax,times,[100*x['mean'] for x in values],[[100*v for v in x['context_seed_bootstrap95']] for x in values],color,label)
            old=next(x for x in s['historical_grid_reference'] if x['environment']==env and x['category']==kind and x['steps']==20)
            ot=next(x for x in t['historical_grid_latency'] if x['environment']==env and x['setting']=='h8_n20')
            ax.plot(1000*ot['median_wall_seconds'],100*old['success'],marker='s',ms=8,mfc='white',mec=color,ls='')
        ax.xaxis.set_major_locator(MaxNLocator(6));ax.set_xlabel('Median uncontended batch-one latency (ms)')
    axes[1].legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.supxlabel('Line points: N=0,5,10,20,40. Same observed-state probes per method. Hollow squares: previously measured grid Adam20 reference.',fontsize=9)
    save(fig,'success_vs_latency')
    write(out/'figure_provenance.json',{'complete':True,'renderer_sha256':sha(__file__),'summary_sha256':sha(root/'summary.json'),
        'latency_sha256':sha(a.latency),'files':{x.name:sha(x) for x in files},'raw_episode_data_exported':False})
    print({'complete':True,'figures':3})

if __name__=='__main__':main()
