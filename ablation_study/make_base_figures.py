"""Matched side-by-side base trajectories from ORIGINAL Gaussian evaluations.

Use the first test context and its first eight codes, without outcome selection.
Raw native arrays stay on the execution host. Export vector PDF/PNG only.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':200})
    evidence={};all_loaded={}
    for environment in ['tworoom','cube']:
        loaded={}
        for method in ['full','adversarial']:
            root=Path('ablation_results/evaluation')/f'original_gaussian_{environment}_{method}_s{args.seed}'
            name='independent_rollouts.npz' if environment=='tworoom' else 'rollouts.npz'
            with np.load(root/name) as data:
                keys=['positions','start','goal','lengths','z','success'] if environment=='tworoom' else ['poses','initial_cube','goal','lengths','z']
                loaded[method]={k:data[k][:8].copy() for k in keys}
            if environment=='cube':
                rows=json.loads((root/'records.json').read_text())[:8]
                loaded[method]['success']=np.array([r['released_success'] for r in rows])
        full,ablated=loaded['full'],loaded['adversarial']
        all_loaded[environment]=loaded
        assert np.array_equal(full['z'],ablated['z']),'Both policies must receive exactly the same codes'
        initial='start' if environment=='tworoom' else 'initial_cube'
        assert np.array_equal(full[initial],ablated[initial]) and np.array_equal(full['goal'],ablated['goal'])
        figure,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for ax,method,title in zip(axes,['full','adversarial'],['Full training','Without latent objectives']):
            data=loaded[method]
            for j in range(8):
                n=int(data['lengths'][j]);path=data['positions'][j,:n+1] if environment=='tworoom' else data['poses'][j,:n+1,3:]
                if environment=='tworoom':x,y=path[:,0],path[:,1]
                else:
                    delta=data['goal'][j,:2]-data[initial][j,:2];axis=delta/max(np.linalg.norm(delta),1e-10)
                    perp=np.array([-axis[1],axis[0]])
                    x=(path[:,:2]-data[initial][j,:2])@axis*100;y=(path[:,:2]-data[initial][j,:2])@perp*100
                z=data['z'][j]
                ax.plot(x,y,lw=1.8,color=plt.get_cmap('tab10')(j),ls='-' if data['success'][j] else '--',
                    label=f'z=({z[0]:.2f}, {z[1]:.2f})')
            if environment=='tworoom':
                for lo,hi in [(14,35),(77,147),(189,210)]:ax.add_patch(plt.Rectangle((107,lo),10,hi-lo,color='.7'))
                ax.scatter(*data[initial][0],c='black',s=28,zorder=4);ax.scatter(*data['goal'][0],c='black',marker='*',s=65,zorder=4)
                ax.set(xlim=(10,214),ylim=(10,214),xlabel='X (pixels)',ylabel='Y (pixels)',aspect='equal')
            else:ax.set(xlabel='Transport progress (cm)',ylabel='Lateral cube displacement (cm)')
            ax.grid(alpha=.15);ax.set_title(f'{title}: {int(data["success"].sum())}/8 successful in this scene')
        h,l=axes[0].get_legend_handles_labels();figure.legend(h,l,loc='outside lower center',ncol=4,frameon=False)
        title='TwoRoom' if environment=='tworoom' else 'Cube'
        figure.suptitle(f'{title}: identical initial state, goal and latent codes\nSolid: successful; dashed: failed. First test scene, no outcome filtering.')
        for extension in ['png','pdf']:figure.savefig(out/f'{environment}_full_vs_without_latent.{extension}')
        plt.close(figure)
        evidence[environment]={'scene_rule':'First test context; first8 original Gaussian codes',
            'same_initial_goal_and_codes_exact':True,'full_successes':int(full['success'].sum()),
            'without_latent_successes':int(ablated['success'].sum()),'codes':full['z'].tolist()}
    plt.rcParams.update({'font.size':12})
    figure,axes=plt.subplots(2,2,figsize=(10,6.2),constrained_layout=True)
    for row,environment in enumerate(['tworoom','cube']):
        paths=[]
        for column,method in enumerate(['full','adversarial']):
            ax=axes[row,column];data=all_loaded[environment][method]
            initial='start' if environment=='tworoom' else 'initial_cube'
            for j in range(8):
                n=int(data['lengths'][j]);path=data['positions'][j,:n+1] if environment=='tworoom' else data['poses'][j,:n+1,3:]
                if environment=='tworoom':x,y=path[:,0],path[:,1]
                else:
                    axis=data['goal'][j,:2]-data[initial][j,:2];axis=axis/max(np.linalg.norm(axis),1e-10)
                    relative=path[:,:2]-data[initial][j,:2]
                    x=relative@axis*100;y=relative@np.array([-axis[1],axis[0]])*100
                paths.append(np.stack([x,y],-1))
                ax.plot(x,y,color=plt.get_cmap('tab10')(j),lw=1.6,
                    ls='-' if data['success'][j] else '--',label=f'Code {j+1}')
            if environment=='tworoom':
                for lo,hi in [(14,35),(77,147),(189,210)]:ax.add_patch(plt.Rectangle((107,lo),10,hi-lo,color='.75'))
                ax.scatter(*data[initial][0],c='black',s=20,zorder=4);ax.scatter(*data['goal'][0],c='black',marker='*',s=45,zorder=4)
                ax.set(xlim=(10,214),ylim=(10,214),xlabel='X (pixels)',ylabel='Y (pixels)',aspect='equal')
            else:ax.set(xlabel='Progress (cm)',ylabel='Lateral (cm)')
            label='TwoRoom' if environment=='tworoom' else 'Cube';training='Full' if method=='full' else 'Without latent objectives'
            ax.set_title(f'{label}, {training}: {int(data["success"].sum())}/8 success',fontsize=12);ax.grid(alpha=.15)
        if environment=='cube':
            joined=np.concatenate(paths);lo=joined.min(0);hi=joined.max(0);margin=np.maximum((hi-lo)*.08,.5)
            for ax in axes[row]:ax.set(xlim=(lo[0]-margin[0],hi[0]+margin[0]),ylim=(lo[1]-margin[1],hi[1]+margin[1]))
    handles,labels=axes[0,0].get_legend_handles_labels();figure.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=10)
    figure.suptitle('Same scene and codes within each row; solid: success, dashed: failure',fontsize=12)
    for extension in ['png','pdf']:figure.savefig(out/f'base_full_vs_without_latent_compact.{extension}')
    plt.close(figure)
    evidence['compact_figure']='Code indices identify the matched Gaussian values within each environment; actual values are recorded above. Both Cube columns share axis limits.'
    (out/'figure_provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':main()
