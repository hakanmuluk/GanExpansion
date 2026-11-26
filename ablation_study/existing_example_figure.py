"""A compact report figure from the first fixed scene of each original model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .latent_analysis import codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'savefig.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),constrained_layout=True)
    tt=torch.load('ablation_results/rq1_tworoom_existing/private_rollouts.pt',map_location='cpu',weights_only=False)
    for j,z in enumerate(codes()):
        path=tt['positions'][j,:int(tt['lengths'][j])+1].numpy()
        axes[0].plot(*path.T,color=plt.get_cmap('tab10')(j),lw=1.7,label=f'z=({z[0]:g}, {z[1]:g})')
    for lo,hi in [(14,35),(77,147),(189,210)]:axes[0].add_patch(plt.Rectangle((107,lo),10,hi-lo,color='.75'))
    axes[0].scatter(*tt['start'][0].numpy(),c='black',s=20);axes[0].scatter(*tt['goal'][0].numpy(),c='black',s=50,marker='*')
    axes[0].set(xlim=(10,214),ylim=(10,214),aspect='equal',xlabel='X (pixels)',ylabel='Y (pixels)',title='TwoRoom: original checkpoint')
    from cube.cube_gan.evaluate import init_worker,rollout
    checkpoint='ablation_inputs/cube_latest_base.pt';init_worker(checkpoint,True)
    with np.load('cube/artifacts/data/test_contexts.npz') as data:scene={k:data[k][0] for k in data.files}
    successes=[]
    for j,z in enumerate(codes()):
        record,traj=rollout({**scene,'context':0,'latent_index':j,'z':z,'rollout_seed':8291+j,'execute':8})
        path=traj['poses'][:int(traj['lengths'])+1,3:];axis=traj['goal'][:2]-traj['initial_cube'][:2];axis/=max(np.linalg.norm(axis),1e-10)
        xy=path[:,:2]-traj['initial_cube'][:2];lateral=np.array([-axis[1],axis[0]])
        linestyle='-' if record['released_success'] else '--'
        axes[1].plot(xy@axis*100,xy@lateral*100,color=plt.get_cmap('tab10')(j),lw=1.7,ls=linestyle)
        axes[2].plot(np.arange(len(path))*.05,path[:,2]*100,color=plt.get_cmap('tab10')(j),lw=1.7,ls=linestyle)
        successes.append(record['released_success'])
    axes[1].set(xlabel='Progress (cm)',ylabel='Lateral (cm)',title='Cube: original latest checkpoint')
    axes[2].set(xlabel='Time (s)',ylabel='Cube height (cm)',title='Cube: height and timing')
    for ax in axes:ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=10)
    fig.suptitle('Fixed first test scene in each environment; only the latent code changes',fontsize=13)
    for suffix in ['png','pdf']:fig.savefig(out/f'existing_latent_examples.{suffix}')
    plt.close(fig)
    provenance={'scene_selection':'First test context, without outcome filtering','codes':codes().tolist(),
        'tworoom_original_successes':int(tt['success'][:8].sum()),'cube_original_successes':int(sum(successes)),
        'cube_checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'scope':'Actual native Cube paths regenerated using the exact diagnostic scene/codes; TwoRoom paths reuse the original fixed-code evaluation. No training.'}
    (out/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':main()
