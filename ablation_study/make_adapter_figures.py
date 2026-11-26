"""Matched first-scene skill paths; no success-based scene selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=['tworoom','cube'],required=True)
    p.add_argument('--skill',choices=['keepout','waypoint'],required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    seed=123 if args.environment=='tworoom' else 42
    variants=[('r1','Rank 1'),('r8','Rank 8'),('r16','Rank 16'),('r8_lora_only','Rank 8, LoRA only')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'savefig.dpi':200})
    fig=plt.figure(figsize=(14,4.8) if args.environment=='tworoom' else (14,7),layout='constrained');reference=None;receipt={}
    cube_axes=[];display_points=[]
    for i,(variant,title) in enumerate(variants):
        root=Path('ablation_results/evaluation')/f'evaluate_{args.environment}_{args.skill}_{variant}_s{seed}'
        assert (root/'DONE.json').exists()
        if args.environment=='tworoom':
            d=torch.load(root/'private_quantile_rollouts.pt',map_location='cpu',weights_only=False)
            loaded={k:d[k][:8].numpy() for k in ['positions','lengths','spec','start','goal','new_z','success','base_z']}
            check={k:loaded[k] for k in ['start','goal','spec','new_z','base_z']}
        else:
            rows=json.loads((root/'native_rollouts/records.json').read_text())[:8]
            trajectories=[]
            for j in range(8):
                with np.load(root/f'native_rollouts/rollout_0000_{j:02d}.npz') as d:
                    trajectories.append({k:d[k].copy() for k in ['poses','initial','goal','spec','new_z','base_z','initial_state']})
            loaded={'success':np.array([r['success'] for r in rows]),'new_z':np.stack([t['new_z'] for t in trajectories])}
            check={k:np.stack([t[k] for t in trajectories]) for k in ['initial','goal','spec','new_z','base_z','initial_state']}
        if reference is None:reference=check
        else:
            for k,v in check.items():assert np.array_equal(v,reference[k]),(variant,k)
        if args.environment=='tworoom':
            ax=fig.add_subplot(1,4,i+1)
            for j in range(8):
                path=loaded['positions'][j,:int(loaded['lengths'][j])+1]
                ax.plot(path[:,0],path[:,1],color=plt.get_cmap('tab10')(j),lw=1.3,
                    ls='-' if loaded['success'][j] else '--',label=f'u={float(loaded["new_z"][j,0]):.2f}')
            for lo,hi in [(14,35),(77,147),(189,210)]:ax.add_patch(plt.Rectangle((107,lo),10,hi-lo,color='.75'))
            spec=loaded['spec'][0];color='#bd4444' if args.skill=='keepout' else '#448d56'
            ax.add_patch(plt.Circle(spec[:2],spec[2],color=color,alpha=.18))
            ax.add_patch(plt.Circle(spec[:2],spec[2],fill=False,color=color,lw=1))
            ax.scatter(*loaded['start'][0],s=18,c='black');ax.scatter(*loaded['goal'][0],s=45,c='black',marker='*')
            ax.set(xlim=(10,214),ylim=(10,214),xlabel='X (pixels)',ylabel='Y (pixels)',aspect='equal');ax.grid(alpha=.1)
        else:
            ax=fig.add_subplot(2,4,i+1);lateral_ax=fig.add_subplot(2,4,i+5)
            initial=trajectories[0]['initial'];goal=trajectories[0]['goal']
            axis=goal[:2]-initial[:2];axis=axis/max(np.linalg.norm(axis),1e-10);perp=np.array([-axis[1],axis[0]])
            def frame(path):
                xy=path[...,:2]-initial[:2]
                return np.stack([xy@axis*100,xy@perp*100,path[...,2]*100],-1)
            for j,t in enumerate(trajectories):
                xyz=frame(t['poses'][:,3:]);style='-' if loaded['success'][j] else '--'
                ax.plot(xyz[:,0],xyz[:,2],color=plt.get_cmap('tab10')(j),lw=1.2,
                    ls=style,label=f'u={float(t["new_z"][0]):.2f}')
                lateral_ax.plot(xyz[:,0],xyz[:,1],color=plt.get_cmap('tab10')(j),lw=1.2,ls=style)
                display_points.append(xyz)
            spec=trajectories[0]['spec'];center=frame(spec[:3]);radius=spec[3]*100
            if args.skill=='keepout':radius+=np.sqrt(3)*2
            color='#bd4444' if args.skill=='keepout' else '#448d56'
            for panel,coordinate in [(ax,2),(lateral_ax,1)]:
                panel.add_patch(plt.Circle((center[0],center[coordinate]),radius,color=color,alpha=.12))
                panel.add_patch(plt.Circle((center[0],center[coordinate]),radius,fill=False,color=color,lw=1))
                for point,marker in [(initial,'o'),(goal,'*')]:
                    value=frame(point);panel.scatter(value[0],value[coordinate],c='black',s=20,marker=marker)
                panel.set(xlabel='Progress (cm)',ylabel='Height (cm)' if coordinate==2 else 'Lateral (cm)')
                panel.grid(alpha=.12)
            display_points.append(np.stack([center-radius,center+radius]))
            display_points.append(frame(np.stack([initial,goal])));cube_axes.append((ax,lateral_ax))
        ax.set_title(f'{title}\n{int(loaded["success"].sum())}/8 successful')
        if i==0:
            original_handles,labels=ax.get_legend_handles_labels()
            handles=[Line2D([],[],color=h.get_color(),lw=1.4) for h in original_handles]
        receipt[variant]={'scene':0,'successes':int(loaded['success'].sum()),'codes':loaded['new_z'].tolist(),
            'exact_initial_goal_region_base_z_and_private_codes_match':True,
            'protocol':'Original quantile sweep' if args.environment=='tworoom' else 'Original paired-Gaussian native test'}
    if cube_axes:
        points=np.concatenate(display_points);low=points.min(0);high=points.max(0)
        pad=np.maximum((high-low)*.05,1.);low-=pad;high+=pad
        for height_ax,lateral_ax in cube_axes:
            for panel,coordinate in [(height_ax,2),(lateral_ax,1)]:
                panel.set(xlim=(low[0],high[0]),ylim=(low[coordinate],high[coordinate]))
                panel.set_aspect('equal',adjustable='box')
        receipt['shared_axis_limits_cm']={'minimum':low.tolist(),'maximum':high.tolist(),
            'scope':'Identical physical axis ranges and equal geometric aspect per projection across all four variants; every displayed path and region included.'}
        receipt['projection']='Orthogonal height/progress and lateral/progress views. Circles are sphere silhouettes; success labels use the original native3D criterion.'
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    detail=('Orthogonal projections; success labels use native 3D geometry. ' if cube_axes else '')
    fig.suptitle(f'{args.environment.capitalize()} {args.skill}: first held-out scene, identical inputs\n{detail}Solid: joint success; dashed: failure. '+('Red: keep-out region.' if args.skill=='keepout' else 'Green: waypoint region.'))
    for suffix in ['png','pdf']:fig.savefig(out/f'{args.environment}_{args.skill}_matched_paths.{suffix}')
    plt.close(fig)
    (out/'figure_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
