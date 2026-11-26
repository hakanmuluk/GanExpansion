"""Execute existing policies at paired initial conditions and save GPU-side plots.

Original task completion rules and action horizons are imported unchanged.
Additional latent sweeps are diagnostics, separate from official test scores.
Raw rollouts stay on the execution host; export only figures/aggregate metrics.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch


def json_write(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def codes(path=None):
    if path:
        values=np.asarray(json.loads(Path(path).read_text()),np.float32)
        if values.ndim!=2 or values.shape[1]!=2 or len(values)<2 or not np.isfinite(values).all():
            raise ValueError('A diagnostic code grid must contain finite two-dimensional latent codes')
        return values
    return np.asarray([[-1,-1],[-1,1],[1,-1],[1,1],[-2,0],[-.25,0],[.25,0],[2,0]], np.float32)


def pairwise_diversity(paths, successes):
    """Mean paired path distance after arc-length interpolation, within scene.

    This describes geometry, not time spent stopped; speed is measured separately.
    Failed and successful-only versions are both retained.
    """
    normalized = []
    for p in paths:
        distance = np.r_[0., np.linalg.norm(np.diff(p, axis=0), axis=-1).cumsum()]
        unique = np.r_[True, np.diff(distance) > 1e-10]
        if distance[-1] <= 1e-10:
            normalized.append(np.repeat(p[:1], 64, axis=0))
        else:
            grid = np.linspace(0, distance[-1], 64)
            normalized.append(np.stack([np.interp(grid, distance[unique], p[unique,j]) for j in range(p.shape[1])], -1))
    all_values, good_values = [], []
    for i in range(len(paths)):
        for j in range(i):
            value = float(np.linalg.norm(normalized[i]-normalized[j], axis=-1).mean())
            all_values.append(value)
            if successes[i] and successes[j]: good_values.append(value)
    return {'path_distance_all': float(np.mean(all_values)) if all_values else None,
            'path_distance_successful': float(np.mean(good_values)) if good_values else None}


def plot_paths(environment, trajectories, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.dpi':200})
    contexts = sorted(trajectories)[:4]
    fig, axes = plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    colors = plt.get_cmap('tab10').colors
    for ax, c in zip(axes.flat, contexts):
        runs = trajectories[c]
        for i,r in enumerate(runs):
            p=r['path'];z=r['z']
            if environment=='tworoom':x,y=p[:,0],p[:,1]
            else:
                axis=(r['goal'][:2]-r['initial'][:2]);axis=axis/max(np.linalg.norm(axis),1e-9)
                perp=np.array([-axis[1],axis[0]])
                x=(p[:,:2]-r['initial'][:2])@axis*100
                y=(p[:,:2]-r['initial'][:2])@perp*100
            ax.plot(x,y,color=colors[i%10],lw=1.6,alpha=.9,
                    linestyle='-' if r['success'] else '--',label=f'z=({z[0]:g}, {z[1]:g})')
        if environment=='tworoom':
            for lo,hi in [(14,35),(77,147),(189,210)]:
                ax.add_patch(plt.Rectangle((107,lo),10,hi-lo,color='.7'))
            ax.scatter(*runs[0]['initial'][:2],c='black',s=25,marker='o',zorder=4)
            ax.scatter(*runs[0]['goal'][:2],c='black',s=60,marker='*',zorder=4)
            ax.set(xlim=(10,214),ylim=(10,214),xlabel='X (pixels)',ylabel='Y (pixels)',aspect='equal')
        else:
            ax.set(xlabel='Transport progress (cm)',ylabel='Lateral displacement (cm)',aspect='equal')
            ax.axhline(0,color='.8',lw=.7)
        ax.set_title(f'Fixed scene {c}: {sum(r["success"] for r in runs)}/{len(runs)} successful')
        ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle(f'{environment}: same initial state and goal, different latent codes\nSolid: successful; dashed: failed')
    fig.savefig(out/'paired_trajectories.png');fig.savefig(out/'paired_trajectories.pdf');plt.close(fig)
    if environment=='cube':
        fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
        for ax,c in zip(axes.flat,contexts):
            for i,r in enumerate(trajectories[c]):
                ax.plot(np.arange(len(r['path']))*.05,r['path'][:,2]*100,color=colors[i%10],
                        label=f'z=({r["z"][0]:g}, {r["z"][1]:g})')
            ax.set(title=f'Fixed scene {c}',xlabel='Time (s)',ylabel='Cube height (cm)');ax.grid(alpha=.15)
        fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
        fig.suptitle('Cube: height and timing under latent interventions')
        fig.savefig(out/'paired_heights.png');fig.savefig(out/'paired_heights.pdf');plt.close(fig)


def analyze_tworoom(args, out):
    from tworoom_gan.evaluate import load_checkpoint, rollout, summarize_rollouts
    g,e,checkpoint=load_checkpoint(args.checkpoint,args.device)
    encoder_available=checkpoint.get('train_config',{}).get('rec_weight',1)!=0
    if not encoder_available:e=None
    with np.load(Path(args.data)/'test.npz') as d:
        _,ids=np.unique(d['pair_id'],return_index=True)
        starts=d['start'][ids];goals=d['goal'][ids]
    count=min(args.contexts or len(ids),len(ids));z=codes(getattr(args,'codes_json',None));k=len(z)
    st=torch.as_tensor(starts[:count],device=args.device).repeat_interleave(k,0)
    goal=torch.as_tensor(goals[:count],device=args.device).repeat_interleave(k,0)
    zz=torch.as_tensor(z,device=args.device).repeat(count,1)
    result=rollout(g,st,goal,zz)
    metric=summarize_rollouts(result,count,k,e,shared_latents=True)
    # These are physical code-control metrics, not encoder-recovery metrics.
    # Compute the original formulas even when the ablation has no usable E.
    from tworoom_gan.style import decode_style
    from tworoom_gan.env import TASK
    mode=result['mode'];ok=result['success'];good=ok&(mode>=0)
    intended_mode,intended_speed,intended_offset=decode_style(result['z'])
    crossed=(result['positions'][:,1:,0]>112)!=(result['positions'][:,:-1,0]>112)
    cross_t=crossed.long().argmax(1)+1
    cross_y=result['positions'][torch.arange(len(ok),device=ok.device),cross_t,1]
    offset=cross_y-result['positions'].new_tensor(TASK.door_y)[mode.clamp_min(0)]
    metric.update({'intended_route_accuracy_successful':float((mode[ok]==intended_mode[ok]).float().mean()) if ok.any() else None,
        'crossing_offset_rmse_pixels_successful':float((offset[good]-intended_offset[good]).square().mean().sqrt()) if good.any() else None,
        'mean_speed_mae_successful':float((result['mean_speed'][ok]-intended_speed[ok]).abs().mean()) if ok.any() else None})
    per_code=[]
    for j,code in enumerate(z):
        chosen=torch.arange(j,len(ok),k,device=ok.device);successful=chosen[ok[chosen]];crossed_ids=chosen[good[chosen]]
        per_code.append({'z':code.tolist(),'episodes':count,'success':float(ok[chosen].float().mean()),
            'mean_successful_steps':float(result['lengths'][successful].float().mean()) if len(successful) else None,
            'mean_successful_speed':float(result['mean_speed'][successful].mean()) if len(successful) else None,
            'mean_successful_crossing_offset':float(offset[crossed_ids].mean()) if len(crossed_ids) else None})
    metric['per_latent_aggregate']=per_code
    paths=result['positions'].cpu().numpy();lens=result['lengths'].cpu().numpy();success=result['success'].cpu().numpy()
    grouped={};diversity=[]
    for c in range(count):
        p=[paths[c*k+j,:lens[c*k+j]+1] for j in range(k)]
        diversity.append(pairwise_diversity(p,success[c*k:(c+1)*k]))
        if c<4:grouped[c]=[{'path':p[j],'z':z[j],'success':bool(success[c*k+j]),'initial':starts[c],'goal':goals[c]} for j in range(k)]
    # Actions at exactly the same observation: finite-difference latent sensitivity.
    state=result['states'][::k,0];base=torch.ones((count,2),device=args.device)
    sensitivity=[]
    with torch.no_grad():
        for j in range(2):
            plus=base.clone();minus=base.clone();plus[:,j]+=.01;minus[:,j]-=.01
            sensitivity.append(float(((g(state,plus)-g(state,minus))/.02).square().mean().sqrt()))
    metric.update({'sweep_codes':z.tolist(),'latent_action_derivative_rms':sensitivity,
                   'encoder_recovery_applicable':encoder_available,
                   'mean_path_diversity_pixels':{key:float(np.mean([x[key] for x in diversity if x[key] is not None])) if any(x[key] is not None for x in diversity) else None for key in diversity[0]}})
    json_write(out/'summary.json',metric);plot_paths('tworoom',grouped,out)
    # Export only aggregated quantities; retain detailed tensors on GPU host.
    torch.save(result,out/'private_rollouts.pt')


def analyze_cube(args,out):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from cube.cube_gan.evaluate import init_worker,rollout,summarize,select_contexts,load_models
    with np.load(Path(args.data)/'test_contexts.npz') as d:contexts={k:d[k] for k in d.files}
    count=min(args.contexts or len(contexts['seed']),len(contexts['seed']));selected=select_contexts(contexts['task_id'],count)
    z=codes(getattr(args,'codes_json',None));specs=[]
    for c in selected:
        for j,code in enumerate(z):
            specs.append({**{key:value[c] for key,value in contexts.items()},'context':int(c),'latent_index':j,
                          'z':code,'rollout_seed':8291+int(c)*100+j,'execute':8})
    records=[];grouped={};paths={};successes={};state_rows=[];initial_observations={}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(args.checkpoint,True)) as pool:
        for i,(row,traj) in enumerate(pool.map(rollout,specs,chunksize=1)):
            c=row['context'];n=int(traj['lengths']);p=traj['poses'][:n+1,3:]
            if c in initial_observations:
                assert np.array_equal(traj['states'][0],initial_observations[c]),'Latent interventions must share the initial observation'
            else:initial_observations[c]=traj['states'][0].copy()
            row['peak_cube_height_m']=float(p[:,2].max());records.append(row)
            paths.setdefault(c,[]).append(p);successes.setdefault(c,[]).append(row['released_success'])
            if c in selected[:4]:grouped.setdefault(c,[]).append({'path':p,'z':np.asarray(row['z']),
                'success':row['released_success'],'initial':traj['initial_cube'],'goal':traj['goal']})
            if row['latent_index']==0:state_rows.append(traj['states'][0])
            if (i+1)%40==0:print(json.dumps({'completed':i+1,'total':len(specs)}),flush=True)
    metric=summarize(records);diversity=[pairwise_diversity(paths[c],successes[c]) for c in paths]
    metric['per_latent_aggregate']=[]
    for j,code in enumerate(z):
        chosen=[r for r in records if r['latent_index']==j];successful=[r for r in chosen if r['released_success']]
        metric['per_latent_aggregate'].append({'z':code.tolist(),'episodes':len(chosen),
            'released_success':float(np.mean([r['released_success'] for r in chosen])),
            'mean_successful_steps':float(np.mean([r['steps'] for r in successful])) if successful else None,
            'mean_successful_abs_carry_amplitude_m':float(np.mean([abs(r['carry_amplitude']) for r in successful])) if successful else None,
            'mean_successful_peak_height_m':float(np.mean([r['peak_cube_height_m'] for r in successful])) if successful else None})
    g,_,checkpoint=load_models(args.checkpoint,device=args.device)
    encoder_available=checkpoint.get('args',{}).get('rec_weight',1)!=0 and not checkpoint.get('ablation_config',{}).get('base_without_latent_losses',False)
    state=torch.as_tensor(np.stack(state_rows),device=args.device);base=torch.ones((len(state),2),device=args.device)
    sensitivity=[]
    with torch.no_grad():
        for j in range(2):
            plus=base.clone();minus=base.clone();plus[:,j]+=.01;minus[:,j]-=.01
            sensitivity.append(float(((g(state,plus)-g(state,minus))/.02).square().mean().sqrt()))
    if not encoder_available:metric.pop('closed_loop_latent_rmse',None)
    metric.update({'sweep_codes':z.tolist(),'latent_action_derivative_rms':sensitivity,'encoder_recovery_applicable':encoder_available,
        'initial_observations_exactly_matched_across_codes':True,
        'mean_path_diversity_m':{key:float(np.mean([x[key] for x in diversity if x[key] is not None])) if any(x[key] is not None for x in diversity) else None for key in diversity[0]},
        'height_range_m':{'mean_successful_peak':float(np.mean([p[:,2].max() for c in paths for p,ok in zip(paths[c],successes[c]) if ok])) if any(any(x) for x in successes.values()) else None}})
    json_write(out/'summary.json',metric);json_write(out/'private_episode_records.json',records)
    plot_paths('cube',grouped,out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=['tworoom','cube'],required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--contexts',type=int,default=0);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--codes-json',help='Optional supplementary diagnostic grid; default eight-code sweep is unchanged')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.time();json_write(out/'config.json',vars(args))
    (analyze_tworoom if args.environment=='tworoom' else analyze_cube)(args,out)
    json_write(out/'DONE.json',{'completed':True,'seconds':time.time()-started})


if __name__=='__main__':main()
