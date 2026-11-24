"""Recorded comparisons, context-level uncertainty, and research figures."""
import argparse,json,glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from matplotlib.animation import FuncAnimation,PillowWriter


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [float(c-half),float(c+half)]


def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None


def environment(ax,data,i):
    ax.set(xlim=(0,224),ylim=(224,0),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
    for box in [(0,0,224,14),(0,210,224,14),(0,0,14,224),(210,0,14,224),(107,14,10,21),(107,77,10,70),(107,189,10,21)]:
        ax.add_patch(Rectangle(box[:2],*box[2:],color='#555'))
    x,y,r=data['keepout'][i];ax.add_patch(Circle((x,y),r,color='crimson',alpha=.5));ax.add_patch(Circle((x,y),r+7,fill=False,color='crimson',ls=':'))
    x,y,r=data['waypoint'][i];ax.add_patch(Circle((x,y),r,color='gold',alpha=.7));ax.scatter(*data['goal'][i],marker='*',color='green',s=90)
    ax.add_patch(Circle(data['goal'][i],16,fill=False,color='green',ls=':'));ax.scatter(*data['start'][i],color='black',s=15)


def aggregate(root):
    records=[]
    for path in sorted(root.glob('*/*/summary.json')):
        run=path.parent;summary=json.loads(path.read_text());rows=json.loads((run/'records.json').read_text());cfg=json.loads((run/'config.json').read_text())
        records.append({'run':str(run),'bank':run.parent.name,'name':run.name,'config':cfg['search'],'summary':summary,
            'success_count':sum(r['success'] for r in rows),'success_wilson95':wilson(sum(r['success'] for r in rows),len(rows)),
            'disagreement_success_correlation':corr([r['disagreement_mean'] for r in rows],[r['success'] for r in rows])})
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--benchmark');p.add_argument('--capacity');p.add_argument('--visual-root');args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    allrecords=[]
    for root in args.roots:allrecords.extend(aggregate(Path(root)))
    (out/'comparison.json').write_text(json.dumps(allrecords,indent=2))
    columns=['root','bank','method','n','success_count','success','wilson_low','wilson_high','goal','keepout_violation','early_goal','wall_violation','disagreement','batch_plan_seconds']
    lines=[','.join(columns)]
    for r in allrecords:
        s=r['summary'];lines.append(','.join(map(str,[Path(r['run']).parents[1].name,r['bank'],r['name'],s['episodes'],r['success_count'],s['success'],*r['success_wilson95'],s['goal_success'],s['keepout_violation'],s['early_goal_entry'],s['strict_wall_violation'],s['disagreement_mean'],s['planning_batch_seconds_mean']])))
    (out/'comparison.csv').write_text('\n'.join(lines)+'\n')
    # Validation ablation plots are explicitly separated from held-out final figures.
    for root in args.roots:
        selected=[r for r in allrecords if Path(r['run']).parents[1]==Path(root)]
        if not selected:continue
        methods=list(dict.fromkeys(r['name'] for r in selected));banks=list(dict.fromkeys(r['bank'] for r in selected))
        matrix=np.full((len(methods),len(banks)),np.nan)
        for r in selected:matrix[methods.index(r['name']),banks.index(r['bank'])]=r['summary']['success']
        fig,ax=plt.subplots(figsize=(max(7,len(banks)*2),max(5,len(methods)*.3)))
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_yticks(range(len(methods)),methods,fontsize=8);ax.set_xticks(range(len(banks)),banks,rotation=25,ha='right',fontsize=8)
        for j in range(len(methods)):
            for k in range(len(banks)):
                if np.isfinite(matrix[j,k]):ax.text(k,j,f'{100*matrix[j,k]:.1f}',ha='center',va='center',fontsize=7,color='white' if matrix[j,k]<.6 else 'black')
        fig.colorbar(im,ax=ax,label='Strict joint success');ax.set_title(Path(root).name+' (separate bank denominators)');fig.tight_layout();fig.savefig(out/(Path(root).name+'_success.png'),dpi=180);plt.close(fig)
        if Path(root).name=='validation':
            fig,axes=plt.subplots(1,3,figsize=(13,4))
            for method,label in [('adam','Multistart Adam, H32'),('hybrid','Sample + gradient, H32')]:
                subset=[r for r in selected if r['config']['method']==method and r['config']['objective']=='task']
                steps=sorted(set(r['config']['steps'] for r in subset))
                if steps:axes[0].plot(steps,[np.mean([r['summary']['success'] for r in subset if r['config']['steps']==s]) for s in steps],'o-',label=label)
            horizon=[r for r in selected if r['name'] in ['structured_task_h16','structured_task_h32','structured_task_h64']]
            hs=sorted(set(r['config']['lookahead'] for r in horizon))
            axes[1].plot(hs,[np.mean([r['summary']['success'] for r in horizon if r['config']['lookahead']==h]) for h in hs],'o-')
            temporals=['fixed','cold','warm','tail'];names=['structured_h64_fixed','structured_h64_cold','structured_task_h64','structured_h64_tail']
            rates=[np.mean([r['summary']['success'] for r in selected if r['name']==name]) for name in names]
            axes[2].bar(temporals,rates)
            for ax in axes:ax.set_ylim(0,1.04);ax.set_ylabel('Validation strict success');ax.grid(alpha=.2)
            axes[0].set_xlabel('Gradient refinement steps');axes[0].legend(fontsize=7);axes[1].set_xlabel('Prediction horizon (controls)');axes[2].set_xlabel('Temporal strategy, H64')
            fig.tight_layout();fig.savefig(out/'validation_steps_horizon_temporal.png',dpi=180);plt.close(fig)
    # Success versus synchronized single-context planning latency, never batch time / B.
    if args.benchmark:
        latency=json.loads(Path(args.benchmark).read_text())['results'];timing={r['name']:r for r in latency}
        main=[r for r in allrecords if Path(r['run']).parents[1].name=='main_seed42' and 'supported' in r['bank']]
        fig,ax=plt.subplots(figsize=(11,6))
        labels={'naive_sum':'Fixed summed adapters','agreement_adam_k':'Agreement Adam, execute keep-out',
            'agreement_random_k64':'Agreement sampling64, execute keep-out','agreement_grid_sum':'Agreement grid, summed execution',
            'structured_task_h64':'Task sampling16 H64, CUDA graph','task_cem16x2_h64':'Task CEM2x16 H64, CUDA graph',
            'task_adam4x4_h32':'Task Adam4x4 H32','action_adam10_h64':'Direct action Adam10 H64',
            'structured_task_h16':'Task sampling16 H16, CUDA graph'}
        for name in labels:
            if name not in timing:continue
            subset=[r for r in main if r['name']==name];success=sum(r['success_count'] for r in subset)/sum(r['summary']['episodes'] for r in subset)
            x=1000*timing[name]['seconds_p50'];ax.scatter(x,success,label=labels[name],s=45)
        ax.set(xscale='log',xlabel='Batch-one planning latency p50 (ms, CUDA synchronized)',ylabel='Held-out strict success, model seed42',ylim=(-.03,1.04));ax.grid(alpha=.2);ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),fontsize=8);fig.tight_layout();fig.savefig(out/'success_vs_latency.png',dpi=180);plt.close(fig)
    # Within-context analysis across all fixed code pairs, holding executor fixed.
    capacity=[]
    if args.capacity:
        for path in sorted(Path(args.capacity).glob('*_sum_records.json')):
            rows=json.loads(path.read_text());bins=[[] for _ in range(10)];context_stats=[]
            for context in sorted(set(r['context'] for r in rows)):
                subset=[r for r in rows if r['context']==context];order=np.argsort([r['disagreement_initial'] for r in subset]);success=np.array([r['success'] for r in subset])
                for j,ids in enumerate(np.array_split(order,10)):bins[j].extend(success[ids].tolist())
                context_stats.append({'context':context,'any_success':bool(success.any()),'minimum_disagreement':min(r['disagreement_initial'] for r in subset),
                    'minimum_agreement_pair_success':bool(success[order[0]]),'within_context_correlation':corr([r['disagreement_initial'] for r in subset],success)})
            means=[float(np.mean(b)) for b in bins];record={'bank':path.name,'decile_success':means,'contexts':context_stats}
            bank_name=path.name.replace('_sum_records.json','')
            dynamic=[r for r in allrecords if r['bank']==bank_name and r['name']=='structured_task_h64' and (Path(r['run']).parents[1].name.startswith('main_seed') or Path(r['run']).parents[1].name=='generalization_seed42')]
            if dynamic:
                observed=json.loads((Path(dynamic[0]['run'])/'records.json').read_text())[:len(context_stats)]
                fixed=np.array([r['any_success'] for r in context_stats]);adaptive=np.array([r['success'] for r in observed])
                record['matched_context_capability_comparison']={'contexts':len(fixed),'fixed_grid_oracle_success':float(fixed.mean()),
                    'adaptive_primary_success':float(adaptive.mean()),'fixed_grid_has_solution_but_primary_failed':int((fixed&~adaptive).sum()),
                    'primary_succeeded_without_fixed_grid_solution':int((~fixed&adaptive).sum())}
            capacity.append(record)
        (out/'agreement_capacity_analysis.json').write_text(json.dumps(capacity,indent=2))
        fig,ax=plt.subplots()
        for r in capacity:ax.plot(np.arange(1,11),r['decile_success'],'o-',label=r['bank'].replace('_sum_records.json',''))
        ax.set(xlabel='Within-context initial disagreement decile (1 = lowest)',ylabel='Strict success of fixed-code summed execution',ylim=(-.03,1.03));ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'success_vs_disagreement.png',dpi=180);plt.close(fig)
    if args.visual_root:
        root=Path(args.visual_root);names=['base','keepout_only','waypoint_only','naive_sum','agreement_adam_k','structured_task_h64','action_adam10_h64']
        for bank in ['test_supported_easy','test_supported_conflict','test_coupled_ood']:
            available=[name for name in names if (root/bank/name/'trajectories.npz').exists()]
            if not available:continue
            data={name:np.load(root/bank/name/'trajectories.npz') for name in available};ref=data[available[0]]
            selected=list(range(min(3,len(ref['start']))))
            allrows={name:json.loads((root/bank/name/'records.json').read_text()) for name in available}
            if 'structured_task_h64' in allrows:
                failures=[i for i,r in enumerate(allrows['structured_task_h64']) if not r['success']]
                if failures and failures[0] not in selected:selected.append(failures[0])
            for i in selected:
                fig,axes=plt.subplots(1,len(available),figsize=(3*len(available),3.5));axes=np.atleast_1d(axes)
                for ax,name in zip(axes,available):
                    d=data[name];environment(ax,d,i);length=int(d['lengths'][i]);pp=d['positions'][i,:length+1];ax.plot(pp[:,0],pp[:,1],lw=1.5)
                    row=allrows[name][i];reason='success' if row['success'] else ', '.join(label for key,label in [('strict_wall_violation','wall'),('keepout_violation','keep-out'),('early_goal_entry','early goal')] if row[key]) or 'missing goal/waypoint'
                    ax.set_title(name+'\n'+reason,fontsize=8)
                selection='first three contexts' if i<3 else 'first primary-method failure'
                fig.suptitle(bank+f' context {i}; '+selection);fig.tight_layout();fig.savefig(out/f'{bank}_trajectories_{i}.png',dpi=170);plt.close(fig)
            i=0;fig,ax=plt.subplots(figsize=(6,6));environment(ax,ref,i);colors=plt.cm.tab10(np.linspace(0,1,len(available)));artists=[]
            for name,color in zip(available,colors):
                line,=ax.plot([],[],label=name,color=color,lw=1.5);dot,=ax.plot([],[],'o',color=color,ms=5);artists.append((line,dot))
            ax.legend(loc='upper right',fontsize=6);ax.set_title(bank+' context 0; actual executed positions')
            def animate(t):
                result=[]
                for name,(line,dot) in zip(available,artists):
                    d=data[name];end=min(t,int(d['lengths'][i]));pp=d['positions'][i,:end+1];line.set_data(pp[:,0],pp[:,1]);dot.set_data([pp[-1,0]],[pp[-1,1]]);result.extend([line,dot])
                return result
            FuncAnimation(fig,animate,frames=range(0,241,3),interval=80,blit=True).save(out/(bank+'.gif'),writer=PillowWriter(fps=12));plt.close(fig)
    (out/'DONE.json').write_text(json.dumps({'complete':True,'runs':len(allrecords)}))


if __name__=='__main__':main()
