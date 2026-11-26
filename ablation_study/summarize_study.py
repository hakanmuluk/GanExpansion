"""Presentation tables/figures from finished aggregate evidence only.

No model selection, test filtering, fitting or policy changes occur here.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def training_stats(folder):
    path=folder/'train.jsonl'
    if not path.exists():return {}
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:return {}
    nonfinite=[]
    for row in rows:
        for key,value in row.items():
            if isinstance(value,(int,float)) and not math.isfinite(value):nonfinite.append([row.get('step'),key])
    # Whole-run elapsed time includes the original validation procedure. It is
    # not labeled pure training throughput, and excludes private E calibration.
    last=rows[-1];timed=[r for r in rows if 'elapsed_seconds' in r]
    result={'last_logged_step':last.get('step'),'logged_nonfinite_fields':nonfinite,
        'elapsed_seconds_last_training_log_including_validation':timed[-1]['elapsed_seconds'] if timed else None}
    for name in ['DONE.json','completed.json']:
        if (folder/name).exists():result['completed_run_wall_seconds_including_validation']=read(folder/name).get('elapsed_seconds')
    if (folder/'config.json').exists():
        cfg=read(folder/'config.json');cfg=cfg.get('args',cfg.get('train_config',cfg));every=cfg.get('eval_every')
        windows=[]
        if every:
            for before,after in zip(timed,timed[1:]):
                seconds=after['elapsed_seconds']-before['elapsed_seconds'];steps=after['step']-before['step']
                if seconds>0 and steps>0 and before['step']%every!=0:windows.append(steps/seconds)
        if windows:result['training_intervals_excluding_validation']={'windows':len(windows),
            'median_updates_per_second':float(np.median(windows)),'p25':float(np.quantile(windows,.25)),
            'p75':float(np.quantile(windows,.75)),
            'scope':'Consecutive training-log intervals whose preceding step was not a validation boundary; retains original feedback/physics training work.'}
    for key in ['loss','d_loss','grad_norm']:
        values=[r[key] for r in rows if key in r and isinstance(r[key],(int,float)) and math.isfinite(r[key])]
        if values:result[key]={'last':values[-1],'minimum':min(values),'maximum':max(values)}
    return result


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--partial',action='store_true')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    evaluation=Path('ablation_results/evaluation');base=[];skills=[];stability={};disabled=[];fixed=[]
    status_path=Path('ablation_results/training_queue_v2/status.json')
    queue=read(status_path)['jobs'] if status_path.exists() else {}
    def wall(names):
        rows=[queue.get(n,{}) for n in names]
        return sum(r['finished']-r['started'] for r in rows) if all('finished' in r for r in rows) else None
    for folder in sorted(Path('ablation_results/baseline').glob('disabled_base_*')):
        if not (folder/'DONE.json').exists():continue
        m=read(folder/'summary.json');env,skill=folder.name.removeprefix('disabled_base_').split('_')
        disabled.append({'environment':env,'skill':skill,'success':m['success'],
            'goal_success':m.get('goal_success',m.get('released_success')),
            'episodes':m.get('rollouts',m.get('episodes')),'contexts':m['contexts'],
            'both_successful_strategies':m.get('both_successful_strategies_per_context')})
    for folder in sorted(evaluation.glob('original_gaussian_*')):
        if not (folder/'DONE.json').exists():continue
        row=read(folder/'study_summary.json');m=row['primary_original_gaussian_protocol']
        match=re.fullmatch(r'original_gaussian_(tworoom|cube)_(full|adversarial)_s(\d+)',folder.name)
        env,method,seed=match.groups();base.append({'environment':env,'method':method,'seed':int(seed),
            'success':m.get('success_rate',m.get('released_success')),
            'both_successful_routes':m.get('both_modes_covered_fraction',m.get('both_routes_per_context')),
            'episodes':m.get('rollouts',m.get('attempts')),'contexts':m.get('start_goal_pairs',m.get('contexts')),
            'within_scene_speed_variation':m.get('mean_within_task_speed_std'),
            'speed_spearman':m.get('within_context_speed_spearman'),
            'amplitude_spearman':m.get('within_context_side_amplitude_spearman'),
            'encoder_recovery_applicable':row['encoder_recovery_applicable']})
    for folder in sorted(evaluation.glob('rq1_*')):
        if not (folder/'DONE.json').exists():continue
        match=re.fullmatch(r'rq1_(tworoom|cube_latest)_(full|adversarial)_s(\d+)',folder.name)
        if not match:continue
        env,method,seed=match.groups();env='cube' if env=='cube_latest' else env;m=read(folder/'summary.json')
        unit='pixels' if env=='tworoom' else 'm';geometry=m[f'mean_path_diversity_{unit}']
        consistency_file='latent_consistency_s42.json' if int(seed)==42 else 'latent_consistency_second_seeds.json'
        consistency_path=Path('ablation_results/aggregate')/consistency_file
        c=read(consistency_path)[env][method] if consistency_path.exists() else {}
        fixed.append({'environment':env,'method':method,'seed':int(seed),
            'success':m.get('success_rate',m.get('released_success')),
            'both_successful_routes':m.get('both_modes_covered_fraction',m.get('both_routes_per_context')),
            'path_distance_all':geometry['path_distance_all'],'path_distance_successful':geometry['path_distance_successful'],
            'path_distance_unit':unit,'action_derivative_rms_z1':m['latent_action_derivative_rms'][0],
            'action_derivative_rms_z2':m['latent_action_derivative_rms'][1],
            'mean_successful_route_entropy_bits':c.get('mean_successful_route_entropy_bits_including_no_success_as_zero'),
            'same_code_mode_consistency':c.get('mean_per_code_successful_mode_consistency'),
            'successful_scene_fraction':c.get('scenes_with_any_success'),
            'speed_spearman':m.get('within_context_speed_spearman'),
            'amplitude_spearman':m.get('within_context_side_amplitude_spearman')})
    for folder in sorted(evaluation.glob('evaluate_*')):
        if not (folder/'DONE.json').exists():continue
        match=re.fullmatch(r'evaluate_(tworoom|cube)_(keepout|waypoint)_r(\d+)(_lora_only)?_s(\d+)',folder.name)
        if not match:continue
        env,skill,rank,lora,seed=match.groups();d=read(folder/'summary.json');cfg=read(folder/'config.json');m=d['gaussian']
        row={'environment':env,'skill':skill,'rank':int(rank),'lora_only':bool(lora),'seed':int(seed),
            'success':m['success'],'goal_success':m.get('goal_success',m.get('released_success')),
            'both_successful_strategies':m['both_successful_strategies_per_context'],
            'code_strategy_agreement':m.get('strategy_matches_code'),
            'latent_recovery_rmse':m.get('closed_loop_latent_rmse',m.get('physical_latent_rmse')),
            'latent_recovery_sign_accuracy':m.get('closed_loop_latent_sign_accuracy',m.get('physical_latent_sign_accuracy')),
            'magnitude_response_correlation':m.get('code_magnitude_amount_correlation_successful',m.get('within_context_magnitude_excursion_correlation')),
            'successful_paired_vertical_separation_m':m.get('successful_paired_vertical_separation_m'),
            'private_sign_action_rms':d['functional_checks']['private_sign_action_rms'],
            'disabled_prediction_bitwise_equal':d['functional_checks']['disabled_prediction_bitwise_equal'],
            'adapter_parameters':d['parameters']['action_adapter'],
            'private_encoder_parameters':d['parameters']['private_encoder'],
            'inference_median_ms':d['inference']['median_ms'],
            'episodes':m.get('rollouts',m.get('episodes')),'contexts':m['contexts']}
        if 'quantiles' in d:row['quantile_success']=d['quantiles']['success']
        case=folder.name.removeprefix('evaluate_')
        row['actor_training_wall_seconds_including_validation']=wall([case]+([case+'_initial_1000'] if env=='cube' and skill=='keepout' else []))
        row['private_physical_encoder_procedure_wall_seconds']=wall(['physical_'+case,'calibrate_'+case]) if env=='cube' else None
        row['training_median_updates_per_second_excluding_validation']=training_stats(Path('ablation_results/training')/case).get('training_intervals_excluding_validation',{}).get('median_updates_per_second')
        # Original D architecture is fixed within an environment, independent of
        # LoRA rank, style branch or private E calibration. Counts exclude the
        # spectral-normalization and observation-normalization buffers.
        row['private_discriminator_parameters']=116993 if env=='tworoom' else 285697
        row['actor_training_total_trainable_parameters']=row['adapter_parameters']+row['private_encoder_parameters']+row['private_discriminator_parameters']
        row['online_active_generator_parameters']=cfg['parameters']['frozen_base_generator']+row['adapter_parameters']
        skills.append(row)
    for folder in sorted(Path('ablation_results/training').glob('*')):
        if 'invalid' in folder.name:continue
        if (folder/'controlled_ablation.json').exists():stability[folder.name]=training_stats(folder)
    expected_base=8;expected_skills=48
    if not args.partial:
        assert len(base)==expected_base,(len(base),expected_base)
        assert len(skills)==expected_skills,(len(skills),expected_skills)
        assert len(fixed)==8,len(fixed)
    write_csv(out/'base_results.csv',base);write_csv(out/'adapter_results.csv',skills);write_csv(out/'disabled_base_results.csv',disabled)
    write_csv(out/'latent_diagnostics.csv',fixed)
    (out/'training_stability.json').write_text(json.dumps(stability,indent=2)+'\n')
    summary={'base':base,'fixed_code_diagnostics':fixed,'adapters':skills,'disabled_base_on_skill_banks':disabled,'training_stability':stability,
        'partial':len(base)!=expected_base or len(skills)!=expected_skills,
        'interpretation':{'error_bars':'Minimum and maximum across two training seeds; no population-level confidence claim.',
            'skill_success':'Original held-out joint goal-and-named-skill criterion. TwoRoom strategy coverage additionally requires original doorway style, as in its original evaluator.',
            'generalization':'Original unfiltered held-out context banks; no new task layouts or OOD claim.',
            'efficiency':'CUDA warmed batch1 action-generator latency excludes encoder, simulator and loading; whole-run logs include validation.'}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':200})
    def save(fig,name):
        for suffix in ['png','pdf']:fig.savefig(out/f'{name}.{suffix}')
        plt.close(fig)
    if base:
        fig,axes=plt.subplots(1,2,figsize=(9,3.8),constrained_layout=True)
        for ax,metric,title in zip(axes,['success','both_successful_routes'],['Goal success','Scenes with both successful routes']):
            for i,env in enumerate(['tworoom','cube']):
                for j,method in enumerate(['full','adversarial']):
                    values=np.array([r[metric]*100 for r in base if r['environment']==env and r['method']==method])
                    if not len(values):continue
                    mean=values.mean();ax.bar(i+(j-.5)*.32,mean,.3,color=['#3975ad','#d77c39'][j],label=['Full','Without latent objectives'][j] if i==0 else None)
                    ax.errorbar(i+(j-.5)*.32,mean,yerr=[[mean-values.min()],[values.max()-mean]],fmt='none',c='black',capsize=3)
            ax.set(xticks=[0,1],xticklabels=['TwoRoom','Cube'],ylim=(0,105),ylabel='Percent',title=title);ax.grid(axis='y',alpha=.15)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False,fontsize=10)
        save(fig,'base_training_comparison')
    full=[r for r in skills if not r['lora_only']]
    if full:
        fig,axes=plt.subplots(2,3,figsize=(10.5,5.2),constrained_layout=True)
        for i,env in enumerate(['tworoom','cube']):
            for j,(metric,title,scale) in enumerate([('success','Success (%)',100),('both_successful_strategies','Both strategies (%)',100),('adapter_parameters','Adapter parameters (k)',.001)]):
                ax=axes[i,j]
                for skill,color in [('keepout','#3975ad'),('waypoint','#d77c39')]:
                    selected=[r for r in full if r['environment']==env and r['skill']==skill];ranks=sorted({r['rank'] for r in selected})
                    means=[];low=[];high=[]
                    for rank in ranks:
                        v=np.array([r[metric]*scale for r in selected if r['rank']==rank]);m=v.mean();means.append(m);low.append(m-v.min());high.append(v.max()-m)
                    if ranks:ax.errorbar(ranks,means,yerr=[low,high],marker='o',capsize=3,color=color,label=skill.capitalize())
                ax.set(xscale='log',xticks=[1,2,4,8,16],xticklabels=['1','2','4','8','16'],xlabel='LoRA rank',ylabel=title,title='TwoRoom' if env=='tworoom' else 'Cube');ax.grid(alpha=.15)
                if j<2:ax.set_ylim(0,105)
        axes[0,0].legend(frameon=False);save(fig,'rank_ablation')
        fig,axes=plt.subplots(2,2,figsize=(9,6.5),constrained_layout=True)
        for i,env in enumerate(['tworoom','cube']):
            for j,(metric,title) in enumerate([('training_median_updates_per_second_excluding_validation','Training updates/s (validation excluded)'),('inference_median_ms','Batch1 action-generator latency (ms)')]):
                ax=axes[i,j]
                for skill,color in [('keepout','#3975ad'),('waypoint','#d77c39')]:
                    selected=[r for r in full if r['environment']==env and r['skill']==skill and r.get(metric) is not None];ranks=sorted({r['rank'] for r in selected});means=[];low=[];high=[]
                    for rank in ranks:
                        v=np.array([r[metric] for r in selected if r['rank']==rank]);m=v.mean();means.append(m);low.append(m-v.min());high.append(v.max()-m)
                    if ranks:ax.errorbar(ranks,means,yerr=[low,high],marker='o',capsize=3,color=color,label=skill.capitalize())
                ax.set(xscale='log',xticks=[1,2,4,8,16],xticklabels=['1','2','4','8','16'],xlabel='LoRA rank',ylabel=title,title=env.capitalize());ax.grid(alpha=.15)
        axes[0,0].legend(frameon=False);save(fig,'rank_efficiency')
    if any(r['lora_only'] for r in skills):
        fig,axes=plt.subplots(1,3,figsize=(12,4.2),constrained_layout=True)
        labels=['TwoRoom\nkeep-out','TwoRoom\nwaypoint','Cube\nkeep-out','Cube\nwaypoint']
        for ax,metric,title,scale in zip(axes,['success','both_successful_strategies','adapter_parameters'],['Skill success (%)','Both successful strategies (%)','Action-adapter parameters (thousands)'],[100,100,.001]):
            for i,(env,skill) in enumerate([(e,s) for e in ['tworoom','cube'] for s in ['keepout','waypoint']]):
                for j,lora in enumerate([False,True]):
                    v=np.array([r[metric]*scale for r in skills if r['environment']==env and r['skill']==skill and r['rank']==8 and r['lora_only']==lora])
                    if not len(v):continue
                    m=v.mean();x=i+(j-.5)*.35;ax.bar(x,m,.32,color=['#3975ad','#d77c39'][j],label=['LoRA + skill modulation','LoRA only'][j] if i==0 else None)
                    ax.errorbar(x,m,yerr=[[m-v.min()],[v.max()-m]],fmt='none',c='black',capsize=2)
            ax.set(xticks=np.arange(4),xticklabels=labels,ylabel=title);ax.grid(axis='y',alpha=.15)
            if metric!='adapter_parameters':ax.set_ylim(0,105)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False,fontsize=10)
        save(fig,'modulation_ablation')
    print(json.dumps({'base_rows':len(base),'adapter_rows':len(skills),'partial':summary['partial'],'out':str(out)}),flush=True)


if __name__=='__main__':main()
