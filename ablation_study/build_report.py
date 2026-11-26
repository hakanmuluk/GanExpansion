"""Generate editable LaTeX from verified aggregate evidence, with no score edits.

Final mode requires the complete control audit. Layout drafts are marked clearly
and cannot be mistaken for a completed report by this builder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics


def read(path):return json.loads(Path(path).read_text())


def numbers(rows,key,**match):
    return [r[key] for r in rows if all(r.get(k)==v for k,v in match.items()) and r.get(key) is not None]


def mean(rows,key,**match):
    values=numbers(rows,key,**match)
    return statistics.mean(values) if values else None


def fmt(value,scale=1,places=1):
    return r'\textemdash' if value is None else f'{value*scale:.{places}f}'


def interval(rows,key,**match):
    values=numbers(rows,key,**match)
    if not values:return r'\textemdash'
    return f'{statistics.mean(values)*100:.1f} [{min(values)*100:.1f}, {max(values)*100:.1f}]'


def table(headers,rows,alignment=None,note=None):
    align=alignment or 'l'+'r'*(len(headers)-1)
    result=[r'\begin{center}',r'\begin{tabular}{'+align+r'}\toprule',
        ' & '.join(headers)+r'\\\midrule']
    result.extend(' & '.join(str(v) for v in row)+r'\\' for row in rows)
    result.extend([r'\bottomrule\end{tabular}',r'\end{center}'])
    if note:result.append(r'{\small '+note+r'\par}')
    return '\n'.join(result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--results-root',required=True)
    p.add_argument('--audit');p.add_argument('--out',required=True);p.add_argument('--analysis-text')
    p.add_argument('--draft-layout',action='store_true');args=p.parse_args()
    summary=read(args.summary);root=Path(args.results_root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    audit=read(args.audit) if args.audit else {}
    if not args.draft_layout:
        assert not summary['partial'] and len(summary['base'])==8 and len(summary['adapters'])==48
        assert len(summary['fixed_code_diagnostics'])==8
        assert audit.get('complete') and not audit['failures'] and not audit['pending'],'Final control audit is required'
    rows=summary['base'];skills=summary['adapters'];fixed=summary.get('fixed_code_diagnostics',[])
    replacements={};existing=[]
    for env,label in [('tworoom','TwoRoom'),('cube','Cube')]:
        d=read(root/f'rq1_{env}_existing/summary.json')
        existing.append([label,d.get('rollouts',d.get('attempts')),fmt(d.get('success_rate',d.get('released_success')),100),
            fmt(d.get('both_modes_covered_fraction',d.get('both_routes_per_context')),100),
            fmt(d.get('intended_route_accuracy_successful',d.get('route_matches_latent_given_success')),100)])
    replacements['EXISTING_TABLE']=table(['Environment','Rollouts',r'Success (\%)',r'Both routes (\%)',r'Code/route (\%)'],existing,
        note='These are fixed-code diagnostics of unchanged existing checkpoints; they are distinct from the Gaussian tests below.')
    speed_path=root/'z2_magnitude_existing/summary.json'
    replacements['SPEED_SWEEP_NOTE']=''
    if speed_path.exists():
        speed=read(speed_path)['results'];contrasts=[]
        for env,label in [('tworoom','TwoRoom'),('cube','Cube')]:
            d=speed[env]['paired_speed_effect']
            contrasts.append(f'{d["faster_at_z2_plus2_count"]}/{d["paired_successful_contexts"]} {label} scenes')
        replacements['SPEED_SWEEP_NOTE']=(r'A supplementary seven-value sweep holds $z_1=1$ fixed. '
            r'The $z_2=2$ code finishes sooner than $z_2=-2$ in '+(' and '.join(contrasts))+
            r' where both endpoint codes succeed. Cube success changes from '+
            fmt(speed['cube']['per_code'][0]['success'],100)+r'\% to '+
            fmt(speed['cube']['per_code'][-1]['success'],100)+r'\%; '+
            (r'TwoRoom stays at 100\%.' if all(r['success']==1 for r in speed['tworoom']['per_code']) else
             r'TwoRoom endpoint success changes from '+fmt(speed['tworoom']['per_code'][0]['success'],100)+
             r'\% to '+fmt(speed['tworoom']['per_code'][-1]['success'],100)+r'\%.')+
            r' Per-code success and timing curves are supplied separately.')
    elif not args.draft_layout:raise AssertionError('Supplementary speed-axis magnitude diagnostic is required')
    data=[]
    for env,label in [('tworoom','TwoRoom'),('cube','Cube')]:
        for method,name in [('full','Full'),('adversarial','Without latent objectives')]:
            data.append([label,name,interval(rows,'success',environment=env,method=method),interval(rows,'both_successful_routes',environment=env,method=method)])
    replacements['BASE_TABLE']=table(['Environment','Training',r'Success (\%)',r'Both routes (\%)'],data,
        note='Mean [minimum, maximum] across two training seeds; all test contexts and evaluation code seeds are matched.')
    descriptors=[]
    for env,label in [('tworoom','TwoRoom'),('cube','Cube')]:
        full=mean(rows,'success',environment=env,method='full');removed=mean(rows,'success',environment=env,method='adversarial')
        both_full=mean(rows,'both_successful_routes',environment=env,method='full');both_removed=mean(rows,'both_successful_routes',environment=env,method='adversarial')
        if all(v is not None for v in [full,removed,both_full,both_removed]):
            descriptors.append(f'{label}: success {full*100:.1f} to {removed*100:.1f}\\%, and successful two-route coverage {both_full*100:.1f} to {both_removed*100:.1f}\\%.')
    replacements['RQ1_CONCLUSION']=' '.join(descriptors)+r' The comparison supports the combined scheme for successful multimodal control under the original budgets. The semantic axes are partly supervised by known synthetic style labels; this is not evidence of unsupervised disentanglement.'
    stability=summary.get('training_stability',{});bad={n:d['logged_nonfinite_fields'] for n,d in stability.items() if d.get('logged_nonfinite_fields')}
    replacements['STABILITY_PARAGRAPH']=(r'All completed base runs retained their original budgets and logged finite training aggregates. Finite losses do not establish convergence; actual held-out success and diversity are the relevant outcomes.' if not bad else r'Nonfinite logged fields occurred; see the training-stability table. They are retained as observed failures.')
    diagnostic_rows=[]
    for env,label in [('tworoom','TwoRoom'),('cube','Cube')]:
        for method,name in [('full','full'),('adversarial','removed')]:
            match={'environment':env,'method':method}
            distance=fmt(mean(fixed,'path_distance_successful',**match),100 if env=='cube' else 1)
            diagnostic_rows.append([label+' '+name,fmt(mean(fixed,'mean_successful_route_entropy_bits',**match),places=3),
                fmt(mean(fixed,'same_code_mode_consistency',**match),100),distance+(' cm' if env=='cube' else ' px'),
                fmt(mean(fixed,'action_derivative_rms_z1',**match),places=3),fmt(mean(fixed,'action_derivative_rms_z2',**match),places=3)])
    replacements['DIAGNOSTICS_TABLE']=table(['Fixed-code diagnostic','Entropy',r'Consistency (\%)','Path distance',r'$\partial A/\partial z_1$',r'$\partial A/\partial z_2$'],diagnostic_rows,
        note=r'Two-seed means. Route entropy is in bits and is zero for scenes without success. Consistency allows arbitrary relabeling of physical routes; high consistency alone can also indicate collapse. Path distance uses arc-length-aligned successful pairs. Sensitivity is the RMS action derivative at $z=(1,1)$, finite-difference step $.01$, on matched initial observations; compare sensitivities within each environment.')
    tasks=[('tworoom','keepout','TwoRoom K'),('tworoom','waypoint','TwoRoom W'),('cube','keepout','Cube K'),('cube','waypoint','Cube W')]
    rank_rows=[]
    disabled=summary.get('disabled_base_on_skill_banks',[])
    if disabled:
        rank_rows.append(['Base (off)']+[fmt(mean(disabled,'success',environment=e,skill=s),100)+r' / '+fmt(mean(disabled,'both_successful_strategies',environment=e,skill=s),100) for e,s,_ in tasks])
    for rank in [1,2,4,8,16]:
        rank_rows.append([rank]+[fmt(mean(skills,'success',environment=e,skill=s,rank=rank,lora_only=False),100)+r' / '+fmt(mean(skills,'both_successful_strategies',environment=e,skill=s,rank=rank,lora_only=False),100) for e,s,_ in tasks])
    replacements['RANK_TABLE']=table(['Rank']+[t[2] for t in tasks],rank_rows,
        note=r'Cells give joint skill success / contexts with both successful strategies (\%). Adapter values are two-seed means; base-off is the unchanged frozen policy. K: keep-out; W: waypoint. TwoRoom strategy coverage additionally requires its original doorway style.')
    ranges=[]
    for env,skill,label in tasks:
        rates=[mean(skills,'success',environment=env,skill=skill,rank=r,lora_only=False) for r in [1,2,4,8,16]]
        if all(x is not None for x in rates):ranges.append(f'{label}: {min(rates)*100:.1f}--{max(rates)*100:.1f}\\%')
    replacements['RQ2_CONCLUSION']=('Mean success across the five ranks ranges as follows: '+('; '.join(ranges))+'. ' if ranges else '')+r'Rank is tested within an architecture whose style branch retains fixed capacity. Any saturation therefore does not establish that equally small pure LoRA updates alone are sufficient. Two-seed ranges and behavioral coverage must accompany the success means.'
    modulation=[]
    for env,skill,label in tasks:
        def pair(key,scale=100):
            return fmt(mean(skills,key,environment=env,skill=skill,rank=8,lora_only=False),scale)+r' $\to$ '+fmt(mean(skills,key,environment=env,skill=skill,rank=8,lora_only=True),scale)
        modulation.append([label,pair('success'),pair('both_successful_strategies'),pair('latent_recovery_sign_accuracy'),pair('adapter_parameters',.001)])
    replacements['MODULATION_TABLE']=table(['Task',r'Success (\%)',r'Both (\%)',r'E sign (\%)','Params (k)'],modulation,
        note=r'Full rank8 $\to$ pure LoRA-only rank8; two-seed means. E sign is held-out closed-loop private-code recovery, with the original physical calibration for Cube.')
    only=[r for r in skills if r['lora_only']]
    if only:
        assert all(r['private_sign_action_rms']==0 for r in only)
        replacements['RQ3_CONCLUSION']=r'Every completed LoRA-only model has exactly zero action change when the private-code sign is flipped at matched observed states. Its conditional private-code pathway is absent by construction. Actual skill success and strategy coverage above distinguish any remaining default behavior from controllable multimodality.'
    else:replacements['RQ3_CONCLUSION']=r'\textbf{Draft: branch-removal evaluations are pending.}'
    preservation=all(r['disabled_prediction_bitwise_equal'] for r in skills)
    replacements['PRESERVATION_PARAGRAPH']=(r'Every evaluated model passes exact disabled-action equality with its frozen base. TwoRoom additionally checks complete original-goal trajectories; Cube checks two fixed native state/action/pose trajectories per model against the common original-base control.' if skills and preservation else r'\textbf{Draft: complete per-model preservation evidence is pending.}')
    replacements['AUDIT_PARAGRAPH']=(f'The final audit verifies {audit["immutable_input_files_checked"]:,} unchanged input files, configuration equality and every completed training budget; all requested removed objectives are zero in the corrected logs.' if audit.get('complete') else r'\textbf{Draft: the full execution audit is pending.}')
    if args.analysis_text:replacements.update(read(args.analysis_text))
    source=Path(__file__).with_name('report_template.tex').read_text()
    for name,value in replacements.items():source=source.replace('@@'+name+'@@',value)
    assert not re.search(r'@@\w+@@',source),'Unfilled report placeholders'
    figures=out/'figures';figures.mkdir(exist_ok=True);missing=[]
    sources={
        'existing_latent_examples.pdf':root/'existing_latent_examples/existing_latent_examples.pdf',
        'base_full_vs_without_latent_compact.pdf':root/'base_side_by_side_compact_s42/base_full_vs_without_latent_compact.pdf',
        'tworoom_full_vs_without_latent.pdf':root/'base_side_by_side_figures_s42/tworoom_full_vs_without_latent.pdf',
        'cube_full_vs_without_latent.pdf':root/'base_side_by_side_figures_s42/cube_full_vs_without_latent.pdf'}
    sources['latent_speed_magnitude.pdf']=root/'z2_magnitude_existing/latent_speed_magnitude.pdf'
    for name in ['base_training_comparison.pdf','rank_ablation.pdf','modulation_ablation.pdf','rank_efficiency.pdf']:
        sources[name]=Path(args.summary).parent/name
    for name,path in sources.items():
        if path.exists():shutil.copy2(path,figures/name)
        else:
            missing.append(str(path))
            assert args.draft_layout,('Required figure missing',path)
            source=re.sub(r'\\includegraphics\[[^\]]*\]\{figures/'+re.escape(name)+r'\}',
                lambda m:r'\fbox{\parbox[c][28mm][c]{.85\linewidth}{\centering Draft: figure pending}}',source)
    name='layout_draft' if args.draft_layout else 'controlled_ablation_report'
    if args.draft_layout:
        source=source.replace(r'\textbf{Scope.}',r'\textbf{DRAFT LAYOUT: incomplete experiment matrix.}\par\textbf{Scope.}')
        source=source.replace('Controlled ablations |','DRAFT layout |')
    target=out/(name+'.tex');target.write_text(source)
    receipt={'draft_layout':args.draft_layout,'summary':str(Path(args.summary)),
        'summary_sha256':hashlib.sha256(Path(args.summary).read_bytes()).hexdigest(),'missing_figures':missing,
        'source':str(target),'final_audit_complete':audit.get('complete',False)}
    (out/(name+'_provenance.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
