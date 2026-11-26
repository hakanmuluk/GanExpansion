"""Build concise editable LaTeX only from the complete, audited study."""
import argparse,hashlib,json,re,shutil
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def table(headers,rows):
    return '\n'.join([r'\begin{center}\begin{tabular}{l'+'r'*(len(headers)-1)+r'}\toprule',
        ' & '.join(headers)+r'\\\midrule',*[' & '.join(map(str,r))+r'\\' for r in rows],
        r'\bottomrule\end{tabular}\end{center}'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--aggregate',required=True);p.add_argument('--latency',required=True)
    p.add_argument('--audit',required=True);p.add_argument('--study',required=True);p.add_argument('--representatives',required=True)
    p.add_argument('--findings',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.aggregate);s=read(root/'summary.json');audit=read(a.audit);timing=read(a.latency);study=read(a.study)
    assert not s['partial'] and s['completed_cells']==84 and len(s['rows'])==84
    assert audit['complete'] and audit['cells']==84 and audit['episodes']==5376
    assert timing['complete'] and len(timing['rows'])==20
    assert s['study_sha256']==audit['study_sha256']==timing['study_sha256']==sha(a.study)
    assert all(r['all_within_one_percent'] for r in timing['matched_work_validation'])
    provenance=read(root/'figures/figure_provenance.json');rep=read(Path(a.representatives)/'figure_provenance.json')
    assert provenance['complete'] and provenance['source_summary_sha256']==sha(root/'summary.json')
    assert provenance['latency_summary_sha256']==sha(a.latency) and rep['complete'] and rep['study_sha256']==sha(a.study)
    findings=read(a.findings)
    assert set(findings)=={'RQ1_FINDING','RQ2_FINDING','RQ3_FINDING'}
    assert all(len(v.split())<=110 for v in findings.values()),'Keep each interpretation concise'
    kinds=['supported_easy','supported_conflict','coupled_ood'];labels=['Easy','Conflict','Coupled/OOD']
    tasks=[(e,k,('TwoRoom' if e=='tworoom' else 'Cube')+' '+label) for e in ['tworoom','cube'] for k,label in zip(kinds,labels)]
    def row(e,key,k):
        selected=[r for r in s['rows'] if r['environment']==e and r['setting']==key and r['category']==k]
        assert len(selected)==1;return selected[0]
    replacements=dict(findings)
    replacements['HORIZON_TABLE']=table(['Task/category']+[f'$H={h}$' for h in [4,8,12,16]],
        [[label]+[f'{100*row(e,f"h{h}_n20",k)["success"]:.1f}' for h in [4,8,12,16]] for e,k,label in tasks])
    replacements['COMMON8_TABLE']=table(['Task/category']+[f'$H={h}$' for h in [4,8,12,16]],
        [[label]+[f'{1000*row(e,f"h{h}_n20",k)["common8_first"]:.1f}' for h in [4,8,12,16]] for e,k,label in tasks])
    b=read(root/'agreement_predictiveness.json');values=[]
    for e in ['tworoom','cube']:
        for k,label in list(zip(kinds,labels))+[('pooled','Pooled')]:
            r=b[e+'/'+k];auc=r['auroc_negative_raw_disagreement'];ci=r['bootstrap95']
            interval=r'\textemdash' if auc is None else f'{auc:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]'+(r'$^*$' if r['sparse_outcome_class'] else '')
            values.append([('TwoRoom' if e=='tworoom' else 'Cube')+' '+label,r['contexts'],f'{r["successes"]}/{r["contexts"]}',interval])
    replacements['AUROC_TABLE']=table(['Task/category','Contexts','Joint successes','AUROC [bootstrap95]'],values)
    replacements['AUROC_TABLE']+=r'{\small $^*$Fewer than five outcomes in one class. Resampling cannot recover unseen class variation; these intervals remain descriptive.\par}'
    values=[]
    for e,k,label in tasks:
        random=next(r for r in s['random_search_aggregates'] if r['environment']==e and r['category']==k)
        values.append([label]+[f'{row(e,f"h8_n{n}",k)["success"]*100:.1f}' for n in [0,5,10,20,40]]+[f'{100*random["success"]:.1f}'])
    replacements['STEPS_TABLE']=table(['Task/category','Grid/0','5','10','20','40','Random'],values)+r'\textit{Joint success (\%); random is the three-seed mean on the same 64 contexts.}'
    work=[]
    for e,name,g in [('tworoom','TwoRoom',90),('cube','Cube',360)]:
        m=study['environments'][e]['random_compute_budget']['random_candidates_after_unchanged_grid']
        work.append([name,f'{g}/{g+1}',2*(g+4*21+1),160,m,2*(g+m+1)])
    replacements['WORK_TABLE']=table(['Environment','Grid cold/warm','Adam20 F','Adam20 B','Random extra','Random F'],work)+r'{\small F/B count actor forward/input-backward sample equivalents at batch1, cold start. Both add one original-base forward; warm grids add two actor samples. Matrix profiling accounts for the differing forward/backward operations.\par}'
    timing_map={(r['environment'],r['setting']):r for r in timing['rows']}
    replacements['TIME_TABLE']=table(['Median batch1 (ms)','Grid/0','Adam20','Adam40','Random'],
        [[name]+[f'{timing_map[e,key]["median_wall_seconds"]*1000:.1f}' for key in ['h8_n0','h8_n20','h8_n40','random_budget20']] for e,name in [('tworoom','TwoRoom'),('cube','Cube')]])
    replacements['REPRESENTATIVE_NOTE']=r'The preview uses the first reference failure in the conflict category in each environment, with the same case across methods. Full success/failure examples for every category accompany the report. Cube shows a projection; its labels use the original 3D physical monitor.'
    native=sum(r['episodes'] for r in audit['checks'] if r['job'].startswith('tworoom_'))
    replacements['AUDIT_PARAGRAPH']=(f'All {audit["cells"]} cells and {audit["episodes"]:,} evaluated episodes completed on 384 reused contexts. '
        f'The audit checks {audit["planning_calls_checked"]:,} planning calls, all model/task/configuration hashes, matched initial states, retained regularizers and actual forward/backward counts. '
        f'All {native:,} TwoRoom paths matched original-native replay; all six reference cells matched their original complete trajectories and outcomes.')
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True);figs=out/'figures';figs.mkdir()
    figure_names=['horizon_success','horizon_action_disagreement','agreement_predictiveness','agreement_landmarks',
        'steps_success','success_vs_latency','agreement_by_steps','optimization_convergence']
    for name in figure_names:
        source=root/'figures'/(name+'.pdf');assert sha(source)==provenance['files'][source.name]
        shutil.copy2(source,figs/source.name)
    for name in ['representative_tworoom','representative_cube','representative_overview']:
        source=Path(a.representatives)/(name+'.pdf');assert sha(source)==rep['files'][source.name]
        shutil.copy2(source,figs/source.name)
    source=Path(__file__).with_name('report_template.tex').read_text()
    for name,value in replacements.items():source=source.replace('@@'+name+'@@',value)
    assert not re.search(r'@@\w+@@',source)
    target=out/'latent_composition_ablation_report.tex';target.write_text(source)
    proof={'complete_evidence':True,'source_summary_sha256':sha(root/'summary.json'),'study_sha256':sha(a.study),
        'audit_sha256':sha(a.audit),'timing_sha256':sha(a.latency),'findings_sha256':sha(a.findings),
        'latex_sha256':sha(target),'figure_sha256':{p.name:sha(p) for p in sorted(figs.iterdir())},
        'scope':'Five-page target layout; final PDF must still be rendered and visually verified.'}
    (out/'report_provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'source':str(target),'figures':len(proof['figure_sha256']),'complete_evidence':True}))

if __name__=='__main__':main()
