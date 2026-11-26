"""Stdlib-only concise LaTeX builder, requiring complete actual evidence."""
import argparse,json,hashlib,shutil,re
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(head,rows):
    return '\n'.join([r'\begin{center}\begin{tabular}{l'+'r'*(len(head)-1)+r'}\toprule',' & '.join(head)+r'\\\midrule',*[' & '.join(map(str,r))+r'\\' for r in rows],r'\bottomrule\end{tabular}\end{center}'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--findings',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    e=Path(a.evidence);s=read(e/'aggregate/summary.json');audit=read(e/'protocol/final_audit.json');timing=read(e/'latency/summary.json');study=e/'protocol/study.json'
    assert s['complete'] and not s['partial'] and s['completed_cells']==90 and s['evaluations']==5760 and len(s['rows'])==30
    assert audit['complete'] and audit['cells']==90 and audit['episodes']==5760 and timing['complete'] and len(timing['rows'])==10
    assert s['study_sha256']==audit['study_sha256']==timing['study_sha256']==sha(study)
    findings=read(a.findings);assert set(findings)=={'MAIN_FINDING','LOSS_FINDING','RECOVERY_FINDING'} and all(len(x.split())<=110 for x in findings.values())
    kinds=['supported_easy','supported_conflict','coupled_ood'];labels=['Easy','Conflict','Coupled/OOD'];ns=[0,5,10,20,40]
    strata=[(env,kind,('TwoRoom' if env=='tworoom' else 'Cube')+' '+label) for env in ['tworoom','cube'] for kind,label in zip(kinds,labels)]
    def row(env,kind,n):return next(x for x in s['rows'] if (x['environment'],x['category'],x['steps'])==(env,kind,n))
    values=[]
    for env,kind,label in strata:
        old=next(x for x in s['historical_grid_reference'] if (x['environment'],x['category'],x['steps'])==(env,kind,20))
        values.append([label]+[f'{100*row(env,kind,n)["metrics"]["success"]["mean"]:.1f}' for n in ns]+[f'{100*old["success"]:.1f}'])
    replacement=dict(findings);replacement['SUCCESS_TABLE']=table(['Task/category']+[f'$N={n}$' for n in ns]+['Grid20'],values)
    values=[]
    for env,kind,label in strata:
        r20=row(env,kind,20)['paired_with_zero'];r40=row(env,kind,40)['paired_with_zero'];ci=r40['difference_context_seed_bootstrap95']
        values.append([label,r40['N0_fail_count'],f'{r20["N0_fail_refined_success_count"]}/{r20["N0_success_refined_fail_count"]}',f'{r40["N0_fail_refined_success_count"]}/{r40["N0_success_refined_fail_count"]}',f'{100*r40["success_difference"]:+.1f} [{100*ci[0]:+.1f},{100*ci[1]:+.1f}]'])
    replacement['RECOVERY_TABLE']=table(['Task/category','N0 fails','N20 R/G','N40 R/G',r'$\Delta$N40 [95\%]'],values)
    replacement['SEED_TABLE']=table(['Task/category',r'N0 seeds (\%)',r'N20 seeds (\%)',r'N40 seeds (\%)'],
        [[label]+[' / '.join(f'{v*100:.1f}' for v in row(env,kind,n)['metrics']['success']['per_seed_means']) for n in [0,20,40]] for env,kind,label in strata])
    replacement['TIME_TABLE']=table(['Median batch1 (ms)']+[f'$N={n}$' for n in ns],
        [[name]+[f'{1000*next(x for x in timing["rows"] if x["environment"]==env and x["steps"]==n)["median_wall_seconds"]:.1f}' for n in ns] for env,name in [('tworoom','TwoRoom'),('cube','Cube')]])
    replacement['AUDIT']=f'{audit["planning_calls"]:,} recorded planner calls'
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True);(out/'figures').mkdir();digests={}
    for folder,names in [('figures',['success_vs_steps','disagreement_vs_steps','success_vs_latency']),('representatives',['recovery','persistent_failure'])]:
        root=e/'aggregate'/folder;proof=read(root/'figure_provenance.json');assert proof['complete']
        if folder=='figures':assert proof['summary_sha256']==sha(e/'aggregate/summary.json') and proof['latency_sha256']==sha(e/'latency/summary.json')
        else:assert proof['study_sha256']==sha(study)
        for name in names:
            source=root/(name+'.pdf');assert sha(source)==proof['files'][source.name];shutil.copy2(source,out/'figures'/source.name);digests[source.name]=sha(source)
    text=Path(__file__).with_name('report_template.tex').read_text()
    for name,value in replacement.items():text=text.replace('@@'+name+'@@',value)
    assert not re.search(r'@@\w+@@',text)
    tex=out/'single_gaussian_composition_report.tex';tex.write_text(text)
    write={'complete_evidence':True,'study_sha256':sha(study),'summary_sha256':sha(e/'aggregate/summary.json'),
        'audit_sha256':sha(e/'protocol/final_audit.json'),'timing_sha256':sha(e/'latency/summary.json'),
        'findings_sha256':sha(a.findings),'latex_sha256':sha(tex),'figure_sha256':digests,
        'scope':'Final PDF still requires compilation and inspection of every rendered page.'}
    (out/'report_provenance.json').write_text(json.dumps(write,indent=2)+'\n');print({'latex':str(tex),'figure_families':5})

if __name__=='__main__':main()
