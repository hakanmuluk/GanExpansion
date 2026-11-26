"""Stdlib-only delivery gates for complete actual single-start results."""
import argparse,hashlib,json,zipfile,shutil
from pathlib import Path,PurePosixPath
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--report-source',required=True);p.add_argument('--report-pdf',required=True);p.add_argument('--findings',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    e=Path(a.evidence);snap=Path(a.snapshot);export=read(e/'EXPORT_MANIFEST.json');s=read(e/'aggregate/summary.json');audit=read(e/'protocol/final_audit.json')
    assert not export['partial'] and export['completed_cells']==90 and not export['raw_episode_data_exported']
    for n,h in export['file_sha256'].items():
        assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
        assert sha(e/n)==h,n
    assert s['complete'] and len(s['rows'])==30 and s['evaluations']==5760 and audit['complete'] and audit['cells']==90 and audit['episodes']==5760
    assert s['study_sha256']==audit['study_sha256']==sha(e/'protocol/study.json')
    manifest=read(snap/'INPUT_MANIFEST.json');proof=read(snap/'proof.json')
    assert manifest['study_sha256']==sha(e/'protocol/study.json') and not manifest['raw_episode_data_exported']
    assert proof['passed'] and not proof['native_task_data_used'] and proof['single_start_CPU_N0_and_N20_checked']
    assert proof['source_zip_sha256']==manifest['files']['single_gaussian_code_and_protocol.zip']['sha256']
    original=read(e/'protocol/reference_study.json');expected={p.removeprefix('/content/'):h for p,h in original['original_hashes'].items() if p.endswith('.pt')}
    for name,meta in manifest['files'].items():
        assert sha(snap/name)==meta['sha256']
        with zipfile.ZipFile(snap/name) as z:
            assert z.testzip() is None;m=json.loads(z.read('SOURCE_MANIFEST.json'));assert set(z.namelist())==set(m)|{'SOURCE_MANIFEST.json'}
            for n,v in m.items():
                assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and not n.endswith(('.npz','.npy'))
                assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
            if name=='frozen_reference_weights.zip':assert len(m)==len(expected)==10 and {n:v['sha256'] for n,v in m.items()}==expected
    source=Path(a.report_source);pdf=Path(a.report_pdf);r=read(source.parent/'report_provenance.json')
    assert r['complete_evidence'] and r['study_sha256']==sha(e/'protocol/study.json') and r['summary_sha256']==sha(e/'aggregate/summary.json')
    assert r['findings_sha256']==sha(a.findings) and r['latex_sha256']==sha(source) and r['pdf_sha256']==sha(pdf)
    assert r['visual_quality_check']['passed'] and r['visual_quality_check']['all_pages_inspected']
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    def archive(target,files):
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path,name in sorted(files,key=lambda x:x[1]):z.write(path,name)
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
    archive(out/'aggregate_results_and_figures.zip',[(p,str(p.relative_to(e))) for p in e.rglob('*') if p.is_file()])
    editable=[(source,source.name),(source.parent/'report_provenance.json','report_provenance.json'),(Path(a.findings),'findings.json')]
    for p in (source.parent/'figures').iterdir():
        assert sha(p)==r['figure_sha256'][p.name];editable.append((p,'figures/'+p.name))
    archive(out/'editable_latex_and_figures.zip',editable)
    for name in manifest['files']:shutil.copy2(snap/name,out/name)
    shutil.copy2(snap/'INPUT_MANIFEST.json',out/'INPUT_MANIFEST.json');shutil.copy2(snap/'proof.json',out/'SOURCE_IMPORT_CHECK.json')
    shutil.copy2(pdf,out/pdf.name);shutil.copy2(Path(__file__).with_name('REPRODUCE.txt'),out/'REPRODUCE.txt')
    files=[p for p in sorted(out.iterdir()) if p.is_file()]
    result={'complete':True,'cells':90,'evaluations':5760,'independent_contexts':384,'initialization_seeds':3,'unchanged_checkpoints':10,
        'raw_episode_data_exported':False,'all_report_pages_visually_checked':True,'source_imports_and_CPU_actors_checked':True,
        'study_sha256':sha(e/'protocol/study.json'),'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}}
    (out/'DELIVERY_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(out/'complete_single_gaussian_study_bundle.zip','w',compression=zipfile.ZIP_STORED) as z:
        for path in files+[out/'DELIVERY_MANIFEST.json']:z.write(path,path.name)
    with zipfile.ZipFile(out/'complete_single_gaussian_study_bundle.zip') as z:assert z.testzip() is None
    print(json.dumps({'complete':True,'bundle':str(out/'complete_single_gaussian_study_bundle.zip'),'sha256':sha(out/'complete_single_gaussian_study_bundle.zip')}))

if __name__=='__main__':main()
