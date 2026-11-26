"""Package only fully verified composition-study artifacts; no native arrays."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path,PurePosixPath

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--export',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--source-proof',required=True);p.add_argument('--report-source',required=True)
    p.add_argument('--report-pdf',required=True);p.add_argument('--findings',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();e=Path(a.export);reference=Path(a.reference);receipt=read(e/'EXPORT_MANIFEST.json')
    assert not receipt['partial'] and receipt['completed_cells']==84 and not receipt['raw_episode_data_exported']
    for name,digest in receipt['file_sha256'].items():
        path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts
        assert sha(e/name)==digest,name
    summary=read(e/'aggregate/summary.json');audit=read(e/'protocol/final_audit.json');study=read(e/'protocol/study.json')
    assert not summary['partial'] and len(summary['rows'])==84 and audit['complete'] and audit['episodes']==5376
    assert summary['study_sha256']==audit['study_sha256']==sha(e/'protocol/study.json')
    ref=read(reference/'REFERENCE_INPUT_MANIFEST.json');proof=read(a.source_proof)
    assert ref['study_sha256']==sha(e/'protocol/study.json') and not ref['raw_episode_data_exported']
    assert len(ref['omitted_protected_task_banks'])==6
    assert proof['passed'] and not proof['task_observations_or_rollouts_used']
    assert proof['source_zip_sha256']==ref['source']['sha256']
    for key,name in [('source','reference_code_and_protocol.zip'),('weights','unchanged_reference_weights.zip')]:
        assert sha(reference/name)==ref[key]['sha256']
        with zipfile.ZipFile(reference/name) as z:
            assert z.testzip() is None
            m=json.loads(z.read('SOURCE_MANIFEST.json'));assert len(m)==ref[key]['members']
            assert set(z.namelist())==set(m)|{'SOURCE_MANIFEST.json'}
            for member,r in m.items():
                assert not PurePosixPath(member).is_absolute() and '..' not in PurePosixPath(member).parts
                assert not member.endswith(('.npz','.npy'))
                assert hashlib.sha256(z.read(member)).hexdigest()==r['sha256']
            if key=='weights':
                expected={p.removeprefix('/content/'):h for p,h in study['original_hashes'].items() if p.endswith('.pt')}
                assert len(expected)==len(m)==10 and {name:r['sha256'] for name,r in m.items()}==expected
    source=Path(a.report_source);pdf=Path(a.report_pdf);r=read(source.parent/'report_provenance.json')
    assert r['complete_evidence'] and r['study_sha256']==sha(e/'protocol/study.json')
    assert r['source_summary_sha256']==sha(e/'aggregate/summary.json') and r['audit_sha256']==sha(e/'protocol/final_audit.json')
    assert r['findings_sha256']==sha(a.findings) and r['latex_sha256']==sha(source)
    assert r['pdf_sha256']==sha(pdf) and r['visual_quality_check']['passed'] and r['visual_quality_check']['all_pages_inspected']
    assert '@@' not in source.read_text()
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    def zip_files(target,files):
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path,name in sorted(files,key=lambda x:x[1]):z.write(path,name)
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
    evidence=[(p,str(p.relative_to(e))) for p in e.rglob('*') if p.is_file()]
    zip_files(out/'aggregate_results_and_figures.zip',evidence)
    editable=[(source,source.name),(source.parent/'report_provenance.json','report_provenance.json'),(Path(a.findings),'findings.json')]
    for path in (source.parent/'figures').iterdir():
        assert sha(path)==r['figure_sha256'][path.name]
        editable.append((path,'figures/'+path.name))
    zip_files(out/'editable_latex_and_figures.zip',editable)
    for original,new in [('reference_code_and_protocol.zip','original_code_and_configs.zip'),('unchanged_reference_weights.zip','frozen_reference_weights.zip'),('REFERENCE_INPUT_MANIFEST.json','REFERENCE_INPUT_MANIFEST.json')]:
        shutil.copy2(reference/original,out/new)
    shutil.copy2(a.source_proof,out/'SOURCE_IMPORT_CHECK.json');shutil.copy2(pdf,out/pdf.name)
    shutil.copy2(Path(__file__).with_name('REPRODUCE.txt'),out/'REPRODUCE.txt')
    artifacts=sorted(p for p in out.iterdir() if p.is_file())
    manifest={'complete':True,'evaluated_cells':84,'evaluated_episodes':5376,'independent_contexts':384,
        'unchanged_reference_checkpoints':10,'all_final_pdf_pages_visually_checked':True,
        'source_archive_imports_and_actor_loading_checked':True,'raw_episode_data_exported':False,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in artifacts},
        'protected_case_banks_not_included':True,'study_sha256':sha(e/'protocol/study.json')}
    (out/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out/'complete_composition_study_bundle.zip','w',compression=zipfile.ZIP_STORED) as z:
        for path in artifacts+[out/'DELIVERY_MANIFEST.json']:z.write(path,path.name)
    with zipfile.ZipFile(out/'complete_composition_study_bundle.zip') as z:assert z.testzip() is None
    print(json.dumps({'complete':True,'out':str(out),'bundle_bytes':(out/'complete_composition_study_bundle.zip').stat().st_size,
        'bundle_sha256':sha(out/'complete_composition_study_bundle.zip')}))

if __name__=='__main__':main()
