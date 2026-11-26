"""Package a completed study using authoritative checkpoint hashes.

The final export supplies aggregate evidence. Earlier model exports are used
only when their ORIGINAL source hash matches the final checkpoint inventory.
This prevents superseded Cube attempts from being selected by pathname.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import zipfile


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def safe(name):
    path=PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts and '\\' not in name,name
    return str(path)


def zip_files(target,files,metadata=None):
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for source,name in sorted(files,key=lambda pair:pair[1]):archive.write(source,safe(name))
        if metadata:archive.writestr('PACKAGE_MANIFEST.json',json.dumps(metadata,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--summary',required=True);p.add_argument('--final-export',required=True)
    p.add_argument('--backups',default='ablation_results/backups');p.add_argument('--report-source',required=True)
    p.add_argument('--report-pdf',required=True);p.add_argument('--smoke-proof',required=True)
    p.add_argument('--out',required=True);p.add_argument('--verify-tensors',action='store_true')
    args=p.parse_args();inventory=read(args.inventory);audit=read(args.audit);summary=read(args.summary)
    assert audit.get('complete') and not audit['failures'] and not audit['pending'],'Complete control audit required'
    assert not summary['partial'] and len(summary['base'])==8 and len(summary['adapters'])==48
    assert len(summary['fixed_code_diagnostics'])==8 and inventory['selected_count']==56
    smoke=read(args.smoke_proof)
    assert smoke['complete'] and smoke['selected_count']==56 and len(smoke['cases'])==56
    assert smoke['inventory_sha256']==sha(args.inventory)
    checked={r['case']:r for r in smoke['cases']}
    assert len(checked)==56 and set(checked)=={r['case'] for r in inventory['selected_models']}
    assert all(r['checks'] and all(r['checks'].values()) for r in smoke['cases'])
    source=Path(args.report_source);pdf=Path(args.report_pdf);assert source.is_file() and pdf.is_file()
    provenance=read(source.with_name(source.stem+'_provenance.json'))
    assert not provenance['draft_layout'] and provenance['final_audit_complete']
    assert provenance['summary_sha256']==sha(args.summary)
    assert '@@' not in source.read_text() and 'DRAFT LAYOUT' not in source.read_text()
    final=Path(args.final_export);receipt=read(final/'EXPORT_MANIFEST.json')
    assert not receipt['raw_episode_data_exported']
    assert sha(final/'aggregate/base_evaluation_checkpoint_binding.json')==inventory['base_evaluation_binding_sha256']
    for name,expected in receipt['file_sha256'].items():
        safe(name);assert sha(final/name)==expected,('Final-export member mismatch',name)
    # Search verified exports by original source hash, never by case name alone.
    available={}
    for manifest in sorted(Path(args.backups).glob('completed_*/EXPORT_MANIFEST.json')):
        exported=read(manifest)
        for model in exported['models']:
            available[(model['original'],model['original_sha256'])]=(manifest.parent,model)
    selected=[]
    for model in inventory['selected_models']:
        key=(model['source'],model['original_source_sha256']);assert key in available,('Missing selected checkpoint export',model['case'],key)
        folder,row=available[key];path=folder/row['path']
        assert sha(path)==row['sha256'] and row['tensor_sha256']==model['tensor_sha256'],model['case']
        assert checked[model['case']]['original_source_sha256']==model['original_source_sha256']
        assert checked[model['case']]['compact_sha256']==row['sha256'],('Inference proof differs from selected compact file',model['case'])
        if args.verify_tensors:
            import torch
            from .export_completed import state_hash
            ck=torch.load(path,map_location='cpu',weights_only=False)
            assert ck['export_provenance']['source_sha256']==model['original_source_sha256']
            assert all(state_hash(ck[k])==h for k,h in model['tensor_sha256'].items()),model['case']
        selected.append({'case':model['case'],'path':path,'archive_path':safe(model['source']),
            'original_source_sha256':model['original_source_sha256'],'compact_sha256':row['sha256'],
            'tensor_sha256':model['tensor_sha256'],'kind':model['kind']})
    frozen=[]
    for model in inventory['unchanged_original_bases']:
        path=Path(model['source'])
        if not path.exists() and model['case']=='cube_latest_existing':
            path=Path('independent_composition_artifacts/z_regularized_composition_release/weights/cube/base_policy.pt')
        assert path.is_file() and sha(path)==model['sha256'],model['case']
        frozen.append({'case':model['case'],'path':path,'archive_path':safe(model['source']),'sha256':model['sha256']})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    weights_meta={'selected_models':[{k:v for k,v in row.items() if k!='path'} for row in selected],
        'unchanged_original_bases':[{k:v for k,v in row.items() if k!='path'} for row in frozen],
        'scope':'Exact selected inference tensors. Compact new checkpoints omit optimizer/RNG state. Original frozen base files retain exact bytes for adapter-parent SHA validation.',
        'raw_episode_data_exported':False}
    zip_files(out/'selected_weights.zip',[(r['path'],r['archive_path']) for r in selected+frozen],weights_meta)
    # Every scientific result/log here is from the final, completed export.
    evidence=[]
    for folder in ['results','training_evidence','controls']:
        for path in (final/folder).rglob('*'):
            if not path.is_file():continue
            assert path.suffix in ['.json','.jsonl','.py','.diff','.pdf','.png','.csv']
            assert not any(s in path.name for s in ['private_episode','private_rollouts','records.json'])
            evidence.append((path,str(path.relative_to(final))))
    for row in receipt.get('reference_rng',[]):
        path=final/row['path'];assert sha(path)==row['sha256']
        evidence.append((path,'reproduction/'+safe(row['path'])))
    presentation=Path(args.summary).parent
    for path in presentation.iterdir():
        if path.is_file() and path.suffix in ['.csv','.json','.pdf','.png']:evidence.append((path,'tables_and_figures/'+path.name))
    proof_names=['environment.json','cube_loss_correction.json','base_comparison_input_audit.json',
                 'export_retry_rng_temp_receipt.json','first_trained_pure_cube_actor_functional_audit.json',
                 'supplementary_speed_axis_queue_receipt.json','adapter_figure_layout_v2.json',
                 'base_evaluation_checkpoint_binding.json','code_bundle_import_check.json']
    for name in proof_names:
        path=final/'aggregate'/name
        if path.exists():evidence.append((path,'audit/'+name))
    evidence.extend([(Path(args.audit),'audit/final_control_audit.json'),
        (Path(args.inventory),'audit/final_checkpoint_inventory.json'),
        (Path(args.smoke_proof),'audit/final_compact_inference_checked.json'),
        (Path('ablation_study/recorded/base_architecture_normalization_audit.json'),'audit/base_architecture_normalization_audit.json')])
    for name in ['protocol.json','queue_manifest.json','cube_known_rec_correction.json','bootstrap_manifest.json','adapter_input_manifest.json','REPRODUCE.txt']:
        evidence.append((Path('ablation_study')/name,'reproduction/'+name))
    zip_files(out/'results_logs_and_protocol.zip',evidence,{'final_control_audit_sha256':sha(args.audit),
        'selected_inventory_sha256':sha(args.inventory),'summary_sha256':sha(args.summary),'raw_episode_data_exported':False,
        'scope':'Final aggregate metrics, vector/PNG figures, training logs and protocol. Detailed native trajectories remain on execution host.'})
    tex_files=[(source,source.name)]
    for path in (source.parent/'figures').glob('*'):
        if path.is_file():tex_files.append((path,'figures/'+path.name))
    tex_files.append((source.with_name(source.stem+'_provenance.json'),'report_provenance.json'))
    zip_files(out/'editable_latex_and_figures.zip',tex_files)
    shutil.copy2(pdf,out/pdf.name)
    subprocess.run([sys.executable,'-m','ablation_study.package_code','--out',str(out/'code_and_configs.zip')],check=True)
    (out/'REPRODUCE.txt').write_bytes(Path('ablation_study/REPRODUCE.txt').read_bytes())
    with (out/'selected_checkpoint_locations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','kind','archive_path','original_source_sha256','compact_sha256']);writer.writeheader()
        writer.writerows({k:row[k] for k in writer.fieldnames} for row in selected)
    artifact_files=sorted(path for path in out.iterdir() if path.is_file())
    manifest={'complete':True,'selected_model_count':len(selected),'original_base_count':len(frozen),
        'files':{path.name:{'bytes':path.stat().st_size,'sha256':sha(path)} for path in artifact_files},
        'tensor_verification_performed':args.verify_tensors,'functional_inference_checked_models':56,
        'functional_inference_receipt_sha256':sha(args.smoke_proof),'raw_episode_data_exported':False}
    (out/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out/'complete_study_bundle.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for path in artifact_files+[out/'DELIVERY_MANIFEST.json']:archive.write(path,path.name)
    print(json.dumps({'out':str(out),'selected_models':len(selected),'weights_bytes':(out/'selected_weights.zip').stat().st_size,
        'bundle_bytes':(out/'complete_study_bundle.zip').stat().st_size,'bundle_sha256':sha(out/'complete_study_bundle.zip')}),flush=True)


if __name__=='__main__':main()
