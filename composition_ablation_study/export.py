"""Whitelist aggregate evidence/source only; raw native records remain remote."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
from .analyze import read,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True)
    p.add_argument('--aggregate');p.add_argument('--partial',action='store_true');a=p.parse_args()
    study=read(a.study);root=Path(a.study).parent;out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    done=[j for j in study['jobs'] if (Path(j['out'])/'DONE.json').exists()]
    if not a.partial:
        assert len(done)==84 and read(root/'final_audit.json')['complete'] and read(root/'latency/summary.json')['complete']
    files=[]
    def copy(source,name):
        source=Path(source);target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);files.append(target)
    for name in ['study.json','preflight.json','queue_process.json','final_audit.json']:
        if (root/name).exists():copy(root/name,'protocol/'+name)
    for name in ['restore_proof.json','case_restoration_proof.json','environment_proof.json',
                 'native_check.json','recovery_preflight.json','reset_recovery.json',
                 'reference_repeatability.json','six_reference_replay_proof.json',
                 'runtime_wrapper_precheck.json','horizon_repeatability.json',
                 'steps42_repeatability.json']:
        path=root/'recovery_metadata'/name
        if path.exists():copy(path,'protocol/recovery/'+name)
    if (root/'queue/status.json').exists():copy(root/'queue/status.json','protocol/queue_status.json')
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ['.py','.json','.txt','.tex']:copy(path,'source/composition_ablation_study/'+path.name)
    for job in done:
        for name in ['config.json','summary.json','DONE.json','native_replay_aggregate.json','original_reference_replay.json']:
            path=Path(job['out'])/name
            if path.exists():copy(path,'cells/'+job['id']+'/'+name)
    aggregate=Path(a.aggregate) if a.aggregate else root/'aggregate'
    if not a.partial:assert not read(aggregate/'summary.json')['partial'] and read(aggregate/'summary.json')['completed_cells']==84
    for folder,prefix in [(aggregate,'aggregate'),(root/'latency','latency')]:
        if folder.exists():
            for path in folder.rglob('*'):
                if not path.is_file():continue
                assert path.suffix in ['.csv','.json','.pdf','.png'],path
                assert path.name not in ['records.json','planning.json'] and not path.name.startswith('rollout_')
                copy(path,prefix+'/'+str(path.relative_to(folder)))
    record={'partial':a.partial,'completed_cells':len(done),'expected_cells':84,'study_sha256':sha(a.study),
        'raw_episode_data_exported':False,'file_sha256':{str(p.relative_to(out)):sha(p) for p in sorted(files)},
        'scope':'Aggregate outcomes/statistics/figures, protocol, source and cell-level checks only. No native arrays, poses, controls, case banks, planning traces or individual episode records.'}
    manifest=out/'EXPORT_MANIFEST.json';manifest.write_text(json.dumps(record,indent=2)+'\n');files.append(manifest)
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as t:
        for path in sorted(files):t.add(path,arcname=str(path.relative_to(out)),recursive=False)
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(files),'completed_cells':len(done)}))

if __name__=='__main__':main()
