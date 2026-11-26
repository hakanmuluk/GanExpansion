"""Whitelist aggregate statistics/protocol/code; never native episode arrays."""
import argparse,shutil,tarfile,json
from pathlib import Path
from .freeze import read,write,sha,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);p.add_argument('--aggregate');p.add_argument('--partial',action='store_true');a=p.parse_args()
    s=read(a.study);verify(s);root=Path(a.study).parent;out=Path(a.out);assert not out.exists();out.mkdir(parents=True);files=[]
    jobs=[j for j in s['jobs'] if (Path(j['out'])/'DONE.json').exists()]
    aggregate=Path(a.aggregate) if a.aggregate else root/'aggregate'
    if not a.partial:assert len(jobs)==90 and read(root/'final_audit.json')['complete'] and read(root/'latency/summary.json')['complete'] and read(aggregate/'summary.json')['complete']
    def copy(p,name):
        p=Path(p);target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);files.append(target)
    for name in ['study.json','preflight.json','queue_process.json','final_audit.json','serial_parallel_native_parity.json','technical_correction.json']:
        if (root/name).exists():copy(root/name,'protocol/'+name)
    copy(s['reference_study'],'protocol/reference_study.json')
    if (root/'queue/status.json').exists():copy(root/'queue/status.json','protocol/queue_status.json')
    for p in (root/'configs').rglob('*.json'):copy(p,'protocol/configs/'+str(p.relative_to(root/'configs')))
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ['.py','.txt','.json','.tex']:copy(p,'source/single_gaussian_study/'+p.name)
    for j in jobs:
        for name in ['config.json','summary.json','DONE.json','native_replay_aggregate.json']:
            p=Path(j['out'])/name
            if p.exists():copy(p,'cells/'+j['id']+'/'+name)
    for folder,prefix in [(aggregate,'aggregate'),(root/'latency','latency')]:
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if not p.is_file():continue
            assert p.suffix in ['.json','.csv','.pdf','.png']
            assert p.name not in ['records.json','planning.json'] and not p.name.startswith('rollout_')
            copy(p,prefix+'/'+str(p.relative_to(folder)))
    write(out/'EXPORT_MANIFEST.json',{'partial':a.partial,'completed_cells':len(jobs),'expected_cells':90,'study_sha256':sha(a.study),
        'file_sha256':{str(p.relative_to(out)):sha(p) for p in files},'raw_episode_data_exported':False,
        'scope':'Code, frozen protocol, aggregate results/figures and cell-level checks only. No native arrays, Gaussian samples per episode, observations, actions, poses or planning traces.'})
    files.append(out/'EXPORT_MANIFEST.json');archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(files):t.add(p,arcname=str(p.relative_to(out)),recursive=False)
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'completed_cells':len(jobs),'partial':a.partial}))

if __name__=='__main__':main()
