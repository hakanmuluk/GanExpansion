"""Archive invalid Cube loss-ablation arms and activate corrected jobs safely.

Only the owned queue supervisor is briefly stopped. Its current child is never
killed or paused. Original scientific inputs/models are not touched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tarfile
import time


def alive(pid):
    try:return bool((Path('/proc')/str(pid)/'cmdline').read_bytes()) and (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--tools',required=True);a=p.parse_args()
    with tarfile.open(a.tools) as bundle:
        correction=json.load(bundle.extractfile('ablation_study/cube_known_rec_correction.json'))
        contents={m.name:bundle.extractfile(m).read() for m in bundle.getmembers() if m.isfile()}
    assert all(Path(n).parts[0]=='ablation_study' and '..' not in Path(n).parts for n in contents)
    status_path=Path('ablation_results/training_queue_v2/status.json');s=json.loads(status_path.read_text());pid=s['pid']
    cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
    assert b'ablation_study.run_queue' in cmd and alive(pid),'Refusing to signal a different process'
    invalid=set(correction['invalidated_jobs']);out=Path('ablation_results/invalid_cube_known_rec_20261126')
    assert not out.exists(),'Correction already applied or partially applied; inspect its receipt'
    os.kill(pid,signal.SIGSTOP)
    try:
        s=json.loads(status_path.read_text())
        running=[n for n,r in s['jobs'].items() if r['state']=='running' and n in invalid and alive(r['pid'])]
        assert not running,('An affected evaluation is still writing; retry after it finishes',running)
        paths=[]
        for seed in [42,43]:
            paths.extend([Path(f'ablation_results/training/cube_original_base_adversarial_s{seed}'),
                Path(f'ablation_results/training/cube_latest_base_adversarial_s{seed}'),
                Path(f'ablation_results/initialization/cube_latest_initialize_adversarial_s{seed}.pt'),
                Path(f'ablation_results/initialization/cube_latest_initialize_adversarial_s{seed}.json'),
                Path(f'ablation_results/evaluation/rq1_cube_latest_adversarial_s{seed}'),
                Path(f'ablation_results/evaluation/original_gaussian_cube_adversarial_s{seed}')])
        paths.extend([Path('ablation_results/figures/base_side_by_side_figures_s42'),
            Path('ablation_results/aggregate/latent_consistency_s42.json'),
            Path('ablation_results/aggregate/latent_consistency_second_seeds.json')])
        out.mkdir(parents=True);archived=[]
        shutil.copy2('ablation_study/base_ablation.py',out/'invalid_wrapper.py')
        for source in paths:
            if not source.exists():continue
            destination=out/source.relative_to('ablation_results');destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(destination));archived.append({'original':str(source),'archive':str(destination)})
        for name,data in contents.items():
            if name.endswith('queue_manifest.json'):continue
            path=Path(name);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.new');temporary.write_bytes(data);temporary.replace(path)
        path=Path('ablation_study/queue_manifest.json');temporary=path.with_suffix('.next.json')
        temporary.write_bytes(contents[str(path)]);temporary.replace(path)
        receipt={'reason':correction['reason'],'archived':archived,'queue_pid':pid,'time':time.time(),
            'current_unaffected_child':[n for n,r in s['jobs'].items() if r['state']=='running' and n not in invalid],
            'original_inputs_changed':False,'new_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (out/'INVALIDATED.json').write_text(json.dumps(receipt,indent=2)+'\n')
        Path('ablation_results/aggregate/cube_loss_correction.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'archived_paths':len(archived),'queue_resumes_with_corrected_jobs':True}),flush=True)
    finally:os.kill(pid,signal.SIGCONT)


if __name__=='__main__':main()
