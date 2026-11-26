"""Prepare the first development-only validation of TwoRoom private paths."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--manifest',required=True)
    args=p.parse_args();root=Path('independent_composition_artifacts');out=Path(args.out)
    manifest=Path(args.manifest)
    if out.exists() or manifest.exists():raise FileExistsError('Fresh validation manifest required')
    ready=out.parent/(out.name+'_ready.json')
    jobs=[{'id':'wait_for_training','command':[sys.executable,'-u','-m','independent_composition.wait_jobs',
        '--queues',str(root/'tworoom_private_routes_v8'),str(root/'tworoom_private_controllers_v8'),
        '--out',str(ready)],'result':str(ready)}]
    for method in ['keepout','waypoint','fixed','compose']:
        for kind in ['supported_easy','supported_conflict','coupled_ood']:
            dest=out/'results'/f'{method}_{kind}'
            command=[sys.executable,'-u','-m','independent_composition.evaluate_private_paths',
                '--env','tworoom','--routes',
                *[str(root/'tworoom_private_routes_v8/models'/skill/'best.pt') for skill in ['keepout','waypoint']],
                '--controllers',*[str(root/'tworoom_private_controllers_v8/models'/skill/'best.pt') for skill in ['keepout','waypoint']],
                '--tasks',f'composition_artifacts/tasks/val_{kind}.npz','--out',str(dest),
                '--count','16','--method',method,'--iterations','120','--starts','8','--execute','1',
                '--executor','0','--seed','41','--guard-inference']
            jobs.append({'id':f'{method}_{kind}','command':command,'result':str(dest/'summary.json')})
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({'scope':'Initial development validation only. Own full-skill rates are distinct from strict joint success. No final test access. Composition uses only whole-route agreement plus Gaussian-code prior.','jobs':jobs},indent=2)+'\n')
    print(manifest)


if __name__=='__main__':main()
