"""Generate fresh witness-certified scenes only after a frozen protocol exists.

Native teachers certify feasibility during benchmark construction. Their
trajectories stay in a separate directory and never reach any learned policy.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=64);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();protocol=json.loads(Path(args.protocol).read_text())
    assert protocol['environment']=='cube' and protocol['created']<time.time()
    assert args.count>=64
    root=Path(args.out)
    if root.exists():raise FileExistsError('Fresh final scenes required')
    root.mkdir(parents=True)
    from cube_composition.data import candidate,init_worker
    manifest={'protocol_sha256':digest_file(args.protocol),'created':time.time(),
              'selection':'Geometry and independent native witness only; no learned-policy outcome filtering',
              'generator_sha256':digest_file(__file__),'strata':{}}
    with mp.get_context('spawn').Pool(args.workers,initializer=init_worker) as pool:
        for k,kind in enumerate(['supported_easy','supported_conflict','coupled_ood']):
            offset=protocol['fresh_scene_seed']+k*100000;next_seed=offset;accepted=[];attempts=[]
            while len(accepted)<args.count:
                jobs=[(s,kind) for s in range(next_seed,next_seed+max(32,args.workers*8))];next_seed+=len(jobs)
                for result in pool.imap(candidate,jobs,chunksize=1):
                    attempts.append({key:value for key,value in result.items() if key not in ['case','witness']})
                    if result.get('accepted') and len(accepted)<args.count:
                        index=len(accepted);accepted.append(result['case'])
                        assert result['witness'] is not None
                        witness=root/'witnesses'/kind;witness.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(witness/f'{index:04d}.npz',**result['witness'])
                print(json.dumps({'kind':kind,'accepted':len(accepted),'candidates':len(attempts)}),flush=True)
                if next_seed-offset>args.count*400:raise RuntimeError('Insufficient independently certified scenes')
            arrays={key:np.stack([c[key] for c in accepted]) for key in accepted[0]}
            target=root/f'final_{kind}.npz';np.savez_compressed(target,**arrays)
            manifest['strata'][kind]={'path':str(target),'count':len(accepted),'sha256':digest_file(target),
                'generation_seed':offset,'candidate_results':attempts}
            (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'DONE.json').write_text(json.dumps({'complete':True,'strata':3})+'\n')


if __name__=='__main__':main()
