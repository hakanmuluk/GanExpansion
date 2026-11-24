"""Fresh joint banks, independent native certification and sealed test inputs."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time,multiprocessing as mp,hashlib
from pathlib import Path
import numpy as np
from cube_composition.data import candidate,init_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=64);p.add_argument('--workers',type=int,default=4);p.add_argument('--splits',nargs='+',default=['val','test']);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);manifest={};started=time.perf_counter()
    with mp.get_context('spawn').Pool(args.workers,initializer=init_worker) as pool:
        for split,first in [('val',20000001),('test',30000001)]:
            if split not in args.splits:continue
            for j,kind in enumerate(['supported_easy','supported_conflict','coupled_ood']):
                name=split+'_'+kind;next_seed=first+j*1000000;accepted=[];attempts=[]
                while len(accepted)<args.count:
                    seeds=list(range(next_seed,next_seed+max(32,args.workers*8)));next_seed+=len(seeds)
                    for result in pool.imap(candidate,[(seed,kind) for seed in seeds],chunksize=1):
                        attempts.append({k:v for k,v in result.items() if k not in ['case','witness']})
                        if result.get('accepted') and len(accepted)<args.count:
                            index=len(accepted);accepted.append(result['case']);folder=out/'witnesses'/name;folder.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(folder/f'{index:04d}.npz',**result['witness'])
                arrays={k:np.stack([v[k] for v in accepted]) for k in accepted[0]};path=out/(name+'.npz');np.savez_compressed(path,**arrays)
                manifest[name]={'count':len(accepted),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'candidate_results':attempts,'selection':'First native independently certified feasible scenes, no learned-policy filter. Witnesses never train or initialize skill models/composition.'}
                (out/'manifest.json').write_text(json.dumps({'args':vars(args),'banks':manifest,'scope':'Fresh validation/test scenes for the retraining study. Neither bank supplies training demonstrations. Final test outcomes remain sealed until method selection is fixed.'},indent=2)+'\n');print('SAVED',name,len(accepted),'seconds',time.perf_counter()-started,flush=True)
    (out/'DONE.json').write_text(json.dumps({'complete':True,'banks':len(manifest),'seconds':time.perf_counter()-started}))


if __name__=='__main__':main()
