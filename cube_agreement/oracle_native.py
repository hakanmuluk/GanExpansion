"""Actual native execution of geometric oracle paths; no neural-success claim."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,multiprocessing as mp,time,hashlib
from pathlib import Path
import numpy as np
from cube_addons.env import make_env
from cube_composition.rollout import rollout
from .physical import PathTeacher

_ENV=None


def init():
    global _ENV
    _ENV=make_env()


def run(job):
    source,family,out=job
    with np.load(source) as f:case={k:f[k] for k in ['seed','kind','initial_state','initial_cube','goal','base_z','keepout','waypoint','force_contacts','feasible']};path=f['path_'+family]
    teacher=PathTeacher(case,path);row,traj,plans=rollout(_ENV,None,case,planner_override=teacher,execute=1)
    folder=Path(out)/str(case['kind'])/family;folder.mkdir(parents=True,exist_ok=True);target=folder/(Path(source).stem+'.npz');np.savez_compressed(target,**case,**traj,reference_path=path)
    return {'source':str(source),'path_family':family,'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'saved':str(target),**row}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=8);p.add_argument('--workers',type=int,default=8);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for kind in ['supported_easy','supported_conflict','coupled_ood']:
        for source in sorted(Path(args.source).glob(kind+'_*.npz'))[:args.count]:
            for family in ['K','W']:jobs.append((source,family,str(out)))
    started=time.perf_counter()
    with mp.get_context('spawn').Pool(args.workers,initializer=init) as pool:rows=list(pool.imap(run,jobs,chunksize=1))
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    for kind in ['supported_easy','supported_conflict','coupled_ood']:
        for family in ['K','W']:
            selected=[r for r in rows if r['kind']==kind and r['path_family']==family];print(kind,family,sum(r['success'] for r in selected),'/',len(selected),flush=True)
    (out/'DONE.json').write_text(json.dumps({'complete':True,'episodes':len(rows),'seconds':time.perf_counter()-started,'scope':'Hand-coded single-skill path-following reference executing analytic-agreement paths, not a learned controller.'}))


if __name__=='__main__':main()
