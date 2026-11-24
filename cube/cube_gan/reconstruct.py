"""Encode held-out complete demonstrations, then execute G(E(demo)) in MuJoCo."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json
from pathlib import Path
import time
import numpy as np
import torch
from .env import make_env,reset_context,restore,features,close_env
from .evaluate import load_models,init_worker,rollout,summarize


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",default="artifacts/data");p.add_argument("--out",required=True);p.add_argument("--workers",type=int,default=6);args=p.parse_args()
    torch.set_num_threads(1);started=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with np.load(Path(args.data)/"test.npz") as f:d={k:f[k] for k in f.files}
    with np.load(Path(args.data)/"test_contexts.npz") as f:contexts={k:f[k] for k in f.files}
    _,e,_=load_models(args.checkpoint);codes=[]
    with torch.inference_mode():
        for t in range(0,len(d["lengths"]),64):
            mu,_=e(torch.from_numpy(d["states"][t:t+64]),torch.from_numpy(d["actions"][t:t+64]),torch.from_numpy(d["lengths"][t:t+64]));codes.append(mu.numpy())
    codes=np.concatenate(codes)
    env=make_env();initials=[]
    for c in range(len(contexts["seed"])):
        reset_context(env,int(contexts["seed"][c]),int(contexts["task_id"][c]))
        ob,info=restore(env,contexts["initial_state"][c])
        initials.append(features(env,ob,info,contexts["initial_cube"][c]))
    close_env(env);initials=np.stack(initials);specs=[]
    for j,z in enumerate(codes):
        error=np.abs(initials-d["states"][j,0]).max(1);c=int(error.argmin())
        assert error[c]<1e-5,(j,error[c])
        spec={k:v[c] for k,v in contexts.items()}
        spec.update(context=c,latent_index=j,z=z,rollout_seed=20000+j,resample=False,execute=8)
        specs.append(spec)
    records=[];trajectories=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn"),initializer=init_worker,initargs=(args.checkpoint,True)) as pool:
        for j,(record,trajectory) in enumerate(pool.map(rollout,specs,chunksize=4)):
            n=max(int(d["lengths"][j]),trajectory["lengths"])+1
            record["demonstrated_mode"]=int(d["mode"][j]);record["demonstrated_route_match"]=record["route"]==int(2*d["mode"][j]-1)
            error=trajectory["poses"][:n,3:]-d["poses"][j,:n,3:]
            record["cube_position_rmse_m"]=float(np.sqrt(np.mean(np.sum(error**2,axis=-1))))
            records.append(record);trajectories.append(trajectory)
            if len(records)%128==0:print(json.dumps({"done":len(records),"total":len(specs),"elapsed":time.time()-started}),flush=True)
    np.savez_compressed(out/"rollouts.npz",**{k:np.stack([r[k] for r in trajectories]) for k in trajectories[0]})
    (out/"records.json").write_text(json.dumps(records,indent=2))
    success=[r for r in records if r["released_success"]]
    report={"args":vars(args),**summarize(records),"demonstrated_route_match":float(np.mean([r["demonstrated_route_match"] for r in records])),
            "demonstrated_route_match_given_success":float(np.mean([r["demonstrated_route_match"] for r in success])) if success else None,
            "mean_time_aligned_cube_rmse_m":float(np.mean([r["cube_position_rmse_m"] for r in records])),"elapsed_seconds":time.time()-started}
    (out/"summary.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=="__main__":main()
