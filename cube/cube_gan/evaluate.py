"""Closed-loop CPU evaluation in actual MuJoCo, with saved full trajectories."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import json
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from .env import TASK,make_env,reset_context,restore,features,completion,goal_position
from .models import ModelConfig,Generator,TrajectoryEncoder,reconstruction
from .style import encode_style
from .dataset import EpisodeDataset
from .metrics import carry_geometry

_ENV=_G=_E=None


def load_models(path,device="cpu",ema=True):
    c=torch.load(path,map_location=device,weights_only=False)
    config=ModelConfig(**c["model"])
    g,e=Generator(config).to(device),TrajectoryEncoder(config).to(device)
    g.load_state_dict(c.get("generator_ema",c["generator"]) if ema else c["generator"])
    e.load_state_dict(c["encoder"])
    return g.eval(),e.eval(),c


def init_worker(checkpoint,ema):
    global _ENV,_G,_E
    torch.set_num_threads(1)
    _G,_E,_=load_models(checkpoint,ema=ema)
    _ENV=make_env()


def rollout(spec):
    env,g,e=_ENV,_G,_E
    reset_context(env,int(spec["seed"]),int(spec["task_id"]))
    ob,info=restore(env,spec["initial_state"])
    start=np.asarray(spec["initial_cube"])
    goal=goal_position(env)
    z=np.asarray(spec["z"],np.float32)
    rng=np.random.default_rng(spec["rollout_seed"])
    states=np.zeros((TASK.max_steps+1,TASK.observation_dim),np.float32)
    actions=np.zeros((TASK.max_steps,TASK.action_dim),np.float32)
    poses=np.zeros((TASK.max_steps+1,6),np.float32)
    applied_z=np.zeros((TASK.max_steps,2),np.float32)
    stable,native_any,previous_ee=0,False,None
    with torch.inference_mode():
        for t in range(TASK.max_steps):
            state=features(env,ob,info,start,previous_ee)
            states[t]=state
            poses[t]=np.r_[info["proprio/effector_pos"],info["privileged/block_0_pos"]]
            if t%spec.get("execute",TASK.execute)==0:
                current_z=rng.standard_normal(2).astype(np.float32) if spec.get("resample",False) else z
                chunk=g(torch.from_numpy(state)[None],torch.from_numpy(current_z)[None])[0].numpy()
            a=chunk[t%spec.get("execute",TASK.execute)].copy()
            applied_z[t]=current_z
            a[4]=1. if a[4]>=0 else -1.
            actions[t]=a
            previous_ee=info["proprio/effector_pos"].copy()
            ob,_,_,_,info=env.step(a)
            native,strict=completion(env,info)
            native_any|=native
            stable=stable+1 if strict else 0
            if stable>=TASK.stable_steps:
                break
        length=t+1
        applied_z[length:]=current_z
        states[length:]=features(env,ob,info,start,previous_ee)
        poses[length:]=np.r_[info["proprio/effector_pos"],info["privileged/block_0_pos"]]
        mu,_=e(torch.from_numpy(states)[None],torch.from_numpy(actions)[None],torch.tensor([length]))
    xyz=poses[:length+1,3:]
    record={"context":spec["context"],"task_id":int(spec["task_id"]),"latent_index":spec["latent_index"],
            "native_success":bool(native_any),"released_success":bool(stable>=TASK.stable_steps),"steps":length,
            "final_distance":float(np.linalg.norm(xyz[-1]-goal)),**carry_geometry(poses,length,start,goal),
            "z":z.tolist(),"recovered_z":mu[0].tolist()}
    return record,{"states":states,"actions":actions,"poses":poses,"lengths":length,"z":z,"applied_z":applied_z,"initial_cube":start,"goal":goal}


def summarize(records):
    def rate(rs,key):return float(np.mean([r[key] for r in rs])) if rs else None
    per_task={str(i):{"n":len(rs),"native_success":rate(rs,"native_success"),"released_success":rate(rs,"released_success")}
              for i in range(6) if (rs:=[r for r in records if r["task_id"]==i])}
    coverage=[]
    for c in sorted({r["context"] for r in records}):
        routes={r["route"] for r in records if r["context"]==c and r["released_success"]}
        coverage.append(-1 in routes and 1 in routes)
    success=[r for r in records if r["released_success"]]
    recovered=np.asarray([r["recovered_z"] for r in records])
    z=np.asarray([r["z"] for r in records])
    rmse=np.sqrt(np.mean((recovered-z)**2,axis=0)).tolist()
    aligned=[r["route"]==int(np.sign(r["z"][0])) for r in success]
    speed_z=[];speed_steps=[];amplitude_z=[];amplitude_value=[]
    for c in sorted({r["context"] for r in success}):
        group=[r for r in success if r["context"]==c]
        if len(group)>1:
            steps=np.array([r["steps"] for r in group],float)
            zz=np.array([r["z"][1] for r in group])
            speed_z.extend(zz-zz.mean());speed_steps.extend(steps-steps.mean())
        for route in [-1,1]:
            group=[r for r in success if r["context"]==c and r["route"]==route]
            if len(group)>1:
                aa=np.abs([r["carry_amplitude"] for r in group]);zz=np.abs([r["z"][0] for r in group])
                amplitude_z.extend(zz-zz.mean());amplitude_value.extend(aa-aa.mean())
    correlation=lambda a,b:float(spearmanr(a,b).statistic) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else None
    return {"attempts":len(records),"contexts":len(coverage),"native_success":rate(records,"native_success"),
            "released_success":rate(records,"released_success"),"both_routes_per_context":float(np.mean(coverage)),
            "route_matches_latent_given_success":float(np.mean(aligned)) if aligned else None,
            "successful_route_counts":{str(k):sum(r["route"]==k for r in success) for k in [-1,0,1]},
            "mean_steps_success":float(np.mean([r["steps"] for r in success])) if success else None,
            "mean_final_distance":float(np.mean([r["final_distance"] for r in records])),
            "mean_final_distance_success":float(np.mean([r["final_distance"] for r in success])) if success else None,
            "released_within_2cm":float(np.mean([r["released_success"] and r["final_distance"]<=.02 for r in records])),
            "mean_carry_fit_rmse_success":float(np.mean([r["carry_fit_rmse"] for r in success if r.get("carry_fit_rmse") is not None])) if any(r.get("carry_fit_rmse") is not None for r in success) else None,
            "no_carry_sign_change_given_success":float(np.mean([r.get("carry_sign_changes",0)==0 for r in success])) if success else None,
            "within_context_speed_spearman":correlation(speed_z,speed_steps),
            "within_context_side_amplitude_spearman":correlation(amplitude_z,amplitude_value),
            "closed_loop_latent_rmse":rmse,"per_task":per_task}


def select_contexts(task_ids,n):
    """Stratify partial validation so every native goal is represented."""
    if n>=len(task_ids):return np.arange(len(task_ids))
    tasks,counts=np.unique(task_ids,return_counts=True)
    allocation=n*counts/len(task_ids)
    take=np.floor(allocation).astype(int)
    for j in np.argsort(-(allocation-take))[:n-int(take.sum())]:take[j]+=1
    return np.sort(np.concatenate([np.flatnonzero(task_ids==t)[:k] for t,k in zip(tasks,take)]))


def offline(checkpoint,data,seed=707,batch=256):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    g,e,_=load_models(checkpoint)
    ds=EpisodeDataset(data,"cpu")
    with torch.inference_mode():
        b=ds.sample(min(batch,ds.size),4)
        mu,logvar=e(b["episode_states"],b["episode_actions"],b["lengths"])
        z=encode_style(b["mode"],b["amplitude"],b["speed"])
        repeated=mu[:,None].expand(-1,4,-1)
        rec=g(b["states"],repeated)
        prior=g(b["states"],z[:,None].expand_as(repeated))
        cz=torch.randn_like(z)
        synthetic=g(b["episode_states"][:,:-1],cz[:,None].expand(-1,TASK.max_steps,-1))[:,:,0]
        recovered,_=e(b["episode_states"],synthetic,b["lengths"])
        return {"reconstruction_loss":float(reconstruction(rec,b["actions"],b["mask"])),
                "known_style_reconstruction_loss":float(reconstruction(prior,b["actions"],b["mask"])),
                "expert_style_encoder_rmse":((mu-z)**2).mean(0).sqrt().tolist(),
                "teacher_state_cycle_rmse":((recovered-cz)**2).mean(0).sqrt().tolist()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",default="artifacts/data")
    p.add_argument("--split",default="test",choices=["val","test"])
    p.add_argument("--contexts",type=int,default=0)
    p.add_argument("--latents",type=int,default=8)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--seed",type=int,default=8291)
    p.add_argument("--out",required=True)
    p.add_argument("--resample",action="store_true")
    p.add_argument("--shared-codes",action="store_true")
    p.add_argument("--zero",action="store_true")
    p.add_argument("--raw",action="store_true")
    p.add_argument("--execute",type=int,default=8)
    args=p.parse_args()
    started=time.time()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with np.load(Path(args.data)/f"{args.split}_contexts.npz") as d:
        contexts={k:d[k] for k in d.files}
    n=min(args.contexts or len(contexts["seed"]),len(contexts["seed"]))
    rng=np.random.default_rng(args.seed)
    shared=rng.standard_normal((args.latents,2)) if args.shared_codes else None
    specs=[]
    for c in select_contexts(contexts["task_id"],n):
        for j in range(args.latents):
            spec={k:v[c] for k,v in contexts.items()}
            z=np.zeros(2) if args.zero else shared[j] if shared is not None else rng.standard_normal(2)
            spec.update(context=int(c),latent_index=j,z=z,rollout_seed=int(rng.integers(2**31)),resample=args.resample,execute=args.execute)
            specs.append(spec)
    records=[];trajs=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn"),initializer=init_worker,initargs=(args.checkpoint,not args.raw)) as pool:
        for record,trajectory in pool.map(rollout,specs,chunksize=4):
            records.append(record);trajs.append(trajectory)
            if len(records)%128==0:print(json.dumps({"done":len(records),"total":len(specs),"elapsed":time.time()-started}),flush=True)
    np.savez_compressed(out/"rollouts.npz",**{k:np.stack([t[k] for t in trajs]) for k in trajs[0]})
    (out/"records.json").write_text(json.dumps(records,indent=2))
    report={"args":vars(args),"route_metric_version":2,**summarize(records),"offline":offline(args.checkpoint,Path(args.data)/f"{args.split}.npz"),"elapsed_seconds":time.time()-started}
    if args.resample:
        for key in ["closed_loop_latent_rmse","route_matches_latent_given_success","within_context_speed_spearman","within_context_side_amplitude_spearman"]:report[key]=None
        report["latent_target_note"]="No single episode latent exists in the resampling ablation."
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
