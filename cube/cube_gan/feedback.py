"""DAgger labels at learned-policy states, using real MuJoCo expert futures.

This is training-time oracle supervision. Inference uses G alone. Both the
learner trajectory and each expert future are simulated by the original env.
"""
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
from .env import TASK,make_env,reset_context,restore,snapshot,features,completion
from .expert import Style,select_action
from .style import decode_style
from .evaluate import load_models

_ENV=_G=None


def init_worker(path):
    global _ENV,_G
    torch.set_num_threads(1)
    _ENV=make_env();_G,_,_=load_models(path)


def collect(spec):
    env,g=_ENV,_G
    reset_context(env,int(spec["seed"]),int(spec["task_id"]))
    ob,info=restore(env,spec["initial_state"])
    start=spec["initial_cube"]
    z=np.asarray(spec["z"],np.float32)
    m,a,s=decode_style(torch.from_numpy(z))
    style=Style(int(m),float(a),float(s))
    rng=np.random.default_rng(spec["rollout_seed"])
    states=[];targets=[];previous_ee=None;stable=0;native_any=False
    rollout_states=np.zeros((TASK.max_steps+1,TASK.observation_dim),np.float32)
    rollout_actions=np.zeros((TASK.max_steps,TASK.action_dim),np.float32)
    with torch.inference_mode():
        for t in range(TASK.max_steps):
            state=features(env,ob,info,start,previous_ee)
            rollout_states[t]=state
            if t%TASK.execute==0:
                saved=snapshot(env)
                states.append(state)
                target=[]
                for h in range(TASK.horizon):
                    action,_=select_action(env,info,start,style)
                    target.append(action)
                    _,_,_,_,info=env.step(action)
                targets.append(np.stack(target))
                ob,info=restore(env,saved)
                chunk=g(torch.from_numpy(state)[None],torch.from_numpy(z)[None])[0].numpy()
                teacher_chunk=rng.random()<spec["teacher_probability"]
            action=chunk[t%TASK.execute].copy()
            if teacher_chunk:action,_=select_action(env,info,start,style)
            else:
                action[:3]+=rng.normal(0,spec["action_noise"],3)
                action=np.clip(action,-1,1)
                action[4]=1 if action[4]>=0 else -1
            previous_ee=info["proprio/effector_pos"].copy()
            rollout_actions[t]=action
            ob,_,_,_,info=env.step(action)
            native,strict=completion(env,info);native_any|=native
            stable=stable+1 if strict else 0
            if stable>=TASK.stable_steps:break
    rollout_states[t+1:]=features(env,ob,info,start,previous_ee)
    return {"states":np.stack(states),"actions":np.stack(targets),"z":np.broadcast_to(z,(len(states),2)).copy(),
            "mask":np.ones((len(states),TASK.horizon),np.float32),
            "rollout_states":rollout_states,"rollout_actions":rollout_actions,"rollout_length":t+1,"rollout_z":z,
            "record":{"task_id":int(spec["task_id"]),"native_success":bool(native_any),"released_success":bool(stable>=TASK.stable_steps),"steps":t+1,"queries":len(states)}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",default="artifacts/data")
    p.add_argument("--out",required=True)
    p.add_argument("--contexts",type=int,default=256)
    p.add_argument("--latents",type=int,default=4)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--seed",type=int,default=991)
    p.add_argument("--teacher-probability",type=float,default=.1)
    p.add_argument("--action-noise",type=float,default=.03)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    started=time.time();rng=np.random.default_rng(args.seed)
    with np.load(Path(args.data)/"train_contexts.npz") as d:contexts={k:d[k] for k in d.files}
    ids=rng.choice(len(contexts["seed"]),min(args.contexts,len(contexts["seed"])),replace=False)
    specs=[]
    for c in ids:
        for _ in range(args.latents):
            spec={k:v[c] for k,v in contexts.items()}
            spec.update(z=rng.standard_normal(2),rollout_seed=int(rng.integers(2**31)),teacher_probability=args.teacher_probability,action_noise=args.action_noise)
            specs.append(spec)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn"),initializer=init_worker,initargs=(args.checkpoint,)) as pool:
        for r in pool.map(collect,specs,chunksize=2):
            results.append(r)
            if len(results)%64==0:print(json.dumps({"done":len(results),"total":len(specs),"elapsed":time.time()-started}),flush=True)
    np.savez_compressed(out/"feedback.npz",**{k:np.concatenate([r[k] for r in results]) for k in ["states","actions","z","mask"]})
    np.savez_compressed(out/"learner_rollouts.npz",**{k:np.stack([r[v] for r in results]) for k,v in [("states","rollout_states"),("actions","rollout_actions"),("lengths","rollout_length"),("z","rollout_z")]})
    records=[r["record"] for r in results]
    report={"args":vars(args),"rollouts":len(results),"expert_chunk_queries":sum(r["queries"] for r in records),
            "native_success":float(np.mean([r["native_success"] for r in records])),"released_success":float(np.mean([r["released_success"] for r in records])),"elapsed_seconds":time.time()-started,"records":records}
    (out/"summary.json").write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!="records"}),flush=True)


if __name__=="__main__":main()
