"""Independent NumPy swept checks and replay in the actual SWM environment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tworoom_gan.env import make_reference,reset_reference,TASK
from .evaluate import load,contexts,evaluate


def numpy_geometry(positions,spec,lengths,skill):
    a,b=positions[:,:-1],positions[:,1:];delta=b-a
    u=((spec[:,None,:2]-a)*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-8)
    nearest=a+np.clip(u,0,1)[...,None]*delta
    d=np.linalg.norm(nearest-spec[:,None,:2],axis=-1)
    mask=np.arange(a.shape[1])[None]<lengths[:,None]
    if skill=="keepout":return ~((d<spec[:,None,2]+7-1e-5)&mask).any(1)
    return ((d<=spec[:,None,2])&mask).any(1)


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--skill",required=True)
    p.add_argument("--out",required=True);p.add_argument("--count",type=int,default=64);args=p.parse_args()
    torch.set_num_threads(1);model,encoder,ck=load(args.checkpoint)
    d=contexts(Path("addon_artifacts/data")/args.skill/"test_contexts.npz","cuda",args.count)
    with torch.no_grad():_,r=evaluate(model,encoder,d,args.skill,codes=4)
    r={k:v.cpu().numpy() for k,v in r.items()}
    # Include failures explicitly, then evenly cover the remaining rollouts.
    failures=np.flatnonzero(~r["success"])[:args.count//4]
    remaining=np.setdiff1d(np.arange(len(r["lengths"])),failures)
    regular=remaining[np.linspace(0,len(remaining)-1,args.count-len(failures),dtype=int)]
    indices=np.sort(np.concatenate([failures,regular]))
    env=make_reference();max_error=0.;reference_positions=[]
    for index in indices:
        reset_reference(env,r["start"][index],r["goal"][index])
        pp=[r["start"][index]]
        for t in range(int(r["lengths"][index])):
            obs,_,_,_,_=env.step(r["actions"][index,t])
            pp.append(np.asarray(obs[:2]).copy())
        p_array=np.stack(pp)
        max_error=max(max_error,float(np.max(np.abs(p_array-r["positions"][index,:len(pp)]))))
        padded=np.repeat(p_array[-1:],241,0);padded[:len(pp)]=p_array;reference_positions.append(padded)
    replay=np.stack(reference_positions)
    constraint=numpy_geometry(replay,r["spec"][indices],r["lengths"][indices],args.skill)
    reached=np.linalg.norm(replay[:,-1]-r["goal"][indices],axis=-1)<16
    success=constraint&reached
    agreement=bool(np.array_equal(success,r["success"][indices]))
    report={"skill":args.skill,"native_environment":"stable_worldmodel.envs.two_room.TwoRoomEnv",
            "upstream_commit":"21446f1ede6d5284e981bd7b47f432b994e6d812","reference_episodes":len(indices),
            "reference_steps":int(r["lengths"][indices].sum()),"max_position_error_pixels":max_error,
            "numpy_constraint_and_goal_agreement":agreement,"native_skill_success":float(success.mean()),
            "expected_subset_success":float(r["success"][indices].mean()),"explicit_failure_cases":len(failures),
            "subset_selection":"Failures (up to one quarter) plus evenly spaced remaining rollouts; subset success is not a population estimate.",
            "termination":"Waypoint termination requires observed waypoint visit and goal; native dynamics unchanged, native goal flag alone does not end modifier task." if args.skill=="waypoint" else "Goal terminates the episode; success also requires no swept forbidden-body intersection.",
            "checkpoint":args.checkpoint}
    assert max_error<.003,report
    assert agreement,report
    Path(args.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=="__main__":main()
