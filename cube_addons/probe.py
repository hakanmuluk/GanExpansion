"""Check candidate skills with the actual frozen policy and real MuJoCo."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("MUJOCO_GL","egl")
import argparse,json,time
from pathlib import Path
import math
import numpy as np
import torch
from cube.cube_gan.evaluate import load_models
from cube.cube_gan.style import decode_style
from cube.cube_gan.expert import Style
from .env import make_env,reset_case,snapshot,restore,features,completion,goal_position,close_env,step_traced
from .geometry import MAX_STEPS,Observer,axis_frame
from .expert import SkillExpert
from .contacts import contact_info,force_holding


def case(env,seed):
    ob,info=reset_case(env,seed);initial=info["privileged/block_0_pos"].copy();goal=goal_position(env)
    state=snapshot(env);rng=np.random.default_rng(seed+53);z=rng.standard_normal(2).astype(np.float32)
    mode,amp,speed=decode_style(torch.tensor(z));style=Style(int(mode),float(amp),float(speed))
    return {"seed":seed,"initial_state":state,"initial_cube":initial,"goal":goal,"base_z":z},style


def constraint(case,style,skill,seed):
    start,goal=case["initial_cube"],case["goal"];_,_,_,perp,_=axis_frame(start,goal,start)
    side=-1 if style.mode==0 else 1;rng=np.random.default_rng(seed+79)
    xy=(start[:2]+goal[:2])/2+side*style.amplitude*perp
    if skill=="waypoint":xy+=side*rng.uniform(.050,.065)*perp
    return np.r_[xy,.20,rng.uniform(.020,.025)]


@torch.no_grad()
def execute(env,base,c,spec,skill,code=None):
    ob,info=restore(env,c["initial_state"]);initial=c["initial_cube"];goal=c["goal"]
    contact_info(env,info)
    observer=Observer(initial,goal,spec,skill,force_holding);style_values=decode_style(torch.tensor(c["base_z"]))
    expert=SkillExpert(initial,goal,Style(*[int(style_values[0]),float(style_values[1]),float(style_values[2])]),spec,code,skill) if code is not None else None
    previous=None;stable=0;poses=[];actions=[];phases=[];obs=[];ctx=[]
    for t in range(MAX_STEPS):
        st=features(env,ob,info,initial,previous);obs.append(st);ctx.append(observer.features(info,t))
        poses.append(np.r_[info["proprio/effector_pos"],info["privileged/block_0_pos"]])
        if expert is not None:a,phase=expert.action(info,observer)
        else:
            if t%8==0:chunk=base(torch.tensor(st)[None],torch.tensor(c["base_z"])[None])[0].numpy()
            a=chunk[t%8].copy();a[4]=1 if a[4]>=0 else -1;phase=-1
        actions.append(a);phases.append(phase);before=info["privileged/block_0_pos"].copy();previous=info["proprio/effector_pos"].copy()
        result,path=step_traced(env,a);ob,_,_,_,info=result;contact_info(env,info);after=info["privileged/block_0_pos"]
        observer.update(before,after,info,path);_,released=completion(env,info);stable=stable+1 if released else 0
        if stable>=10:break
    poses.append(np.r_[info["proprio/effector_pos"],info["privileged/block_0_pos"]])
    row={"released_success":stable>=10,"success":stable>=10 and observer.satisfied(),"visited":observer.visited,
         "violated":observer.violated,"table_violation":observer.table_violation,"has_lifted":observer.has_lifted,
         "held_samples":observer.held_transport_samples,"min_clearance":observer.minimum_clearance if skill=="keepout" else None,"steps":t+1}
    return row,{"states":np.asarray(obs),"contexts":np.asarray(ctx),"poses":np.asarray(poses),"actions":np.asarray(actions),"phase":np.asarray(phases)}


def main():
    p=argparse.ArgumentParser();p.add_argument("--contexts",type=int,default=12);p.add_argument("--start-seed",type=int,default=610001);p.add_argument("--out",default="cube_addon_artifacts/probe");args=p.parse_args()
    torch.set_num_threads(1);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    base,_,_=load_models("cube/artifacts/final/base_policy.pt");env=make_env();rows=[];started=time.time()
    try:
        for i in range(args.contexts):
            seed=args.start_seed+i;c,style=case(env,seed)
            for skill in ["keepout","waypoint"]:
                spec=constraint(c,style,skill,seed)
                for label,code in [("base",None),("negative_low",[-.3]),("negative_high",[-1.5]),("positive_low",[.3]),("positive_high",[1.5])]:
                    row,trajectory=execute(env,base,c,spec,skill,code);row.update(context=i,seed=seed,skill=skill,kind=label,base_z=c["base_z"].tolist(),spec=spec.tolist())
                    rows.append(row);np.savez_compressed(out/f"{skill}_{i}_{label}.npz",**trajectory,goal=c["goal"],initial=c["initial_cube"],initial_state=c["initial_state"],spec=spec)
                    print(json.dumps(row),flush=True)
        (out/"summary.json").write_text(json.dumps({"records":rows,"elapsed":time.time()-started},indent=2))
    finally:close_env(env)


if __name__=="__main__":main()
