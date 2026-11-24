"""Verify exported policy execution, latent invariants, and current runtime."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import numpy as np
import torch
from .env import make_env,reset_context,features,completion,close_env
from .policy import CubePolicy
from .evaluate import load_models
from .calibrate_encoder import digest


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--out",default="artifacts/final/verification.json");p.add_argument("--device",default="cpu");args=p.parse_args()
    torch.set_num_threads(1)
    policy=CubePolicy(args.checkpoint,args.device)
    g,_,c=load_models(args.checkpoint,args.device)
    import cube_gan.expert as expert
    def forbidden(*a,**kw):raise AssertionError("Deployment invoked the training expert")
    expert.select_action=forbidden
    env=make_env();records=[]
    try:
        for task in range(6):
            for mode in [-1,1]:
                z=np.array([mode*.8,.5],np.float32);policy.reset(z)
                ob,info=reset_context(env,77123+task,task)
                start=info["privileged/block_0_pos"].copy();previous=None;stable=0;native_any=False;error=0.
                for t in range(200):
                    state=features(env,ob,info,start,previous)
                    with torch.inference_mode():
                        if t%8==0:chunk=g(torch.from_numpy(state)[None].to(args.device),torch.from_numpy(z)[None].to(args.device))[0].cpu().numpy()
                    expected=chunk[t%8].copy();expected[4]=1 if expected[4]>=0 else -1
                    action=policy.act(state)
                    error=max(error,float(np.abs(expected-action).max()))
                    np.testing.assert_array_equal(policy.z,z)
                    assert np.isfinite(action).all() and np.max(np.abs(action))<=1
                    previous=info["proprio/effector_pos"].copy()
                    ob,_,_,_,info=env.step(action)
                    native,strict=completion(env,info);native_any|=native;stable=stable+1 if strict else 0
                    if stable>=10:break
                assert error<1e-6
                records.append({"task_id":task,"z":z.tolist(),"native_success":native_any,"released_success":stable>=10,"steps":t+1,"max_api_action_error":error})
    finally:close_env(env)
    versions={name:metadata.version(name) for name in ["torch","numpy","mujoco","ogbench","dm_control","stable-worldmodel","gymnasium","pymunk"]}
    direct=metadata.distribution("stable-worldmodel").read_text("direct_url.json")
    report={"checkpoint":args.checkpoint,"device":args.device,"generator_sha256":digest(g.state_dict()),"episodes":records,"max_api_action_error":max(r["max_api_action_error"] for r in records),
            "native_success":float(np.mean([r["native_success"] for r in records])),"released_success":float(np.mean([r["released_success"] for r in records])),
            "expert_disabled":True,"latent_unchanged":True,"horizon":16,"execute":8,"runtime":{"python":platform.python_version(),"platform":platform.platform(),"versions":versions,"swm_direct_url":json.loads(direct) if direct else None,"gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None}}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=="__main__":main()
