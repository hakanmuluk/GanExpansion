"""Four fixed-latent learned policies, rendered by the original MuJoCo env."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from matplotlib import font_manager
from .env import TASK,make_env,reset_context,snapshot,restore,features,completion,close_env
from .evaluate import load_models


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--out",default="artifacts/figures/policy_rollouts.gif");p.add_argument("--task",type=int,default=4);args=p.parse_args()
    torch.set_num_threads(1);g,_,_=load_models(args.checkpoint)
    zs=[[-1.,-.8],[1.,-.8],[-1.,.8],[1.,.8]]
    views=[];report=[]
    for z in zs:
        env=make_env(280)
        ob,info=reset_context(env,20261122,args.task)
        start=info["privileged/block_0_pos"].copy();previous=None;stable=0;frames=[]
        with torch.inference_mode():
            for t in range(200):
                if t%2==0:frames.append(Image.fromarray(env.render()))
                if t%TASK.execute==0:
                    x=features(env,ob,info,start,previous)
                    chunk=g(torch.from_numpy(x)[None],torch.tensor([z]))[0].numpy()
                a=chunk[t%TASK.execute].copy();a[4]=1 if a[4]>=0 else -1
                previous=info["proprio/effector_pos"].copy()
                ob,_,_,_,info=env.step(a)
                native,strict=completion(env,info);stable=stable+1 if strict else 0
                if stable>=10:break
        frames.append(Image.fromarray(env.render()));views.append(frames)
        report.append({"z":z,"steps":t+1,"released_success":stable>=10,"native_final":native})
        close_env(env)
    font=ImageFont.truetype(font_manager.findfont("DejaVu Sans"),13)
    frames=[]
    for t in range(max(map(len,views))+12):
        canvas=Image.new("RGB",(572,688),"white");draw=ImageDraw.Draw(canvas)
        draw.text((8,6),"Learned Cube generator | same reset | fixed Gaussian latent",font=font,fill="#202428")
        draw.text((8,25),"16 predict / 8 execute | translucent cube = goal",font=font,fill="#50565b")
        for j,v in enumerate(views):
            x=6+286*(j%2);y=82+310*(j//2)
            canvas.paste(v[min(t,len(v)-1)],(x,y))
            status=("settled" if report[j]["released_success"] else "timeout") if t>=len(v)-1 else "moving"
            draw.text((x,y-20),f"z={zs[j]} | {status}",font=font,fill="#202428")
        frames.append(canvas)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(out,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
    frames[min(22,len(frames)-1)].save(out.with_name("policy_keyframe.png"))
    out.with_suffix(".json").write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=="__main__":main()
