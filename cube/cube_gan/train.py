"""BicycleGAN training; actual MuJoCo evaluation selects checkpoints.

The latent cycle uses generated actions at recorded teacher states. Its metric
is explicitly distinct from encoding actual closed-loop MuJoCo trajectories.
Optional known expert style labels provide a documented auxiliary experiment.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .env import TASK
from .models import ModelConfig,Generator,TrajectoryEncoder,Discriminator,reconstruction,kl_loss
from .dataset import EpisodeDataset
from .style import encode_style


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--data",default="artifacts/data")
    p.add_argument("--steps",type=int,default=4000)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--chunks",type=int,default=8)
    p.add_argument("--cycle-batch",type=int,default=8)
    p.add_argument("--hidden",type=int,default=384)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--warmup",type=int,default=500)
    p.add_argument("--rec-weight",type=float,default=10.)
    p.add_argument("--latent-weight",type=float,default=1.)
    p.add_argument("--kl-weight",type=float,default=.01)
    p.add_argument("--adv-weight",type=float,default=.05)
    p.add_argument("--style-weight",type=float,default=0.)
    p.add_argument("--known-rec-weight",type=float,default=0.)
    p.add_argument("--route-feature",action="store_true")
    p.add_argument("--eval-every",type=int,default=1000)
    p.add_argument("--eval-contexts",type=int,default=32)
    p.add_argument("--eval-latents",type=int,default=4)
    p.add_argument("--feedback")
    p.add_argument("--feedback-weight",type=float,default=5.)
    p.add_argument("--rollout-data")
    p.add_argument("--physical-style-weight",type=float,default=2.)
    p.add_argument("--resume")
    p.add_argument("--device",default="cuda")
    args=p.parse_args()
    run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    device=args.device
    torch.set_num_threads(2)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    config=ModelConfig(hidden=args.hidden,route_feature=args.route_feature)
    g,e,d=Generator(config).to(device),TrajectoryEncoder(config).to(device),Discriminator(config).to(device)
    with np.load(Path(args.data)/"normalization.npz") as norm:
        for model in [g,e,d]:
            model.obs_mean.copy_(torch.as_tensor(norm["mean"],device=device))
            model.obs_std.copy_(torch.as_tensor(norm["std"],device=device))
    ema=deepcopy(g).eval().requires_grad_(False)
    opt_ge=torch.optim.Adam(list(g.parameters())+list(e.parameters()),lr=args.lr,betas=(.5,.99))
    opt_d=torch.optim.Adam(d.parameters(),lr=args.lr*.5,betas=(.5,.99))
    ds=EpisodeDataset(Path(args.data)/"train.npz",device)
    feedback=None
    if args.feedback:
        with np.load(args.feedback) as f: feedback={k:torch.as_tensor(f[k],device=device) for k in ["states","actions","z","mask"]}
    physical=None
    if args.rollout_data:
        with np.load(args.rollout_data) as f:physical={k:torch.as_tensor(f[k],device=device) for k in f.files}
    start_step,best_score=0,-1.
    if args.resume:
        previous=torch.load(args.resume,map_location=device,weights_only=False)
        assert previous["model"]==asdict(config),"Resume architecture must match"
        for model,key in [(g,"generator"),(e,"encoder"),(d,"discriminator"),(ema,"generator_ema")]:model.load_state_dict(previous[key])
        opt_ge.load_state_dict(previous["optimizer_ge"]);opt_d.load_state_dict(previous["optimizer_d"])
        for group in opt_ge.param_groups:group["lr"]=args.lr
        for group in opt_d.param_groups:group["lr"]=args.lr*.5
        start_step=previous["step"]
        if Path(args.resume).parent==run:best_score=previous.get("best_score",-1.)
    info={"args":vars(args),"model":asdict(config),"parameters":{key:sum(p.numel() for p in m.parameters()) for key,m in [("generator",g),("encoder",e),("discriminator",d)]},"torch":torch.__version__,"gpu":torch.cuda.get_device_name() if device.startswith("cuda") else "cpu"}
    (run/"config.json").write_text(json.dumps(info,indent=2))
    print("CONFIG",json.dumps(info),flush=True)
    started=time.time();running={}
    def save(path,step):
        checkpoint={"model":asdict(config),"args":vars(args),"generator":g.state_dict(),"encoder":e.state_dict(),"discriminator":d.state_dict(),"generator_ema":ema.state_dict(),"optimizer_ge":opt_ge.state_dict(),"optimizer_d":opt_d.state_dict(),"step":step,"best_score":best_score,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device.startswith("cuda") else None}
        torch.save(checkpoint,path.with_suffix(".tmp"));path.with_suffix(".tmp").replace(path)
    with (run/"train.jsonl").open("a",buffering=1) as log:
        for step in range(start_step+1,args.steps+1):
            g.train();e.train();d.train()
            b=ds.sample(args.batch,args.chunks)
            mu,logvar=e(b["episode_states"],b["episode_actions"],b["lengths"])
            z=mu+torch.randn_like(mu)*(.5*logvar).exp()*min(1.,step/1000)
            known_z=encode_style(b["mode"],b["amplitude"],b["speed"])
            if args.style_weight>0:
                mix=min(1.,step/1000)
                z=(1-mix)*known_z+mix*z
            state,target,mask=b["states"],b["actions"],b["mask"]
            expand=lambda x:x[:,None].expand(-1,args.chunks,-1)
            rec_actions=g(state,expand(z))
            prior_z=torch.randn_like(z)
            prior_actions=g(state,expand(prior_z))
            gan_scale=args.adv_weight*min(1.,max(0.,(step-args.warmup)/300))
            d_loss=torch.zeros((),device=device)
            if gan_scale>0:
                d.requires_grad_(True);opt_d.zero_grad(set_to_none=True)
                s=state.flatten(0,1).detach();m=mask.flatten(0,1)
                real=target.flatten(0,1).detach().requires_grad_(step%16==0)
                real_logits=d(s,real,m)
                fakes=torch.cat([rec_actions.detach().flatten(0,1),prior_actions.detach().flatten(0,1)])
                fake_logits=d(s.repeat(2,1),fakes,m.repeat(2,1))
                d_loss=F.relu(1-real_logits).mean()+F.relu(1+fake_logits).mean()
                if step%16==0:
                    gradient=torch.autograd.grad(real_logits.sum(),real,create_graph=True)[0]
                    d_loss=d_loss+8*gradient.square().flatten(1).sum(1).mean()
                d_loss.backward();torch.nn.utils.clip_grad_norm_(d.parameters(),5);opt_d.step()
            d.requires_grad_(False);opt_ge.zero_grad(set_to_none=True)
            rec=reconstruction(rec_actions,target,mask)
            kl=kl_loss(mu,logvar)
            cb=min(args.cycle_batch,args.batch)
            cz=torch.randn((cb,config.z_dim),device=device)
            es=b["episode_states"][:cb,:-1]
            synthetic=g(es,cz[:,None].expand(-1,TASK.max_steps,-1))[:,:,0]
            recovered,_=e(es,synthetic,b["lengths"][:cb])
            lat=F.l1_loss(recovered,cz)
            adv=torch.zeros((),device=device)
            if gan_scale>0:
                fake=torch.cat([rec_actions.flatten(0,1),prior_actions.flatten(0,1)])
                adv=-d(state.flatten(0,1).repeat(2,1),fake,mask.flatten(0,1).repeat(2,1)).mean()
            style=F.l1_loss(mu,known_z) if args.style_weight>0 else torch.zeros((),device=device)
            known=reconstruction(g(state,expand(known_z)),target,mask) if args.known_rec_weight>0 else torch.zeros((),device=device)
            feedback_loss=torch.zeros((),device=device)
            if feedback is not None:
                idx=torch.randint(len(feedback["states"]),(args.batch*args.chunks,),device=device)
                fb={k:v[idx] for k,v in feedback.items()}
                feedback_loss=reconstruction(g(fb["states"],fb["z"]),fb["actions"],fb["mask"])
            physical_style=torch.zeros((),device=device)
            if physical is not None:
                idx=torch.randint(len(physical["states"]),(args.batch,),device=device)
                pb={k:v[idx] for k,v in physical.items()}
                pm,_=e(pb["states"],pb["actions"],pb["lengths"])
                physical_style=F.l1_loss(pm,pb["z"])
            loss=args.rec_weight*rec+args.kl_weight*kl+args.latent_weight*lat+gan_scale*adv+args.style_weight*style+args.known_rec_weight*known+args.feedback_weight*feedback_loss+args.physical_style_weight*physical_style
            if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward();torch.nn.utils.clip_grad_norm_(list(g.parameters())+list(e.parameters()),5);opt_ge.step()
            with torch.no_grad():
                decay=.99 if step>200 else .9
                for ep,gp in zip(ema.parameters(),g.parameters()):ep.lerp_(gp,1-decay)
            values={"loss":loss,"rec":rec,"kl":kl,"latent":lat,"adv":adv,"discriminator":d_loss,"style":style,"known_rec":known,"feedback":feedback_loss,"physical_style":physical_style}
            for k,v in values.items():running[k]=running.get(k,0)+float(v.detach())
            if step%100==0:
                row={"step":step,"elapsed_seconds":time.time()-started,**{k:v/100 for k,v in running.items()}}
                running={};log.write(json.dumps(row)+"\n");print(json.dumps(row),flush=True)
            if step%args.eval_every==0 or step==args.steps:
                save(run/"latest.pt",step)
                eval_out=run/f"val_{step:05d}"
                cmd=[sys.executable,"-m","cube_gan.evaluate","--checkpoint",str(run/"latest.pt"),"--data",args.data,"--split","val","--contexts",str(args.eval_contexts),"--latents",str(args.eval_latents),"--out",str(eval_out)]
                subprocess.run(cmd,check=True)
                metrics=json.loads((eval_out/"summary.json").read_text())
                score=metrics["released_success"]+.1*metrics["both_routes_per_context"]
                if score>best_score:
                    best_score=score;save(run/"best.pt",step)
                row={"step":step,"validation":metrics,"best_score":best_score}
                log.write(json.dumps(row)+"\n");print("VALIDATION",json.dumps(row),flush=True)
        save(run/"latest.pt",args.steps)
    (run/"completed.json").write_text(json.dumps({"step":args.steps,"best_score":best_score,"elapsed_seconds":time.time()-started},indent=2))


if __name__=="__main__":main()
