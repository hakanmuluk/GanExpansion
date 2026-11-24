"""Finite-data BicycleGAN adapters. No new expert queries during training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from tworoom_gan.env import observation,transition,TASK
from .data import EpisodeDataset
from .geometry import skill_features,constraint_loss
from .models import GatedGenerator,AdapterConfig,SkillEncoder,SkillDiscriminator
from .evaluate import evaluate,contexts
from .rollout import rollout
from .feedback import finite_path_feedback


def masked_l1(pred,target,mask):
    return ((pred-target).abs()*mask[...,None]).sum()/(mask.sum().clamp_min(1)*2)


def expand(z,count):return z[:,None].expand(-1,count,-1)


def parser():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--skill",choices=["keepout","waypoint"],required=True)
    p.add_argument("--base",default="artifacts/final/base_policy.pt");p.add_argument("--data",default="addon_artifacts/data")
    p.add_argument("--episodes",type=int,default=2048);p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--batch",type=int,default=64);p.add_argument("--chunks",type=int,default=4);p.add_argument("--cycle-batch",type=int,default=8)
    p.add_argument("--eval-every",type=int,default=250);p.add_argument("--eval-contexts",type=int,default=48);p.add_argument("--eval-codes",type=int,default=4)
    p.add_argument("--rank",type=int,default=4);p.add_argument("--modulation-rank",type=int,default=8);p.add_argument("--context-hidden",type=int,default=64)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--lr",type=float,default=.0005)
    p.add_argument("--base-lr",type=float,default=.0001,help="Learning rate of the cloned backbone in full fine-tuning")
    p.add_argument("--rec-weight",type=float,default=20);p.add_argument("--known-rec-weight",type=float,default=10)
    p.add_argument("--label-weight",type=float,default=1);p.add_argument("--latent-weight",type=float,default=1)
    p.add_argument("--adv-weight",type=float,default=.03);p.add_argument("--kl-weight",type=float,default=.005)
    p.add_argument("--keep-weight",type=float,default=.1);p.add_argument("--constraint-weight",type=float,default=.5)
    p.add_argument("--rollout-weight",type=float,default=.2);p.add_argument("--physics-every",type=int,default=32)
    p.add_argument("--physics-start",type=int,default=800);p.add_argument("--augment",type=float,default=1)
    p.add_argument("--feedback-weight",type=float,default=0);p.add_argument("--feedback-every",type=int,default=16);p.add_argument("--feedback-start",type=int,default=500)
    p.add_argument("--no-new-latent",action="store_true");p.add_argument("--finetune-base",action="store_true")
    p.add_argument("--no-sign-feature",action="store_true");p.add_argument("--device",default="cuda");p.add_argument("--resume")
    p.add_argument("--history",action="store_true")
    p.add_argument("--single-strategy",choices=["positive","negative"],help="Diagnostic no-latent control using one side of the same finite data")
    p.add_argument("--output-modulation",action="store_true");p.add_argument("--tracking-weight",type=float,default=0)
    return p


def main():
    args=parser().parse_args();run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    config=AdapterConfig(rank=args.rank,context_hidden=args.context_hidden,modulation_rank=args.modulation_rank,
                         no_new_latent=args.no_new_latent,finetune_base=args.finetune_base,sign_feature=not args.no_sign_feature,context_dim=14 if args.history else 12,output_modulation=args.output_modulation)
    model=GatedGenerator(args.base,config,args.device)
    preservation_reference=model.base
    if args.finetune_base:
        from tworoom_gan.evaluate import load_checkpoint
        preservation_reference,_,_=load_checkpoint(args.base,args.device)
        preservation_reference.eval().requires_grad_(False)
    encoder=SkillEncoder(context_dim=config.context_dim).to(args.device)
    source=torch.load("artifacts/final/discriminator.pt",map_location="cpu",weights_only=False)["discriminator"]
    discriminator=SkillDiscriminator(model.base_config,source,config.context_dim).to(args.device)
    parameters=model.trainable()+(list(encoder.parameters()) if not args.no_new_latent else [])
    optimizer_parameters=parameters
    if args.finetune_base:
        new_parameters=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith("base.")]
        if not args.no_new_latent:new_parameters+=list(encoder.parameters())
        optimizer_parameters=[{"params":list(model.base.parameters()),"lr":args.base_lr,"name":"backbone"},
                              {"params":new_parameters,"lr":args.lr,"name":"new"}]
    opt=torch.optim.Adam(optimizer_parameters,lr=args.lr,betas=(.5,.99))
    opt_d=torch.optim.Adam(discriminator.parameters(),lr=args.lr*.5,betas=(.5,.99))
    data=EpisodeDataset(Path(args.data)/args.skill/"train.npz",args.device,args.episodes)
    if args.single_strategy:
        if not args.no_new_latent:raise ValueError("--single-strategy is a deterministic no-latent control")
        chosen=data.tensors["new_z"][:,0]>=0
        if args.single_strategy=="negative":chosen=~chosen
        data.tensors={k:v[chosen] for k,v in data.tensors.items()};data.n=int(chosen.sum())
    assert data.tensors["contexts"].shape[-1]==config.context_dim
    val=contexts(Path(args.data)/args.skill/"val_contexts.npz",args.device,args.eval_contexts)
    frozen_hash=model.frozen_digest();base_sha=hashlib.sha256(Path(args.base).read_bytes()).hexdigest()
    start_step=0;best=-1
    if args.resume:
        ck=torch.load(args.resume,map_location=args.device,weights_only=False)
        if ck["base_sha256"]!=base_sha:raise ValueError("Resume base fingerprint differs")
        if asdict(AdapterConfig(**ck["adapter_config"]))!=asdict(config):raise ValueError("Resume architecture flags must match the checkpoint")
        if set(ck["addon"])!=set(model.addon_state()):raise ValueError("Incomplete resume adapter")
        if args.steps<=ck["step"]:raise ValueError("--steps must extend beyond the resume checkpoint step")
        if "optimizer" not in ck:raise ValueError("Use the *_resume.pt training checkpoint, not the compact inference checkpoint")
        model.load_state_dict(ck["addon"],strict=False);encoder.load_state_dict(ck["encoder"])
        if "finetuned_base" in ck:model.base.load_state_dict(ck["finetuned_base"])
        discriminator.load_state_dict(ck["discriminator"]);opt.load_state_dict(ck["optimizer"]);opt_d.load_state_dict(ck["optimizer_d"])
        start_step=ck["step"];best=ck["best_score"] if Path(args.resume).parent.resolve()==run.resolve() else -1
        torch.set_rng_state(ck["rng"].cpu())
        if args.device.startswith("cuda"):torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        for group in opt.param_groups:group["lr"]=args.base_lr if group.get("name")=="backbone" else args.lr
        for group in opt_d.param_groups:group["lr"]=args.lr*.5
    info={"args":vars(args),"adapter_config":asdict(config),"base_sha256":base_sha,"normalization":"pixel / 112 - 1; fixed LayerNorm without running statistics",
          "counts":{"base_generator":sum(p.numel() for p in model.base.parameters()),"original_encoder":sum(p.numel() for p in model.original_encoder.parameters()),
                    "new_actor":sum(p.numel() for n,p in model.named_parameters() if not n.startswith(("base.","original_encoder."))),
                    "actor_trainable":sum(p.numel() for p in model.trainable()),"new_encoder":sum(p.numel() for p in encoder.parameters()),"new_discriminator":sum(p.numel() for p in discriminator.parameters())},
          "torch":torch.__version__,"gpu":torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu","expert_action_queries_during_training":0,
          "training_data_sha256":hashlib.sha256((Path(args.data)/args.skill/"train.npz").read_bytes()).hexdigest(),
          "training_demonstrations_used":data.n,"available_prefix_demonstrations":args.episodes,
          "keep_reference":"Immutable original pretrained generator, including for the full-finetuning comparator.",
          "auxiliary_supervision":"Known Gaussian style labels of finite demonstrations for E and paired reconstruction; set weights to zero for vanilla."}
    (run/"config.json").write_text(json.dumps(info,indent=2));print("CONFIG",json.dumps(info),flush=True)
    started=time.time();running={};last_log=start_step
    with (run/"train.jsonl").open("a",buffering=1) as log:
        for step in range(start_step+1,args.steps+1):
            model.train();encoder.train();discriminator.train();b=data.sample(args.batch,args.chunks)
            state,ctx,target,mask=b["states"],b["contexts"],b["actions"],b["mask"]
            z=expand(b["base_z"],args.chunks);true_code=b["new_z"]
            if args.augment>0:
                with torch.no_grad():
                    pos=(state[...,:2]+1)*112;goal=(state[...,2:4]+1)*112
                    perturbed,_=transition(pos,torch.randn_like(pos)*args.augment/5)
                    state=observation(perturbed,goal)
                    adjusted=skill_features(perturbed,goal,expand(b["start"],args.chunks),expand(b["spec"],args.chunks),ctx[...,8]>.5,ctx[...,9]*240,args.skill)
                    ctx=torch.cat([adjusted,ctx[...,-2:]],-1) if config.context_dim==14 else adjusted
                    # Track only the stored finite demonstration positions.
                    reference=pos;current=perturbed;corrected=[]
                    speed=target.norm(dim=-1).amax(-1,keepdim=True).clamp_min(.1)
                    for j in range(16):
                        reference,_=transition(reference,target[...,j,:])
                        delta=(reference-current)/5
                        action=delta*(speed/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)).clamp_max(1)
                        corrected.append(action);current,_=transition(current,action)
                    target=torch.stack(corrected,-2)
            zero=state.new_zeros(())
            if args.no_new_latent:
                mu=lv=torch.zeros_like(true_code);code=mu
            else:
                mu,lv=encoder(b["episode_states"],b["episode_actions"],b["episode_contexts"],b["base_z"],b["lengths"])
                encoded=mu+torch.randn_like(mu)*torch.exp(.5*lv)*min(1.,step/1000)
                blend=min(1.,step/1000) if args.label_weight else 1.
                code=(1-blend)*true_code+blend*encoded
            reconstructed=model(state,z,expand(code,args.chunks),ctx)
            prior=torch.randn_like(true_code)
            fake=model(state,z,expand(prior,args.chunks),ctx)
            gan=args.adv_weight*min(1.,max(0.,(step-300)/300))
            dl=zero;adv=zero
            if gan:
                discriminator.requires_grad_(True);opt_d.zero_grad(set_to_none=True)
                s=state.flatten(0,1).detach();c=ctx.flatten(0,1).detach();m=mask.flatten(0,1)
                real_log=discriminator(s,target.flatten(0,1).detach(),c,m)
                fake_log=discriminator(s.repeat(2,1),torch.cat([reconstructed.detach().flatten(0,1),fake.detach().flatten(0,1)]),c.repeat(2,1),m.repeat(2,1))
                dl=F.relu(1-real_log).mean()+F.relu(1+fake_log).mean();dl.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(),5);opt_d.step()
                discriminator.requires_grad_(False)
                adv=-discriminator(s.repeat(2,1),torch.cat([reconstructed.flatten(0,1),fake.flatten(0,1)]),c.repeat(2,1),m.repeat(2,1)).mean()
            opt.zero_grad(set_to_none=True)
            rec=masked_l1(reconstructed,target,mask)
            known=zero
            if args.known_rec_weight and not args.no_new_latent:
                known=masked_l1(model(state,z,expand(true_code,args.chunks),ctx),target,mask)
            kl=.5*(mu.square()+lv.exp()-1-lv).mean() if not args.no_new_latent else zero
            label=F.l1_loss(mu,true_code) if not args.no_new_latent else zero
            cycle=zero
            if not args.no_new_latent and args.latent_weight and step%4==0:
                cb=args.cycle_batch;ep_s=b["episode_states"][:cb];ep_c=b["episode_contexts"][:cb]
                code_prior=torch.randn_like(true_code[:cb])
                synthetic=model(ep_s,expand(b["base_z"][:cb],240),expand(code_prior,240),ep_c)[:,:,0]
                recovered,_=encoder(ep_s,synthetic,ep_c,b["base_z"][:cb],b["lengths"][:cb])
                cycle=F.l1_loss(recovered,code_prior)
            with torch.no_grad():base_actions=preservation_reference(state,z)
            # Proposal L_keep: UNSQUARED L2 per action, averaged over valid steps.
            keep=(((fake-base_actions).norm(dim=-1)+(reconstructed-base_actions).norm(dim=-1))*.5*mask).sum()/mask.sum().clamp_min(1)
            pos=(state[...,:2]+1)*112;positions=[pos]
            for j in range(16):pos,_=transition(pos,reconstructed[...,j,:]);positions.append(pos)
            short=torch.stack(positions,-2).flatten(0,1)
            con=constraint_loss(short,expand(b["goal"],args.chunks).flatten(0,1),expand(b["spec"],args.chunks).flatten(0,1),args.skill,mask.sum(-1).flatten().long()) if args.skill=="keepout" else zero
            tracking=zero
            if args.tracking_weight:
                current=(state[...,:2]+1)*112;reference_positions=[]
                for j in range(16):current,_=transition(current,target[...,j,:]);reference_positions.append(current)
                reference=torch.stack(reference_positions,-2).flatten(0,1)
                tracking=((short[:,1:]-reference).norm(dim=-1)*mask.flatten(0,1)).sum()/mask.sum().clamp_min(1)/50
            physical=zero;roll_cycle=zero
            if step>=args.physics_start and step%args.physics_every==0 and args.rollout_weight:
                cb=min(8,args.batch);rz=torch.randn_like(true_code[:cb])
                r=rollout(model,b["start"][:cb],b["goal"][:cb],b["base_z"][:cb],rz,b["spec"][:cb],args.skill)
                physical=constraint_loss(r["positions"],r["goal"],r["spec"],args.skill,r["lengths"])
                physical=physical+torch.relu((r["positions"][:,-1]-r["goal"]).norm(dim=-1)-15).mean()/50
                if not args.no_new_latent:
                    recovered,_=encoder(r["states"],r["actions"],r["contexts"],r["base_z"],r["lengths"])
                    roll_cycle=F.l1_loss(recovered,rz)
            feedback=zero
            if args.feedback_weight and step>=args.feedback_start and step%args.feedback_every==0:
                feedback_batch=b
                if not args.no_new_latent and not args.label_weight and not args.known_rec_weight:
                    # Unsupervised ablation uses inferred demonstration codes;
                    # it must not obtain known synthetic codes via feedback.
                    feedback_batch={**b,"new_z":mu.detach()}
                fs,fc,fz,fcode,ft=finite_path_feedback(model,feedback_batch,args.skill)
                if len(fs):feedback=F.l1_loss(model(fs,fz,fcode,fc),ft)
            loss=args.rec_weight*rec+args.known_rec_weight*known+args.label_weight*label+args.kl_weight*kl+args.latent_weight*cycle+gan*adv+args.keep_weight*keep+args.constraint_weight*con+args.rollout_weight*(physical+roll_cycle)+args.feedback_weight*feedback+args.tracking_weight*tracking
            if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at {step}")
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(parameters,5);opt.step()
            values={"loss":loss,"rec":rec,"known_rec":known,"label":label,"kl":kl,"teacher_cycle":cycle,"adv":adv,"d_loss":dl,"keep":keep,"constraint":con,"physical":physical,"rollout_cycle":roll_cycle,"feedback":feedback,"tracking":tracking,"grad":grad}
            for k,v in values.items():running[k]=running.get(k,0)+float(v.detach())
            if step%100==0 or step==args.steps:
                row={"step":step,"elapsed_seconds":time.time()-started,**{k:v/(step-last_log) for k,v in running.items()}}
                log.write(json.dumps(row)+"\n");print("TRAIN",json.dumps(row),flush=True);running={};last_log=step
            if step%args.eval_every==0 or step==args.steps:
                metric,_=evaluate(model,encoder,val,args.skill,args.eval_codes)
                alignment=.05*(metric["strategy_matches_code"] or 0) if args.label_weight else 0
                score=metric["style_preserving_success"]+.2*metric["both_successful_strategies_per_context"]+alignment-.02*metric.get("closed_loop_latent_rmse",0)
                eligible=step>=min(args.steps,600)
                improved=eligible and score>best
                if eligible:best=max(best,score)
                row={"step":step,"elapsed_seconds":time.time()-started,"score":score,"eligible":eligible,**metric}
                with (run/"validation.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
                print("VALIDATION",json.dumps(row),flush=True)
                if not args.finetune_base:assert model.frozen_digest()==frozen_hash,"Frozen tensors changed"
                ck={"step":step,"args":vars(args),"adapter_config":asdict(config),"addon":model.addon_state(),"encoder":encoder.state_dict(),
                    "base_sha256":base_sha,"frozen_digest_before":frozen_hash,"frozen_digest_after":model.frozen_digest(),"best_score":best,"validation":metric,
                    "discriminator":discriminator.state_dict(),"optimizer":opt.state_dict(),"optimizer_d":opt_d.state_dict(),"rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state() if args.device.startswith("cuda") else None}
                if args.finetune_base:ck["finetuned_base"]=model.base.state_dict()
                torch.save(ck,run/"latest.tmp");(run/"latest.tmp").replace(run/"latest.pt")
                if improved:
                    compact={k:v for k,v in ck.items() if k not in ["optimizer","optimizer_d","rng","cuda_rng"]}
                    torch.save(compact,run/"best.tmp");(run/"best.tmp").replace(run/"best.pt")
    (run/"DONE.json").write_text(json.dumps({"step":args.steps,"best_score":best,"elapsed_seconds":time.time()-started,"frozen_unchanged":model.frozen_digest()==frozen_hash},indent=2))


if __name__=="__main__":main()
