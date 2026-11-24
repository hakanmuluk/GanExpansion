"""Held-out closed-loop metrics; behavior labels use executed geometry."""
import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from tworoom_gan.env import TASK,observation
from tworoom_gan.evaluate import rollout as old_rollout,wilson
from tworoom_gan.style import decode_style
from .geometry import frame,segment_distance
from .models import GatedGenerator,AdapterConfig,SkillEncoder
from .rollout import rollout


class OffActor(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,state,z):return self.model(state,z,gate=0)


def load(path,device="cuda",base="artifacts/final/base_policy.pt"):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    if hashlib.sha256(Path(base).read_bytes()).hexdigest()!=ckpt["base_sha256"]:
        raise ValueError("Adapter base checkpoint fingerprint does not match")
    model=GatedGenerator(base,AdapterConfig(**ckpt["adapter_config"]),device)
    if set(ckpt["addon"])!=set(model.addon_state()):raise ValueError("Incomplete adapter state")
    model.load_state_dict(ckpt["addon"],strict=False)
    if "finetuned_base" in ckpt:model.base.load_state_dict(ckpt["finetuned_base"])
    encoder=SkillEncoder(model.config.new_z_dim,context_dim=model.config.context_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    return model.eval(),encoder.eval(),ckpt


def contexts(path,device="cuda",count=0):
    with np.load(path) as f:d={k:torch.tensor(f[k][:count or None],device=device) for k in f.files}
    return d


def behavior(result,skill):
    p=result["positions"];n,t=p.shape[:2];goal=result["goal"]
    direction=torch.where(goal[:,0]>=112,1.,-1.)
    q=p.clone();q[...,0]=112+direction[:,None]*(p[...,0]-112)
    c=frame(result["spec"][:,:2],goal)
    valid=torch.arange(t-1,device=p.device)[None]<result["lengths"][:,None]
    cross=((p[:,:-1,0]>112)!=(p[:,1:,0]>112))&valid
    first=cross.long().argmax(1)
    row=torch.arange(n,device=p.device)
    door=(p[row,first+1,1,None]-p.new_tensor(TASK.door_y)).abs().argmin(-1)
    door=torch.where(cross.any(1),door,-1)
    if skill=="keepout":
        mode,_,offset=decode_style(result["base_z"])
        endpoint=torch.stack([torch.full_like(offset,99.),p.new_tensor(TASK.door_y)[mode]+offset],-1)
        axis=endpoint-q[:,0];axis=axis/axis.norm(dim=-1,keepdim=True)
        perp=torch.stack([-axis[:,1],axis[:,0]],-1)
        relative=q[:,:-1]-c[:,None]
        along=(relative*axis[:,None]).sum(-1);signed=(relative*perp[:,None]).sum(-1)
        near=(along.abs()<8)&(q[:,:-1,0]<100)&valid
        idx=signed.abs().masked_fill(~near,-1).argmax(1)
        offset=signed[row,idx]
        strategy=torch.where(near.any(1)&(offset.abs()>5),offset.sign(),0).long()
        distances=segment_distance(p[:,:-1],p[:,1:],result["spec"][:,None,:2])
        amount=distances.masked_fill(~valid,float("inf")).amin(1)-result["spec"][:,2]-TASK.radius
        # Repeated crossings of the central slice indicate strategy switching.
        has_pos=((signed>5)&near).any(1);has_neg=((signed < -5)&near).any(1)
        switches=has_pos&has_neg
    else:
        dist=segment_distance(p[:,:-1],p[:,1:],result["spec"][:,None,:2])
        touch=(dist<=result["spec"][:,None,2])&valid
        first_visit=touch.long().argmax(1)
        # Position immediately before the first swept contact gives entry side.
        entry=q[row,first_visit,0]-c[:,0]
        strategy=torch.where(touch.any(1)&(entry.abs()>result["spec"][:,2]*.2),entry.sign(),0).long()
        before=(torch.arange(t-1,device=p.device)[None]<=first_visit[:,None])&valid&touch.any(1)[:,None]
        band=before&(q[:,:-1,0]>=132)&((q[:,:-1,1]-c[:,None,1]).abs()<2)
        signed=q[:,:-1,0]-c[:,None,0]
        amount=signed.abs().masked_fill(~band,0).amax(1)
        # Before contact, a route that enters both approach half-planes changed side.
        local=before&(q[:,:-1,0]>=132)&((q[:,:-1,1]-c[:,None,1]).abs()<result["spec"][:,None,2]*.5)&(dist<result["spec"][:,None,2]+8)
        switches=((signed>5)&local).any(1)&((signed < -5)&local).any(1)
    return door,strategy,amount,switches


def corr(x,y):
    x=x.float();y=y.float();x=x-x.mean();y=y-y.mean()
    return float((x*y).sum()/(x.square().sum()*y.square().sum()).sqrt().clamp_min(1e-8)) if len(x)>1 else None


@torch.no_grad()
def summarize(model,encoder,result,skill,pairs,codes,intervention=True):
    door,strategy,amount,switches=behavior(result,skill)
    ok=result["success"];expected_door=(result["base_z"][:,0]>=0).long()
    styled=ok&(door==expected_door);signed=torch.sign(result["new_z"][:,0]).long()
    a=(styled&(strategy==1)).reshape(pairs,codes).any(1)
    b=(styled&(strategy==-1)).reshape(pairs,codes).any(1)
    m={"rollouts":len(ok),"contexts":pairs,"codes_per_context":codes,"success":float(ok.float().mean()),
       "success_wilson95":wilson(int(ok.sum()),len(ok)),"goal_success":float(result["goal_success"].float().mean()),
       "style_preserving_success":float(styled.float().mean()),"base_door_agreement":float((door==expected_door).float().mean()),
       "both_successful_strategies_per_context":float((a&b).float().mean()),
       "successful_positive_strategy":float((styled&(strategy==1)).float().mean()),
       "successful_negative_strategy":float((styled&(strategy==-1)).float().mean()),
       "strategy_matches_code":float((strategy[styled]==signed[styled]).float().mean()) if styled.any() else None,
       "strategy_switch_fraction":float(switches.float().mean()),
       "native_collision_fraction":float((result["collisions"]>0).float().mean()),"mean_steps":float(result["lengths"].float().mean()),
       "fixed_code_max_change":float((result["applied_new_z"]-result["new_z"][:,None]).abs().max()),
       "constraint_failure":float((result["violated"] if skill=="keepout" else ~result["visited"]).float().mean()),
       "continuous_amount_mean":float(amount.mean()),"continuous_amount_std":float(amount.std()) if len(amount)>1 else 0.,
       "code_magnitude_amount_correlation_successful":corr(result["new_z"][styled,0].abs(),amount[styled])}
    # Body/wall overlap is stricter than the upstream permissive doorway margin.
    p=result["positions"][:,:-1];valid=torch.arange(p.shape[1],device=p.device)[None]<result["lengths"][:,None]
    dx=((p[...,0]-112).abs()-5).clamp_min(0);overlap=torch.zeros_like(valid)
    for lo,hi in [(14,35),(77,147),(189,210)]:
        dy=torch.maximum(lo-p[...,1],p[...,1]-hi).clamp_min(0)
        overlap|=dx.square()+dy.square()<49-1e-4
    m["strict_body_wall_overlap_fraction"]=float((overlap&valid).any(1).float().mean())
    if codes>=4:
        grid=amount.reshape(pairs,codes);good=styled.reshape(pairs,codes)
        deltas=[]
        for high,low in [(0,codes//2-1),(codes-1,codes//2)]:
            paired=good[:,high]&good[:,low]
            deltas.append((grid[:,high]-grid[:,low])[paired])
        delta=torch.cat(deltas)
        m["within_context_large_minus_small_code_amount_mean"]=float(delta.mean()) if len(delta) else None
        m["within_context_amount_increases_fraction"]=float((delta>0).float().mean()) if len(delta) else None
    if intervention:
        rng=np.random.default_rng(12031);index=rng.integers(0,pairs,(2000,pairs))
        rates=ok.float().reshape(pairs,codes).mean(1).cpu().numpy()
        m["success_context_bootstrap95"]=np.quantile(rates[index].mean(1),[.025,.975]).tolist()
        both=(a&b).float().cpu().numpy()
        m["both_strategies_context_bootstrap95"]=np.quantile(both[index].mean(1),[.025,.975]).tolist()
    if skill=="waypoint":
        touch=(segment_distance(result["positions"][:,:-1],result["positions"][:,1:],result["spec"][:,None,:2])<=result["spec"][:,None,2])&valid
        at_goal=((result["positions"][:,1:]-result["goal"][:,None]).norm(dim=-1)<16)&valid
        first_visit=touch.long().argmax(1);first_goal=at_goal.long().argmax(1)
        early=at_goal.any(1)&((~touch.any(1))|(first_goal<first_visit))
        m["goal_region_entered_before_waypoint_fraction"]=float(early.float().mean())
        m["strict_first_goal_entry_skill_success"]=float((ok&~early).float().mean())
    if encoder is not None and not model.config.no_new_latent:
        mu,_=encoder(result["states"],result["actions"],result["contexts"],result["base_z"],result["lengths"])
        m["closed_loop_latent_rmse"]=float((mu-result["new_z"]).square().mean().sqrt())
        m["closed_loop_latent_correlation"]=corr(mu[:,0],result["new_z"][:,0])
        m["closed_loop_latent_sign_accuracy"]=float(((mu[:,0]>=0)==(result["new_z"][:,0]>=0)).float().mean())
    if intervention:
        # Compare BOTH predictions at exactly the same on-policy replanning states.
        states=result["states"][:,::8];ctx=result["contexts"][:,::8]
        z=result["base_z"][:,None].expand(-1,states.shape[1],-1)
        zk=result["applied_new_z"][:,::8]
        adapted=model(states,z,zk,ctx);base=model.base(states,z)
        future=torch.arange(states.shape[1],device=p.device)[None,:,None]*8+torch.arange(16,device=p.device)[None,None,:]
        future_valid=future<result["lengths"][:,None,None]
        diff=((adapted-base).norm(dim=-1)*future_valid).sum(-1)/future_valid.sum(-1).clamp_min(1)
        active=torch.arange(states.shape[1],device=p.device)[None]*8<result["lengths"][:,None]
        if skill=="keepout":near=(ctx[...,7]<.65)&(frame((states[...,:2]+1)*112,result["goal"][:,None])[...,0]<125)
        else:near=(frame((states[...,:2]+1)*112,result["goal"][:,None])[...,0]>=125)&(ctx[...,8]<.5)
        for label,mask in [("all",active),("near",active&near),("far",active&~near)]:
            m[f"matched_state_action_l2_{label}"]=float(diff[mask].mean()) if mask.any() else None
        speed=result["actions"].norm(dim=-1)
        _,expected_speed,_=decode_style(result["base_z"])
        m["executed_speed_base_style_correlation"]=corr((speed*valid).sum(1)/result["lengths"],expected_speed)
        full_ctx=result["contexts"]
        canonical_x=frame(p,result["goal"][:,None])[...,0]
        near_full=((full_ctx[...,7]<.65)&(canonical_x<125)) if skill=="keepout" else ((canonical_x>=125)&(full_ctx[...,8]<.5))
        far=valid&~near_full
        measured=(speed*far).sum(1)/far.sum(1).clamp_min(1)
        m["outside_intervention_speed_style_correlation"]=corr(measured[far.any(1)],expected_speed[far.any(1)])
        m["outside_intervention_speed_absolute_error"]=float((measured[far.any(1)]-expected_speed[far.any(1)]).abs().mean())
    result.update(strategy=strategy,amount=amount,door=door,style_success=styled)
    return m


@torch.no_grad()
def evaluate(model,encoder,d,skill,codes=8,seed=701,resample=False,gate=1,intervention=False,gaussian=False):
    n=len(d["start"]);device=d["start"].device
    if gaussian:
        rng=torch.Generator(device=device).manual_seed(seed)
        znew=torch.randn((n,codes,1),device=device,generator=rng).reshape(-1,1)
    else:
        # Symmetric Gaussian quantiles are shared across contexts for consistency.
        u=(torch.arange(codes,device=device)+.5)/codes
        znew=(2**.5*torch.erfinv(2*u-1)).repeat(n)[:,None]
    repeated={k:v.repeat_interleave(codes,0) for k,v in d.items()}
    model.eval()
    result=rollout(model,**repeated,new_z=znew,skill=skill,resample=resample,gate=gate,seed=seed)
    metric=summarize(model,encoder,result,skill,n,codes,intervention)
    return metric,result


@torch.no_grad()
def preservation(model,base_path,d):
    from tworoom_gan.evaluate import load_checkpoint
    original,enc,_=load_checkpoint(base_path,d["start"].device)
    state=observation(d["start"],d["goal"]);z=d["base_z"]
    expected=original(state,z);disabled=model(state,z,gate=0)
    a=old_rollout(original,d["start"],d["goal"],z)
    b=old_rollout(OffActor(model),d["start"],d["goal"],z)
    return {"prediction_max_abs_difference":float((expected-disabled).abs().max()),
            "prediction_bitwise_equal":bool(torch.equal(expected,disabled)),
            "trajectory_max_abs_difference":float((a["positions"]-b["positions"]).abs().max()),
            "trajectory_bitwise_equal":bool(torch.equal(a["positions"],b["positions"])),
            "original_goal_success":float(a["success"].float().mean()),"disabled_goal_success":float(b["success"].float().mean())}


@torch.no_grad()
def offline(model,encoder,path):
    from .data import EpisodeDataset
    from .train import masked_l1,expand
    torch.manual_seed(61017)
    data=EpisodeDataset(path,next(model.parameters()).device)
    b=data.sample(128,4)
    mu,lv=encoder(b["episode_states"],b["episode_actions"],b["episode_contexts"],b["base_z"],b["lengths"])
    z=expand(b["base_z"],4)
    pred=model(b["states"],z,expand(mu,4),b["contexts"])
    metric={"reconstruction_action_mae":float(masked_l1(pred,b["actions"],b["mask"])),
            "known_code_reconstruction_action_mae":float(masked_l1(model(b["states"],z,expand(b["new_z"],4),b["contexts"]),b["actions"],b["mask"]))}
    if model.config.no_new_latent:return metric
    metric["demo_encoder_latent_rmse"]=float((mu-b["new_z"]).square().mean().sqrt())
    metric["demo_encoder_latent_correlation"]=corr(mu[:,0],b["new_z"][:,0])
    s=b["episode_states"][:16];c=b["episode_contexts"][:16];znew=torch.randn_like(b["new_z"][:16])
    actions=model(s,expand(b["base_z"][:16],240),expand(znew,240),c)[:,:,0]
    recovered,_=encoder(s,actions,c,b["base_z"][:16],b["lengths"][:16])
    metric["teacher_state_cycle_latent_rmse"]=float((recovered-znew).square().mean().sqrt())
    return metric


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--skill",choices=["keepout","waypoint"],required=True)
    p.add_argument("--split",default="test");p.add_argument("--codes",type=int,default=8);p.add_argument("--out",required=True)
    p.add_argument("--data",default="addon_artifacts/data");p.add_argument("--device",default="cuda");p.add_argument("--metrics-only",action="store_true");args=p.parse_args()
    torch.set_num_threads(2);model,encoder,ck=load(args.checkpoint,args.device)
    d=contexts(Path(args.data)/args.skill/f"{args.split}_contexts.npz",args.device)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report={"checkpoint":args.checkpoint,"skill":args.skill,"step":ck["step"],"args":ck["args"],"adapter_config":asdict(model.config)}
    report["quantiles"],r=evaluate(model,encoder,d,args.skill,args.codes,intervention=True)
    if not args.metrics_only:np.savez_compressed(out/"trajectories.npz",**{k:v.cpu().numpy() for k,v in r.items()})
    report["gaussian"],_=evaluate(model,encoder,d,args.skill,args.codes,gaussian=True,intervention=True)
    report["resampled"],_=evaluate(model,encoder,d,args.skill,args.codes,resample=True)
    report["base"],_=evaluate(model,None,d,args.skill,args.codes,gate=0)
    report["preservation"]=preservation(model,"artifacts/final/base_policy.pt",d)
    report["offline"]=offline(model,encoder,Path(args.data)/args.skill/f"{args.split}.npz")
    (out/"metrics.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
