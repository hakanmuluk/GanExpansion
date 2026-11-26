"""Train an independent, current-observation/action-window inverse model.

Validation includes real demonstration windows and counterfactual actor windows
with independent random codes at the SAME states. This reveals state shortcuts
and unavoidable ambiguity where the actor does not use its modifier locally.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
from .legacy import load_skill, digest_file
from .window import WindowConfig, WindowEncoder, gaussian_nll
from .windows_data import Windows


@torch.no_grad()
def diagnostics(encoder, batch):
    s,c,bz,a,m,u=[batch[k] for k in ["state","context","base_z","actions","mask","code"]]
    mu,lv=encoder(s,c,bz,a,m)
    # Same observations and masks; only actions are permuted or removed.
    perm=torch.roll(torch.arange(len(a),device=a.device),1)
    shuffled,_=encoder(s,c,bz,a[perm],m)
    zero,_=encoder(s,c,bz,torch.zeros_like(a),m)
    return {"rmse":float((mu-u).square().mean().sqrt()),
            "mae":float((mu-u).abs().mean()),
            "sign_accuracy":float((mu.sign()==u.sign()).float().mean()),
            "nll":float(gaussian_nll(mu,lv,u)),
            "posterior_std_mean":float((lv*.5).exp().mean()),
            "shuffled_actions_rmse":float((shuffled-u).square().mean().sqrt()),
            "zero_actions_rmse":float((zero-u).square().mean().sqrt()),
            "constant_zero_rmse":float(u.square().mean().sqrt())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--env",choices=["tworoom","cube"],required=True)
    p.add_argument("--skill",choices=["keepout","waypoint"],required=True)
    p.add_argument("--train",required=True);p.add_argument("--validation")
    p.add_argument("--out",required=True);p.add_argument("--steps",type=int,default=4000)
    p.add_argument("--batch",type=int,default=512);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--model-seed",type=int,default=42);p.add_argument("--episodes",type=int,default=2048)
    p.add_argument("--lr",type=float,default=.0003);p.add_argument("--device",default="cuda")
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/"config.json").exists():raise FileExistsError("Use a fresh output directory")
    train=Windows(args.train,args.device,args.episodes,split=None if args.validation else "train")
    valid=Windows(args.validation or args.train,args.device,
                  None if args.validation else args.episodes,
                  split=None if args.validation else "validation")
    assert not train.context_fingerprints & valid.context_fingerprints, "Overlapping physical context split"
    actor,_,source=load_skill(args.env,args.skill,args.device,args.model_seed)
    sample=train.sample(4096)
    c=WindowConfig(sample["state"].shape[-1],sample["context"].shape[-1],sample["actions"].shape[-1])
    encoder=WindowEncoder(c).to(args.device)
    encoder.fit_normalization(torch.cat([sample["state"],sample["context"],sample["base_z"]],-1))
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=args.lr,weight_decay=1e-5)
    rng=torch.Generator(device=args.device).manual_seed(893014)
    validation=valid.sample(4096,rng)
    synthetic={k:v.clone() for k,v in validation.items()}
    synthetic["code"]=torch.randn(synthetic["code"].shape,device=args.device,generator=rng)
    with torch.no_grad():
        synthetic["actions"]=actor(synthetic["state"],synthetic["base_z"],synthetic["code"],synthetic["context"])
        synthetic["mask"]=torch.ones_like(synthetic["mask"])
    config={"args":vars(args),"actor":source,"model_config":asdict(c),
            "train_sha256":digest_file(args.train),"validation_sha256":digest_file(args.validation or args.train),
            "train_contexts":len(train.context_ids),"validation_contexts":len(valid.context_ids),
            "train_episodes":train.size,"validation_episodes":valid.size,
            "loss":".5 Gaussian NLL(real expert windows) + .5 Gaussian NLL(counterfactual frozen-actor windows)",
            "source_sha256":{str(f.name):digest_file(f) for f in Path(__file__).parent.glob('*.py')},
            "scope":"Own skill only. No joint scenes, future states, task costs, new shared network, or trainable generator."}
    (out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    started=time.time();best=float("inf")
    with (out/"train.jsonl").open("w",buffering=1) as log:
        for step in range(1,args.steps+1):
            encoder.train();b=train.sample(args.batch)
            with torch.no_grad():
                code=torch.randn_like(b["code"])
                actions=actor(b["state"],b["base_z"],code,b["context"])
            mu,lv=encoder(b["state"],b["context"],b["base_z"],b["actions"],b["mask"])
            sm,sl=encoder(b["state"],b["context"],b["base_z"],actions)
            loss=.5*gaussian_nll(mu,lv,b["code"])+.5*gaussian_nll(sm,sl,code)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),5);optimizer.step()
            if step%250==0 or step==args.steps:
                encoder.eval();real=diagnostics(encoder,validation);fake=diagnostics(encoder,synthetic)
                score=.5*(real["nll"]+fake["nll"])
                row={"step":step,"loss":float(loss),"validation_real":real,"validation_actor":fake,"seconds":time.time()-started}
                log.write(json.dumps(row)+"\n");print(json.dumps(row),flush=True)
                ck={"encoder":encoder.state_dict(),"model_config":asdict(c),"config":config,"step":step,"metrics":row}
                torch.save(ck,out/"latest.pt")
                if score<best:best=score;torch.save(ck,out/"best.pt")
    assert digest_file(source["checkpoint"])==source["sha256"]
    assert all(not x.requires_grad and x.grad is None for x in actor.parameters())
    (out/"DONE.json").write_text(json.dumps({"steps":args.steps,"best_validation_nll":best,"actor_unchanged":True})+"\n")


if __name__=="__main__":main()
