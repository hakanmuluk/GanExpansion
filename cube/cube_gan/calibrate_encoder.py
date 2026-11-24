"""Calibrate E on real learner episodes while keeping both G weights fixed."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .models import reconstruction,kl_loss
from .evaluate import load_models
from .dataset import EpisodeDataset
from .style import encode_style


def digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True);p.add_argument("--run",required=True)
    p.add_argument("--val-rollouts",required=True)
    p.add_argument("--data",default="artifacts/data")
    p.add_argument("--physical",default="artifacts/feedback/round1_seed42/learner_rollouts.npz")
    p.add_argument("--steps",type=int,default=2000);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--device",default="cuda")
    args=p.parse_args();run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    g,e,c=load_models(args.checkpoint,args.device)
    g.requires_grad_(False)
    ghash=digest(g.state_dict());rawhash=digest(c["generator"])
    original_step=c["step"]
    opt=torch.optim.Adam(list(g.parameters())+list(e.parameters()),lr=args.lr,betas=(.5,.99))
    opt.load_state_dict(c["optimizer_ge"])
    for group in opt.param_groups:group["lr"]=args.lr
    with np.load(args.physical) as f:physical={k:torch.as_tensor(f[k],device=args.device) for k in f.files}
    with np.load(args.val_rollouts) as f:val={k:torch.as_tensor(f[k],device=args.device) for k in ["states","actions","lengths","z"]}
    val_summary=json.loads(Path(args.val_rollouts).with_name("summary.json").read_text())
    assert Path(val_summary["args"]["checkpoint"])==Path(args.checkpoint)
    assert val_summary["args"]["split"]=="val"
    ds=EpisodeDataset(Path(args.data)/"train.npz",args.device)
    config={"args":vars(args),"frozen_generator_sha256":ghash,"generator_training_step":original_step,
            "objective":"2*physical style L1 + expert style L1 + frozen-G reconstruction + .01 KL; E only",
            "validation":"Encoder RMSE on frozen-G actual validation rollouts; G predictions are unchanged"}
    (run/"config.json").write_text(json.dumps(config,indent=2))
    best=float("inf");started=time.time();running=0.
    def evaluate():
        e.eval();mus=[]
        with torch.inference_mode():
            for t in range(0,len(val["states"]),64):
                mu,_=e(val["states"][t:t+64],val["actions"][t:t+64],val["lengths"][t:t+64]);mus.append(mu)
        return ((torch.cat(mus)-val["z"])**2).mean(0).sqrt().cpu().tolist()
    def save(path,step,rmse):
        assert digest(g.state_dict())==ghash
        assert digest(c["generator"])==rawhash
        c["encoder"]=e.state_dict();c["optimizer_ge"]=opt.state_dict()
        c["encoder_calibration"]={**config,"step":step,"val_rmse":rmse,"frozen_generator_verified":True}
        torch.save(c,path)
    initial=evaluate()
    best=float(np.mean(initial))
    save(run/"best.pt",0,initial)
    print("INITIAL",json.dumps(initial),flush=True)
    with (run/"train.jsonl").open("w",buffering=1) as log:
        for step in range(1,args.steps+1):
            e.train();opt.zero_grad(set_to_none=True)
            idx=torch.randint(len(physical["states"]),(64,),device=args.device)
            pb={k:v[idx] for k,v in physical.items()}
            mu,lv=e(pb["states"],pb["actions"],pb["lengths"])
            actual=F.l1_loss(mu,pb["z"])
            b=ds.sample(64,4)
            expert_mu,expert_lv=e(b["episode_states"],b["episode_actions"],b["lengths"])
            target=encode_style(b["mode"],b["amplitude"],b["speed"])
            supervised=F.l1_loss(expert_mu,target)
            pred=g(b["states"],expert_mu[:,None].expand(-1,4,-1))
            rec=reconstruction(pred,b["actions"],b["mask"])
            loss=2*actual+supervised+rec+.01*(kl_loss(mu,lv)+kl_loss(expert_mu,expert_lv))
            loss.backward();torch.nn.utils.clip_grad_norm_(e.parameters(),5);opt.step()
            running+=float(loss.detach())
            if step%250==0:
                rmse=evaluate();score=float(np.mean(rmse))
                row={"step":step,"loss":running/250,"closed_loop_val_rmse":rmse,"elapsed_seconds":time.time()-started};running=0
                log.write(json.dumps(row)+"\n");print(json.dumps(row),flush=True)
                save(run/"latest.pt",step,rmse)
                if score<best:best=score;save(run/"best.pt",step,rmse)
    (run/"completed.json").write_text(json.dumps({"initial_rmse":initial,"best_rmse_mean":best,"generator_sha256":ghash,"frozen_generator_verified":digest(g.state_dict())==ghash,"steps":args.steps,"elapsed_seconds":time.time()-started},indent=2))


if __name__=="__main__":main()
