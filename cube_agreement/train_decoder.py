"""Independent single-skill manifold learning, with no joint training inputs."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from .manifolds import oracle,segment_gap
from .neural import Decoder,sample_training


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch',type=int,default=256);p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=.0005);args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);device='cuda';rng=torch.Generator(device=device).manual_seed(args.seed);validation_rng=torch.Generator(device=device).manual_seed(987321)
    model=Decoder(args.skill).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-6);vz,vg=sample_training(args.skill,4096,device,validation_rng)
    with torch.no_grad():vt=oracle(args.skill,vz,vg)
    config={'args':vars(args),'architecture':'Independent equivariant neural single-skill path deformation with a common fixed spline latent basis; no analytic constraint projection in forward.','training':'Fresh independently sampled single-skill geometry and latents only; no joint banks, task pairs, joint expert traces or other skill constraints.','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');best=float('inf');started=time.perf_counter()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            z,g=sample_training(args.skill,args.batch,device,rng)
            with torch.no_grad():target=oracle(args.skill,z,g)
            pred=model(z,g);error=(pred-target)/.1
            # A tail-sensitive regression loss discourages thin unsafe underfits.
            rec=error.square().mean()+.2*error.square().flatten(1).amax(1).mean()
            gap=segment_gap(pred,g,args.skill)
            penalty=(torch.relu(.003-gap)/.1).square().mean() if args.skill=='keepout' else (torch.relu(gap+.002)/.1).square().mean()
            loss=rec+2*penalty
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if step%500==0 or step==args.steps:
                with torch.no_grad():
                    pv=model(vz,vg);ve=(pv-vt).square().mean().sqrt();valid_gap=segment_gap(pv,vg,args.skill);valid=(valid_gap>=-1e-7) if args.skill=='keepout' else (valid_gap<=0)
                    metric={'step':step,'loss':float(loss),'validation_path_RMSE_m':float(ve),'validation_skill_geometry_success':float(valid.float().mean()),'validation_worst_gap_m':float(valid_gap.min() if args.skill=='keepout' else valid_gap.max()),'seconds':time.perf_counter()-started}
                log.write(json.dumps(metric)+'\n');print(metric,flush=True)
                score=(1-float(valid.float().mean()))+float(ve)
                ck={'skill':args.skill,'hidden':model.hidden,'state':model.state_dict(),'step':step,'metric':metric,'config':config,'optimizer':optimizer.state_dict()}
                torch.save(ck,out/'latest.pt')
                if score<best:best=score;torch.save(ck,out/'best.pt')
    (out/'DONE.json').write_text(json.dumps({'complete':True,'steps':args.steps,'best_score':best}))


if __name__=='__main__':main()
