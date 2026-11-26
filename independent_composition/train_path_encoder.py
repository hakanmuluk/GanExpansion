"""Train one private full-route inverse using only own-valid trajectories.

Own geometric predicates select positive training examples only. They are not
part of the encoder or the deployed latent-composition objective.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from .path_encoder import PathEncoder
from .path_policy import load_route
from .tworoom_paths import proposal
from .train_tworoom_routes import Data,margins
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--route',required=True);p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--seed',type=int,default=42);args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh encoder training directory required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    route,meta=load_route(args.route);assert meta['skill']==args.skill
    train,valid=Data(args.train),Data(args.validation)
    assert not set(map(bytes,train.context.cpu().numpy())).intersection(map(bytes,valid.context.cpu().numpy()))
    model=PathEncoder().cuda();model.context_mean.copy_(route.context_mean);model.context_std.copy_(route.context_std)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-6)
    rng=torch.Generator(device='cuda').manual_seed(args.seed+1901)
    vrng=torch.Generator(device='cuda').manual_seed(723001)
    vz,vc,vt=valid.sample(min(2048,valid.size),vrng)
    with torch.no_grad():
        vg=route(vz,vc,proposal(vz,vc));_,vok,_=margins(vg,vc,args.skill)
    config={'args':vars(args),'route_sha256':digest_file(args.route),
        'train_sha256':digest_file(args.train),'validation_sha256':digest_file(args.validation),
        'scope':'Own teacher paths and own-valid generated paths only. Frozen private route decoder, independent encoder weights, no joint scenes or labels. Own geometry is training-only filtering.',
        'loss':'Generated own-valid path -> code SmoothL1 + .25 teacher path -> known code SmoothL1 +2 teacher path -> code -> frozen route MSE.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            z,c,target=train.sample(512,rng)
            prior=torch.randn(z.shape,device='cuda',generator=rng).clamp(-3,3)
            prior[:len(z)//2]=6*torch.rand((len(z)//2,10),device='cuda',generator=rng)-3
            with torch.no_grad():
                generated=route(prior,c,proposal(prior,c));_,keep,_=margins(generated,c,args.skill)
            assert int(keep.sum())>8
            recovered=model(generated[keep],c[keep]);encoded=model(target,c)
            reconstructed=route(encoded,c,proposal(encoded,c))
            loss=torch.nn.functional.smooth_l1_loss(recovered,prior[keep])
            loss=loss+.25*torch.nn.functional.smooth_l1_loss(encoded,z)+2*(reconstructed-target).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if step%1000==0 or step==args.steps:
                model.eval()
                with torch.no_grad():
                    recovery=model(vg,vc);error=(recovery-vz).square().mean(1)
                    teacher=model(vt,vc);recon=route(teacher,vc,proposal(teacher,vc))
                    value=error[vok].mean().sqrt()+(recon-vt).square().mean().sqrt()
                    row={'step':step,'loss':float(loss.detach()),'validation_generated_code_rmse':float(error.mean().sqrt()),
                         'validation_valid_code_rmse':float(error[vok].mean().sqrt()),
                         'validation_invalid_code_rmse':float(error[~vok].mean().sqrt()) if bool((~vok).any()) else None,
                         'validation_teacher_code_rmse':float((teacher-vz).square().mean().sqrt()),
                         'validation_teacher_reconstruction_rmse_px':float(112*(recon-vt).square().mean().sqrt()),
                         'valid_generated_fraction':float(vok.float().mean()),'seconds':time.time()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                ck={'state':model.state_dict(),'model_config':{'context_dim':25,'code_dim':10,'hidden':256},
                    'skill':args.skill,'step':step,'metrics':row,'config':config}
                torch.save(ck,out/'latest.pt')
                if float(value)<best:best=float(value);torch.save(ck,out/'best.pt')
                model.train()
    assert all(p.grad is None and not p.requires_grad for p in route.parameters())
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_validation_loss':best,'route_frozen':True})+'\n')


if __name__=='__main__':main()
