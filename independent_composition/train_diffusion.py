"""Independently train an action-distribution adapter from own-skill data only."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from .diffusion import Denoiser,DiffusionConfig
from .windows_data import Windows
from .legacy import digest_file


def load_base(env,device):
    if env=='tworoom':
        from tworoom_gan.evaluate import load_checkpoint
        path='artifacts/final/base_policy.pt';base,_,_=load_checkpoint(path,device)
    else:
        from cube.cube_gan.evaluate import load_models
        path='cube/artifacts/final/base_policy.pt';base,_,_=load_models(path,device)
    return base.eval().requires_grad_(False),{'path':path,'sha256':digest_file(path)}


def denoising_loss(model,b,base_actions,generator=None,dropout=.15):
    shape=b['actions'].shape;device=b['actions'].device
    noise=torch.randn(shape,device=device,generator=generator)
    t=torch.randint(model.config.diffusion_steps,(shape[0],),device=device,generator=generator)
    a=model.alpha_bar[t,None,None]
    clean=b['actions'].clamp(-1,1)
    weight=b['mask']
    if model.config.score_residual:
        difference=(((clean-base_actions).square()*weight[...,None]).sum((1,2)) /
                    (shape[-1]*weight.sum(-1).clamp_min(1))).sqrt()
        foreground=difference>=model.config.background_threshold
        neutral=base_actions+model.config.base_std*torch.randn(shape,device=device,generator=generator)
        clean=torch.where(foreground[:,None,None],clean,neutral)
        weight=weight*(1+2*foreground.to(weight.dtype))[:,None]
        dropout=0.
    noisy=a.sqrt()*clean+(1-a).sqrt()*noise
    present=(torch.rand((shape[0],1),device=device,generator=generator)>=dropout).to(clean.dtype)
    predicted=model(noisy,t,b['state'],b['context'],b['base_z'],base_actions,present)
    error=(predicted-noise).square()*weight[...,None]
    return error.sum()/(shape[-1]*weight.sum().clamp_min(1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--device',default='cuda');p.add_argument('--hidden',type=int,default=384)
    p.add_argument('--score-residual',action='store_true')
    p.add_argument('--base-std',type=float,default=.2)
    p.add_argument('--background-threshold',type=float,default=.1)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh training directory required')
    out.mkdir(parents=True,exist_ok=True)
    train=Windows(args.train,args.device);valid=Windows(args.validation,args.device)
    assert not train.context_fingerprints&valid.context_fingerprints,'Overlapping context identities'
    base,source=load_base(args.env,args.device)
    alignment=8 if args.env=='cube' else 1
    b=train.sample(4096,alignment=alignment)
    c=DiffusionConfig(b['state'].shape[-1],b['context'].shape[-1],b['actions'].shape[-1],hidden=args.hidden,
                      score_residual=args.score_residual,base_std=args.base_std,background_threshold=args.background_threshold)
    model=Denoiser(c).to(args.device)
    with torch.no_grad():
        ba=base(b['state'],b['base_z'])
        model.fit_normalization(model.condition_vector(b['state'],b['context'],b['base_z'],ba))
    ema=deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-6)
    validation_rng=torch.Generator(device=args.device).manual_seed(4219031)
    vb=valid.sample(2048,validation_rng,alignment=alignment)
    with torch.no_grad():vbase=base(vb['state'],vb['base_z'])
    config={'args':vars(args),'model_config':asdict(c),'base':source,
            'train_sha256':digest_file(args.train),'validation_sha256':digest_file(args.validation),
            'train_episodes':train.size,'train_contexts':len(train.context_fingerprints),
            'validation_episodes':valid.size,'validation_contexts':len(valid.context_fingerprints),
            'loss':('Masked denoising noise MSE; fixed-base Gaussian score + independent learned residual; expert-base action discrepancy marks intervention windows, weighted 3x; background windows are base Gaussian samples.'
                    if args.score_residual else 'Masked denoising noise MSE; own-context dropout .15; independent parameters; original base frozen.'),
            'planning_alignment':alignment,
            'source_sha256':{p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')},
            'scope':'Own skill data only; no shared learned denoiser, executor, critic, joint examples, analytic inference costs, or simulator query in inference.'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            b=train.sample(args.batch,alignment=alignment)
            with torch.no_grad():ba=base(b['state'],b['base_z'])
            loss=denoising_loss(model,b,ba)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            with torch.no_grad():
                rate=.99 if step<1000 else .999
                for a,bp in zip(ema.parameters(),model.parameters()):a.lerp_(bp,1-rate)
            if step%500==0 or step==args.steps:
                with torch.no_grad():
                    vrng=torch.Generator(device=args.device).manual_seed(680123)
                    value=float(denoising_loss(ema,vb,vbase,vrng,dropout=0.))
                row={'step':step,'loss':float(loss.detach()),'validation_denoising_loss':value,'seconds':time.time()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                ck={'ema':ema.state_dict(),'model':model.state_dict(),'model_config':asdict(c),'config':config,'step':step,'metrics':row}
                torch.save(ck,out/'latest.pt')
                if value<best:best=value;torch.save(ck,out/'best.pt')
    assert digest_file(source['path'])==source['sha256']
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_validation_denoising_loss':best,'original_base_unchanged':True})+'\n')


if __name__=='__main__':main()
