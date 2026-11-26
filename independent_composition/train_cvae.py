"""Train one independent long-window generator/encoder/discriminator."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import torch

from .cvae import CVAEConfig, ActionManifold, Discriminator
from .windows_data import Windows
from .train_diffusion import load_base
from .legacy import digest_file


def masked_mean(error, mask):
    return (error*mask[...,None]).sum()/(mask.sum()*error.shape[-1]).clamp_min(1)


def losses(model,b,ba,rng=None):
    condition=model.features(b['state'],b['context'],b['base_z'],ba)
    real=b['actions'];mask=b['mask'];mu,lv=model.encode(condition,real,mask)
    noise=torch.randn(mu.shape,device=mu.device,generator=rng)
    encoded=mu+torch.exp(.5*lv)*noise
    reconstruction=model.decode(condition,ba,encoded)
    mean_reconstruction=model.decode(condition,ba,mu)
    prior=torch.randn(mu.shape,device=mu.device,generator=rng)
    generated=model.decode(condition,ba,prior)
    # Noise prevents recovering codes through arbitrarily small action marks.
    noisy=generated+.02*torch.randn(generated.shape,device=generated.device,generator=rng)
    recovered,_=model.encode(condition,noisy,mask)
    enough=mask.sum(1)>=8
    code=((recovered-prior).abs().mean(1)*enough).sum()/enough.sum().clamp_min(1)
    kl=.5*(mu.square()+lv.exp()-lv-1).mean()
    moments=mu.mean(0).square().mean()+((mu.var(0)+lv.exp().mean(0)+1e-6).sqrt()-1).square().mean()
    rec=masked_mean((reconstruction-real).abs(),mask)
    action_cycle=masked_mean((mean_reconstruction-real).abs(),mask)
    total=rec+.25*action_cycle+.25*code+.01*kl+.05*moments
    metrics={'reconstruction_mae':rec,'action_cycle_mae':action_cycle,'latent_cycle_mae':code,
             'kl':kl,'moments':moments,'prior_action_delta_rms':masked_mean((generated-model.extended_base(ba)).square(),mask).sqrt()}
    return total,metrics,condition,reconstruction,generated


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--horizon',type=int,default=32);p.add_argument('--code-dim',type=int,default=4)
    p.add_argument('--hidden',type=int,default=256);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--device',default='cuda')
    p.add_argument('--start-fraction',type=float,default=0.)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh private manifold output required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    train=Windows(args.train,args.device);valid=Windows(args.validation,args.device)
    assert not train.context_fingerprints & valid.context_fingerprints
    base,source=load_base(args.env,args.device)
    alignment=8 if args.env=='cube' else 1
    b=train.sample(4096,horizon=args.horizon,alignment=alignment,start_fraction=args.start_fraction)
    c=CVAEConfig(b['state'].shape[-1],b['context'].shape[-1],b['actions'].shape[-1],
                 horizon=args.horizon,code_dim=args.code_dim,hidden=args.hidden)
    model=ActionManifold(c).to(args.device);discriminator=Discriminator(c).to(args.device)
    with torch.no_grad():
        ba=base(b['state'],b['base_z']);model.fit_normalization(model.features(b['state'],b['context'],b['base_z'],ba))
    ema=deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    d_optimizer=torch.optim.AdamW(discriminator.parameters(),lr=args.lr,weight_decay=1e-5)
    rng=torch.Generator(device=args.device).manual_seed(7251901)
    vb=valid.sample(2048,rng,horizon=args.horizon,alignment=alignment,start_fraction=args.start_fraction)
    with torch.no_grad():vba=base(vb['state'],vb['base_z'])
    config={'args':vars(args),'model_config':asdict(c),'base':source,
            'train_sha256':digest_file(args.train),'validation_sha256':digest_file(args.validation),
            'train_contexts':len(train.context_fingerprints),'validation_contexts':len(valid.context_fingerprints),
            'loss':'MAE posterior reconstruction + .25 MAE mean action-cycle + .25 noisy-action latent-cycle + .01 KL + .05 posterior moments + .05 least-squares adversarial. Cycles use current state/context and actions only.',
            'scope':'One skill per training corpus. Frozen original base. Private generator, encoder and discriminator. No shared new network, paths, simulator query, joint examples or analytic task cost in inference.',
            'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            b=train.sample(args.batch,horizon=args.horizon,alignment=alignment,start_fraction=args.start_fraction)
            with torch.no_grad():ba=base(b['state'],b['base_z'])
            total,metrics,condition,reconstruction,generated=losses(model,b,ba)
            normalized=model.normalize(condition).detach();mask=b['mask']
            discriminator.requires_grad_(True)
            real_d=discriminator(normalized,b['actions'],mask)
            fake_d=discriminator(torch.cat([normalized,normalized]),torch.cat([reconstruction.detach(),generated.detach()]),torch.cat([mask,mask]))
            d_loss=.5*((real_d-1).square().mean()+fake_d.square().mean())
            d_optimizer.zero_grad(set_to_none=True);d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(),5);d_optimizer.step()
            discriminator.requires_grad_(False)
            fake_logits=discriminator(torch.cat([normalized,normalized]),torch.cat([reconstruction,generated]),torch.cat([mask,mask]))
            adversarial=(fake_logits-1).square().mean();loss=total+.05*adversarial
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            with torch.no_grad():
                rate=.99 if step<1000 else .995
                for a,bp in zip(ema.parameters(),model.parameters()):a.lerp_(bp,1-rate)
            if step%500==0 or step==args.steps:
                with torch.no_grad():
                    vrng=torch.Generator(device=args.device).manual_seed(7296801)
                    value,vmetrics,*_=losses(ema,vb,vba,vrng)
                row={'step':step,'loss':float(loss.detach()),'discriminator_loss':float(d_loss.detach()),
                     'adversarial':float(adversarial.detach()),'validation':{k:float(v) for k,v in vmetrics.items()},
                     'validation_objective':float(value),'seconds':time.time()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                ck={'model':ema.state_dict(),'model_config':asdict(c),'config':config,'step':step,'metrics':row}
                torch.save(ck,out/'latest.pt')
                if float(value)<best:best=float(value);torch.save(ck,out/'best.pt')
    assert digest_file(source['path'])==source['sha256']
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_validation_objective':best,'base_unchanged':True})+'\n')


if __name__=='__main__':main()
