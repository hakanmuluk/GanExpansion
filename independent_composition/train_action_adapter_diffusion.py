"""Own-skill denoising loss; train only original LoRA/FiLM adapter parameters."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .action_adapter_diffusion import initialize,compact,load
from .windows_data import Windows
from .legacy import digest_file


def loss(model,b,rng=None,dropout=.15,intervention_weight=5.,background_threshold=.1):
    target=b['actions'].clamp(-1,1);n=len(target);device=target.device
    importance=target.new_ones(n)
    if model.config.score_residual:
        with torch.no_grad():base=model.actor.base(b['state'],b['base_z'])
        distance=(((target-base).square()*b['mask'][...,None]).sum((1,2))/
                  (target.shape[-1]*b['mask'].sum(-1).clamp_min(1))).sqrt()
        foreground=distance>=background_threshold
        neutral=(base+model.config.base_std*torch.randn(target.shape,device=device,generator=rng)).clamp(-1,1)
        target=torch.where(foreground[:,None,None],target,neutral)
        importance=1+(intervention_weight-1)*foreground.to(target.dtype)
        dropout=0.
    t=torch.randint(model.config.diffusion_steps,(n,),device=device,generator=rng)
    noise=torch.randn(target.shape,device=device,generator=rng)
    a=model.alpha_bar[t,None,None]
    noisy=a.sqrt()*target+(1-a).sqrt()*noise
    present=(torch.rand(n,1,device=device,generator=rng)>=dropout).to(target.dtype)
    mask=b['mask'];snr=((a/(1-a).clamp_min(1e-6)).flatten().clamp_max(5)+.05)*importance
    if model.environment=='cube':
        pred,logits=model.predict_clean(noisy,t,b['state'],b['context'],b['base_z'],present,True)
        per=(pred[:,:,:4]-target[:,:,:4]).square().mean(-1)
        per+=.25*F.binary_cross_entropy_with_logits(2*logits[:,:,4],(target[:,:,4]+1)/2,reduction='none')
    else:
        pred=model.predict_clean(noisy,t,b['state'],b['context'],b['base_z'],present)
        per=(pred-target).square().mean(-1)+.1*(pred-target).abs().mean(-1)
    return (per*mask*snr[:,None]).sum()/(mask*snr[:,None]).sum().clamp_min(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--initialize');p.add_argument('--score-residual',action='store_true')
    p.add_argument('--base-std',type=float,default=.2);p.add_argument('--intervention-weight',type=float,default=5.)
    p.add_argument('--background-threshold',type=float,default=.1)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    train=Windows(args.train);val=Windows(args.validation)
    assert not train.context_fingerprints&val.context_fingerprints
    if args.initialize:
        model,saved=load(args.initialize);source=saved['config']['source']
        assert saved['environment']==args.env and saved['skill']==args.skill
        for name,parameter in model.actor.named_parameters():
            parameter.requires_grad_(not name.startswith(('base.','original_encoder.')))
    else:model,source=initialize(args.env,args.skill)
    model.config=replace(model.config,score_residual=args.score_residual,base_std=args.base_std)
    frozen=model.actor.frozen_digest();ema=deepcopy(model).eval().requires_grad_(False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in parameters}=={id(p) for n,p in model.actor.named_parameters() if not n.startswith(('base.','original_encoder.'))}
    opt=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=1e-6)
    alignment=8 if args.env=='cube' else 1
    vrng=torch.Generator(device='cuda').manual_seed(560391)
    vb=val.sample(2048,vrng,alignment=alignment,start_fraction=.2)
    config={'args':vars(args),'source':source,'train_sha256':digest_file(args.train),'validation_sha256':digest_file(args.validation),
        'architecture':'Original frozen action base with private LoRA/FiLM adapter. Only existing context-layer input width expanded for noisy actions, fixed time features and context-presence bit. No added learned module.',
        'loss':'Min-SNR-weighted masked clean-action MSE + .1 MAE for TwoRoom; MSE continuous controls plus .25 gripper BCE for Cube. Own-context dropout .15 supports same-adapter conditional/unconditional predictions.',
        'scope':'Own skill data only. No routes as policy inputs, judges, inverse models, new shared network, joint labels or simulator query during inference.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    if args.score_residual:
        config['loss']+=' Context dropout disabled. Own-intervention examples weighted '+str(args.intervention_weight)+'x; background targets sampled around original base. Frozen base Gaussian supplies the inference reference score.'
    if args.initialize:config['initialization_sha256']=digest_file(args.initialize)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            model.train();b=train.sample(args.batch,alignment=alignment,start_fraction=.2)
            value=loss(model,b,intervention_weight=args.intervention_weight,background_threshold=args.background_threshold)
            opt.zero_grad(set_to_none=True);value.backward()
            torch.nn.utils.clip_grad_norm_(parameters,5);opt.step()
            with torch.no_grad():
                rate=.99 if step<1000 else .999
                for a,bp in zip(ema.actor.parameters(),model.actor.parameters()):
                    if bp.requires_grad:a.lerp_(bp,1-rate)
            if step%500==0 or step==args.steps:
                with torch.no_grad():
                    rng=torch.Generator(device='cuda').manual_seed(781129)
                    validation=float(loss(ema,vb,rng,dropout=0.,intervention_weight=args.intervention_weight,background_threshold=args.background_threshold))
                row={'step':step,'loss':float(value.detach()),'validation_denoising_loss':validation,'seconds':time.time()-started}
                assert model.actor.frozen_digest()==frozen and ema.actor.frozen_digest()==frozen
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                saved={**compact(ema),'base_sha256':digest_file('artifacts/final/base_policy.pt' if args.env=='tworoom' else 'cube/artifacts/final/base_policy.pt'),
                       'skill':args.skill,'step':step,'metrics':row,'config':config}
                torch.save(saved,out/'latest.pt')
                if validation<best:best=validation;torch.save(saved,out/'best.pt')
    assert digest_file(source['checkpoint'])==source['sha256']
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_validation_denoising_loss':best,'frozen_base_and_encoder':True,'new_learned_modules':0})+'\n')


if __name__=='__main__':main()
