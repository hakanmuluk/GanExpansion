"""Train one classical adapter to edit actions for only its own skill."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from .refinement_adapters import initialize,compact
from .refinement_labels import training_pair
from .windows_data import Windows
from .legacy import digest_file


def loss(model,b):
    dim=3 if model.environment=='cube' else 2
    delta=model.delta(b['state'],b['base_z'],b['code'],b['context'],b['candidate'])
    d=delta.tanh();pred=((b['candidate']+d)/(1+b['candidate']*d).clamp_min(1e-6)).clamp(-1,1)
    logits_target=torch.atanh(b['target'].clamp(-.9999,.9999))-torch.atanh(b['candidate'].clamp(-.9999,.9999))
    error=(pred[:,:dim]-b['target'][:,:dim]).square().mean(-1)+.05*(delta[:,:dim]-logits_target[:,:dim]).square().mean(-1)
    return (error*b['importance']).sum()/b['importance'].sum().clamp_min(1),pred


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=15000);p.add_argument('--batch',type=int,default=512);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    train=Windows(args.train);val=Windows(args.validation);assert not train.context_fingerprints&val.context_fingerprints
    model,source=initialize(args.env,args.skill);frozen=model.actor.frozen_digest()
    parameters=model.actor.trainable();optimizer=torch.optim.AdamW(parameters,lr=.0003,weight_decay=1e-6)
    ema=deepcopy(model).eval().requires_grad_(False)
    rng=torch.Generator(device='cuda').manual_seed(531761)
    start_fraction=.1 if args.env=='cube' else .25
    validation=training_pair(model,val.sample(4096,rng,start_fraction=start_fraction),rng)
    config={'args':vars(args),'source':source,'train_sha256':digest_file(args.train),'validation_sha256':digest_file(args.validation),
        'architecture':'Original action base plus classical LoRA/FiLM adapter, with current candidate action appended to its existing context input. Original first action-head logits give residual edit; no additional learned model.',
        'loss':'Own-skill minimal action-edit MSE plus .05 residual-logit MSE; intervention examples weighted5x. Single-skill counterfactual contexts/candidates only. Cube yaw and gripper copied exactly from candidate.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    out.mkdir(parents=True);(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            model.train();b=training_pair(model,train.sample(args.batch,start_fraction=start_fraction))
            value,_=loss(model,b);optimizer.zero_grad(set_to_none=True);value.backward()
            torch.nn.utils.clip_grad_norm_(parameters,5);optimizer.step()
            with torch.no_grad():
                rate=.99 if step<1000 else .999
                for a,c in zip(ema.actor.parameters(),model.actor.parameters()):
                    if c.requires_grad:a.lerp_(c,1-rate)
            if step%1000==0 or step==args.steps:
                ema.eval()
                with torch.no_grad():
                    v,pred=loss(ema,validation);score=float(v)
                    dim=3 if args.env=='cube' else 2
                    rmse=float((pred[:,:dim]-validation['target'][:,:dim]).square().mean().sqrt())
                    changed=validation['changed'];mistake=(pred[:,:dim]-validation['candidate'][:,:dim]).norm(dim=-1)<.02
                    false_fixed=float((mistake[changed]).float().mean()) if bool(changed.any()) else 0.
                assert model.actor.frozen_digest()==frozen and ema.actor.frozen_digest()==frozen
                row={'step':step,'loss':float(value.detach()),'validation_loss':score,'validation_action_rmse':rmse,
                     'false_identity_fraction_on_interventions':false_fixed,'seconds':time.time()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                saved={**compact(ema),'step':step,'metrics':row,'config':config};torch.save(saved,out/'latest.pt')
                if score<best:best=score;torch.save(saved,out/'best.pt')
    assert digest_file(source['checkpoint'])==source['sha256']
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_validation_loss':best,'base_and_encoder_frozen':True,
        'new_auxiliary_networks':0,'current_action_input_only':True})+'\n')


if __name__=='__main__':main()
