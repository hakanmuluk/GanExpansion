"""Calibrate only the new encoder on actual fixed-generator training rollouts."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .evaluate import load_adapter,load_contexts
from .parallel_eval import evaluate_parallel
from .models import state_digest,SkillEncoder
from .geometry import MAX_STEPS


def physical_data(folder,device):
    folder=Path(folder)
    # A compact delivery can also contain a few illustrative raw episodes.
    # Always use its complete, verified archive rather than that partial set.
    if (folder/'episodes.npz').is_file():
        with np.load(folder/'episodes.npz') as packed:
            names=['states','actions','contexts','lengths','base_z','new_z']
            return {k:torch.as_tensor(packed[k],device=device) for k in names},str(packed['original_trajectory_sha256'])
    paths=sorted(folder.glob('learner_*.npz')) or sorted(folder.glob('rollout_*.npz'))
    if not paths:raise ValueError('No physical trajectories')
    samples=[];digest=hashlib.sha256()
    for p in paths:
        digest.update(p.name.encode());digest.update(hashlib.sha256(p.read_bytes()).digest())
        with np.load(p) as data:samples.append({k:data[k] for k in ['states','actions','contexts','base_z','new_z']})
    arrays={}
    for key in ['states','actions','contexts']:
        x=np.zeros((len(samples),MAX_STEPS,*samples[0][key].shape[1:]),np.float32)
        for i,s in enumerate(samples):x[i,:len(s[key])]=s[key]
        arrays[key]=x
    arrays['lengths']=np.array([len(s['actions']) for s in samples],np.int64)
    for key in ['base_z','new_z']:arrays[key]=np.stack([s[key] for s in samples]).astype(np.float32)
    return {k:torch.as_tensor(v,device=device) for k,v in arrays.items()},digest.hexdigest()


def prediction(encoder,data,indices):
    return encoder(data['states'][indices],data['actions'][indices],data['contexts'][indices],data['base_z'][indices],data['lengths'][indices])


@torch.no_grad()
def metrics(encoder,data):
    encoder.eval();pred=[]
    for ids in torch.arange(len(data['states']),device=data['states'].device).split(64):pred.append(prediction(encoder,data,ids)[0])
    x=data['new_z'];y=torch.cat(pred)
    return {'physical_latent_rmse':float((x-y).square().mean().sqrt()),'physical_latent_mae':float((x-y).abs().mean()),
            'physical_latent_sign_accuracy':float((x.sign()==y.sign()).float().mean())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--physical-train',required=True)
    p.add_argument('--base',default='cube/artifacts/final/base_policy.pt');p.add_argument('--val-contexts',required=True)
    p.add_argument('--skill',required=True,choices=['keepout','waypoint']);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--batch',type=int,default=64);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--eval-every',type=int,default=250);p.add_argument('--val-count',type=int,default=24);p.add_argument('--val-codes',type=int,default=8)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');p.add_argument('--resume')
    p.add_argument('--geometry-encoder',action='store_true',help='Fit a smaller full-episode encoder to observed task geometry and executed actions')
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((Path(args.physical_train)/'summary.json').read_text());source_sha=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if provenance.get('generator_checkpoint_sha256')!=source_sha:raise ValueError('Calibration rollouts must come from these exact frozen actor weights')
    if Path(provenance['args']['contexts']).name!='train_contexts.npz':raise ValueError('Calibration must use training contexts only')
    model,encoder,source=load_adapter(args.checkpoint,args.base,args.device);model.eval().requires_grad_(False)
    if model.config.no_new_latent:raise ValueError('A deterministic comparator has no skill-code encoder to calibrate')
    frozen=state_digest(model)
    train,data_sha=physical_data(args.physical_train,args.device)
    if len(train['states'])!=provenance['actual_learner_rollouts']:raise ValueError('Incomplete physical training inputs; restore the complete archive')
    val_dir=out/'validation_rollouts'
    if not (val_dir/'summary.json').is_file():
        evaluate_parallel(args.checkpoint,args.base,load_contexts(args.val_contexts,args.val_count),args.skill,args.val_codes,seed=878,
                          workers=args.workers,out=val_dir)
    settings=json.loads((val_dir/'evaluation_settings.json').read_text())
    if settings.get('checkpoint_sha256')!=source_sha:raise ValueError('Cached validation actor differs')
    val,val_sha=physical_data(val_dir,args.device)
    if len(val['states'])!=json.loads((val_dir/'summary.json').read_text())['episodes']:raise ValueError('Incomplete physical validation inputs; restore the complete archive')
    before=metrics(encoder,val)
    encoder_config=source.get('encoder_config',{'hidden':128,'geometry_only':False})
    if args.geometry_encoder:
        encoder=SkillEncoder(model.base,model.config.new_z_dim,hidden=64,context_dim=model.config.context_dim,geometry_only=True).to(args.device)
        encoder_config={'hidden':64,'geometry_only':True}
    norm_before=state_digest_buffers(encoder);optimizer=torch.optim.Adam(encoder.parameters(),lr=args.lr)
    best=before['physical_latent_rmse'];started=time.time();start_step=0
    if args.resume:
        resume=torch.load(args.resume,map_location=args.device,weights_only=False)
        if resume.get('stage')!='encoder_calibration' or resume['calibration']['generator_checkpoint_sha256']!=source_sha:raise ValueError('Calibration resume source differs')
        encoder.load_state_dict(resume['encoder']);optimizer.load_state_dict(resume['encoder_optimizer'])
        start_step=resume['calibration']['step'];best=resume['calibration']['best_rmse']
        if args.steps<=start_step:raise ValueError('--steps must extend beyond the calibration resume checkpoint')
        torch.set_rng_state(resume['torch_rng'].cpu())
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all([s.cpu() for s in resume['cuda_rng']])
    description={'args':vars(args),'stage':'new_encoder_physical_calibration','generator_checkpoint_sha256':source_sha,
                 'physical_training_rollouts':len(train['states']),'training_trajectory_sha256':data_sha,'validation_trajectory_sha256':val_sha,
                 'pre_calibration':before,'generator_frozen_digest':frozen,'extra_expert_queries_in_calibration':0}
    (out/'config.json').write_text(json.dumps(description,indent=2)+'\n')
    # Keep an initial checkpoint as the fallback if calibration does not improve.
    if not (out/'best.pt').is_file():
        prior_best=Path(args.resume).parent/'best.pt' if args.resume else None
        if prior_best is not None and prior_best.is_file():shutil.copy2(prior_best,out/'best.pt')
        else:torch.save(source,out/'best.pt')
    with (out/'calibration.jsonl').open('a',buffering=1) as log:
        for step in range(start_step+1,args.steps+1):
            encoder.train();ids=torch.randint(len(train['states']),(args.batch,),device=args.device)
            mu,lv=prediction(encoder,train,ids)
            residual=mu-train['new_z'][ids]
            nll=.5*(residual.square()*(-lv).exp()+lv).mean()
            loss=residual.abs().mean()+.1*nll+.0025*(mu.square()+lv.exp()-1-lv).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(encoder.parameters(),5);optimizer.step()
            if step%args.eval_every==0 or step==args.steps:
                measured=metrics(encoder,val);row={'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.time()-started,**measured}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                assert state_digest(model)==frozen and state_digest_buffers(encoder)==norm_before
                improved=measured['physical_latent_rmse']<best;best=min(best,measured['physical_latent_rmse'])
                ck={**source,'encoder':encoder.state_dict(),'encoder_config':encoder_config,'calibration':{**description,'step':step,'validation':measured,'best_rmse':best}}
                if improved:torch.save(ck,out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
                resume={**ck,'stage':'encoder_calibration','encoder_optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None}
                torch.save(resume,out/'latest.tmp');(out/'latest.tmp').replace(out/'latest.pt')
    (out/'DONE.json').write_text(json.dumps({'best_physical_latent_rmse':best,'generator_unchanged':state_digest(model)==frozen,
                                          'normalization_unchanged':state_digest_buffers(encoder)==norm_before,'elapsed_seconds':time.time()-started},indent=2)+'\n')


def state_digest_buffers(module):
    digest=hashlib.sha256()
    for key,value in sorted(module.named_buffers()):digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


if __name__=='__main__':main()
