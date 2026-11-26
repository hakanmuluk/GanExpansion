"""Supervise original action adapters on known whole-behavior skill codes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .coded_action_adapters import initialize,load,expand
from .windows_data import Windows
from .legacy import digest_file


def reconstruction(actor,b,environment,shuffle=False):
    code=b['code'].roll(1,0) if shuffle else b['code']
    if environment=='cube':
        prediction,logits=actor(b['state'],b['base_z'],code,b['context'],return_logits=True)
        error=(prediction[:,:,:4]-b['actions'][:,:,:4]).square().mean(-1)
        error+=.3*F.binary_cross_entropy_with_logits(2*logits[:,:,4],(b['actions'][:,:,4]+1)/2,reduction='none')
        first=(prediction[:,0,:4]-b['actions'][:,0,:4]).abs().mean()
    else:
        prediction=actor(b['state'],b['base_z'],code,b['context'])
        error=(prediction-b['actions']).square().mean(-1)
        first=(prediction[:,0]-b['actions'][:,0]).abs().mean()
    return (error*b['mask']).sum()/b['mask'].sum().clamp_min(1)+.2*first


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=15000);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--initialize')
    p.add_argument('--counterfactual-weight',type=float,default=0.)
    p.add_argument('--physics-weight',type=float,default=0.)
    p.add_argument('--expand-adapter',action='store_true')
    p.add_argument('--counterfactual-margin',type=float,default=.005)
    p.add_argument('--waypoint-core',type=float,default=1.);args=p.parse_args()
    if args.counterfactual_weight and args.env!='cube':p.error('Counterfactual carry labels are Cube-only')
    if args.physics_weight and args.env!='tworoom':p.error('Own action safety loss is TwoRoom-only')
    torch.set_num_threads(2);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    train=Windows(args.train);validation=Windows(args.validation)
    assert not train.context_fingerprints&validation.context_fingerprints
    code_mean=train.data['new_z'].mean(0);code_std=train.data['new_z'].std(0).clamp_min(.1)
    for data in [train,validation]:data.data['new_z']=(data.data['new_z']-code_mean)/code_std
    code_dim=train.data['new_z'].shape[-1]
    if args.initialize:
        actor,saved=load(args.initialize);source=saved['config']['source']
        assert saved['environment']==args.env and saved['skill']==args.skill and actor.config.new_z_dim==code_dim
        assert saved['config']['code_normalization']['mean']==code_mean.tolist()
        assert saved['config']['code_normalization']['std']==code_std.tolist()
        for name,parameter in actor.named_parameters():parameter.requires_grad_(not name.startswith(('base.','original_encoder.')))
    else:actor,source=initialize(args.env,args.skill,code_dim)
    if args.expand_adapter:actor=expand(actor,args.env)
    frozen=actor.frozen_digest();parameters=actor.trainable()
    assert all(not p.requires_grad for p in actor.base.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.0003,weight_decay=1e-6)
    alignment=8 if args.env=='cube' else 1
    rng=torch.Generator(device='cuda').manual_seed(776341)
    vb=validation.sample(4096,rng,alignment=alignment,start_fraction=.25)
    def counterfactual_loss(actor,b):
        pred,logits=actor(b['state'],b['base_z'],b['code'],b['context'],return_logits=True)
        return F.mse_loss(pred[:,0,:3],b['action'][:,:3])+.1*pred[:,0,3].square().mean()+.3*F.binary_cross_entropy_with_logits(2*logits[:,0,4],(b['action'][:,4]+1)/2)
    cv=None
    physics_vb={k:v[:512] for k,v in vb.items()}
    if args.counterfactual_weight:
        from .coded_action_labels import cube_counterfactuals
        cv=cube_counterfactuals(vb,code_mean,code_std,args.skill,args.counterfactual_margin,args.waypoint_core)
        assert cv is not None
    if args.physics_weight:from .coded_action_physics import tworoom_own_loss
    config={'args':vars(args),'source':source,'code_dim':code_dim,'train_sha256':digest_file(args.train),
        'code_normalization':{'mean':code_mean.tolist(),'std':code_std.tolist(),'policy_code_space':'Standardized own-demonstration code; no learned transform.'},
        'validation_sha256':digest_file(args.validation),'architecture':'Original frozen action base plus original LoRA/FiLM adapter; existing skill mapping input extended to known whole-behavior code. No new learned module; no path or noisy-action inputs.',
        'loss':'Masked action MSE, plus .2 first-control MAE; Cube includes .3 gripper BCEWithLogits. Codes and labels from own-skill demonstrations only.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    if args.initialize:config['initialization_sha256']=digest_file(args.initialize)
    if args.expand_adapter:
        config['architecture']+=' Same adapter graph widened: rank16, context256, modulation rank32, Cube output rank64; original base and action head unchanged.'
        config['trainable_adapter_parameters']=sum(p.numel() for p in parameters)
    if args.counterfactual_weight:
        config['loss']+=' Plus training-only counterfactual own-region/code first-control supervision at the SAME recorded carry states. No joint region, no learned teacher or auxiliary network. Foreground XYZ MSE + yaw penalty + gripper BCE.'
        config['training_only_teacher_margin_m']=args.counterfactual_margin
        config['training_only_waypoint_target_radius_fraction']=args.waypoint_core
    if args.physics_weight:
        config['loss']+=' Plus training-only own-skill/wall/boundary safety of 16 generated actions on recorded and random codes, every fourth update. Waypoint-before-goal ordering uses own visit state. No geometry or transition queried in inference.'
    out.mkdir(parents=True);(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    best=float('inf');best_metrics=None;started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            actor.train();b=train.sample(args.batch,alignment=alignment,start_fraction=.25)
            loss=reconstruction(actor,b,args.env)
            if args.counterfactual_weight:
                cf=cube_counterfactuals(b,code_mean,code_std,args.skill,args.counterfactual_margin,args.waypoint_core)
                if cf is not None:loss=loss+args.counterfactual_weight*counterfactual_loss(actor,cf)
            if args.physics_weight and step%4==0:
                small={k:v[:128] for k,v in b.items()}
                own,_=tworoom_own_loss(actor,small,args.skill)
                prior,_=tworoom_own_loss(actor,small,args.skill,True)
                loss=loss+args.physics_weight*.5*(own+prior)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,5);optimizer.step()
            if step%1000==0 or step==args.steps:
                actor.eval()
                with torch.no_grad():
                    value=float(reconstruction(actor,vb,args.env));shuffled=float(reconstruction(actor,vb,args.env,True))
                    cf_value=float(counterfactual_loss(actor,cv)) if cv is not None else 0.
                    physics_value=float(tworoom_own_loss(actor,physics_vb,args.skill)[0]) if args.physics_weight else 0.
                score=value+args.counterfactual_weight*cf_value+args.physics_weight*physics_value
                assert actor.frozen_digest()==frozen
                row={'step':step,'loss':float(loss.detach()),'validation_action_loss':value,
                     'shuffled_code_validation_loss':shuffled,'counterfactual_own_control_loss':cf_value,
                     'own_action_physics_loss':physics_value,'seconds':time.time()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                base='artifacts/final/base_policy.pt' if args.env=='tworoom' else 'cube/artifacts/final/base_policy.pt'
                saved={'environment':args.env,'skill':args.skill,'adapter_config':asdict(actor.config),'addon':actor.addon_state(),
                       'base_sha256':digest_file(base),'step':step,'metrics':row,'config':config}
                torch.save(saved,out/'latest.pt')
                if score<best:best=score;best_metrics=row;torch.save(saved,out/'best.pt')
    assert digest_file(source['checkpoint'])==source['sha256']
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_validation_action_loss':best_metrics['validation_action_loss'],
        'best_validation_selection_score':best,'base_and_encoder_frozen':True,'new_auxiliary_learned_networks':0})+'\n')


if __name__=='__main__':main()
