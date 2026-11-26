"""Original Cube trainer plus paired code and magnitude-effect supervision."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from cube.cube_gan.models import kl_loss
from cube_addons.losses import reconstruction
from cube_addons.data import EpisodeDataset
from cube_addons.models import AdapterConfig,GatedGenerator,SkillEncoder,SkillDiscriminator
from cube_addons.evaluate import load_contexts,evaluate
from .counterfactual_action_loss import build_pairs, action_difference_loss
from .counterfactual_magnitude_loss import build_quartets, magnitude_difference_loss
from .observation_replay import install_single_skill_restore
install_single_skill_restore()


def expand(z,n):return z[:,None].expand(-1,n,-1)


def parser():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--skill',required=True,choices=['keepout','waypoint'])
    p.add_argument('--data',default='cube_addon_artifacts/data_force');p.add_argument('--base',default='cube/artifacts/final/base_policy.pt')
    p.add_argument('--discriminator-source',default='cube/artifacts/final/training_resume.pt')
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=42);p.add_argument('--resume');p.add_argument('--initialize')
    p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch',type=int,default=64);p.add_argument('--chunks',type=int,default=4)
    p.add_argument('--episodes',type=int);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--base-lr',type=float,default=.00005)
    p.add_argument('--rank',type=int,default=4);p.add_argument('--context-hidden',type=int,default=128);p.add_argument('--modulation-rank',type=int,default=8)
    p.add_argument('--output-rank',type=int,default=8)
    p.add_argument('--rec-weight',type=float,default=10.);p.add_argument('--known-rec-weight',type=float,default=10.)
    p.add_argument('--label-weight',type=float,default=2.);p.add_argument('--latent-weight',type=float,default=1.)
    p.add_argument('--kl-weight',type=float,default=.005);p.add_argument('--adv-weight',type=float,default=.03);p.add_argument('--keep-weight',type=float,default=.1)
    p.add_argument('--cycle-batch',type=int,default=4);p.add_argument('--cycle-every',type=int,default=4)
    p.add_argument('--transport-weight',type=float,default=1.,help='Relative supervised chunk weight during lifted transport')
    p.add_argument('--separate-clip',action='store_true',help='Clip actor and new-encoder gradients independently')
    p.add_argument('--legacy-gripper-loss',action='store_true',help='Reproduce earlier probability-clamped reconstruction')
    p.add_argument('--legacy-cycle-grip',action='store_true',help='Reproduce earlier continuous gripper tokens in teacher-state cycle')
    p.add_argument('--eval-every',type=int,default=1000);p.add_argument('--eval-contexts',type=int,default=16);p.add_argument('--eval-codes',type=int,default=4)
    p.add_argument('--eval-workers',type=int,default=4)
    p.add_argument('--execute',type=int,choices=[1,2,4,8,16],default=8)
    p.add_argument('--aligned-chunks',type=int,default=1)
    p.add_argument('--cycle-execute',type=int,default=1,choices=[1,2,4,8,16])
    p.add_argument('--no-new-latent',action='store_true');p.add_argument('--finetune-base',action='store_true');p.add_argument('--output-modulation',action='store_true')
    p.add_argument('--transport-gate',action='store_true')
    p.add_argument('--ignore-time',action='store_true',help='Condition the adapter on geometry and observed task events, excluding elapsed episode time')
    p.add_argument('--single-strategy',choices=['positive','negative'])
    p.add_argument('--feedback');p.add_argument('--feedback-weight',type=float,default=5.)
    p.add_argument('--counterfactual-weight',type=float,default=20.)
    p.add_argument('--counterfactual-batch',type=int,default=64)
    p.add_argument('--counterfactual-horizon',type=int,default=8)
    p.add_argument('--magnitude-weight',type=float,default=100.)
    p.add_argument('--magnitude-batch',type=int,default=32)
    p.add_argument('--magnitude-horizon',type=int,default=8)
    return p


def main():
    args=parser().parse_args();run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    # Disabling TF32 keeps the exact-output invariant comparable across codepaths.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=AdapterConfig(rank=args.rank,context_hidden=args.context_hidden,modulation_rank=args.modulation_rank,
                         no_new_latent=args.no_new_latent,finetune_base=args.finetune_base,output_modulation=args.output_modulation,output_rank=args.output_rank,transport_gate=args.transport_gate,ignore_time=args.ignore_time)
    model=GatedGenerator(args.base,config,args.device)
    model.execute_steps=args.execute
    encoder=SkillEncoder(model.base,config.new_z_dim,context_dim=config.context_dim).to(args.device)
    source=torch.load(args.discriminator_source,map_location='cpu',weights_only=False)['discriminator']
    discriminator=SkillDiscriminator(model.base,source,config.context_dim).to(args.device)
    data=EpisodeDataset(Path(args.data)/args.skill/'train.npz',args.device,args.episodes,args.single_strategy,args.aligned_chunks)
    val=load_contexts(Path(args.data)/args.skill/'val_contexts.npz',args.eval_contexts)
    params=model.trainable()+([] if args.no_new_latent else list(encoder.parameters()))
    if args.finetune_base:
        groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('active_backbone.')],'lr':args.base_lr,'name':'backbone'},
                {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('active_backbone.')]+([] if args.no_new_latent else list(encoder.parameters())),'lr':args.lr,'name':'new'}]
    else:groups=params
    optimizer=torch.optim.Adam(groups,lr=args.lr,betas=(.5,.99))
    optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=args.lr*.5,betas=(.5,.99))
    feedback=None
    if args.feedback:
        with np.load(args.feedback) as f:feedback={k:torch.as_tensor(f[k],device=args.device) for k in f.files}
        if args.single_strategy:
            select=feedback['new_z'][:,0]>=0
            if args.single_strategy=='negative':select=~select
            feedback={k:v[select] for k,v in feedback.items()}
    assert args.feedback and not args.no_new_latent and not args.finetune_base
    pairs,pair_description=(build_pairs(feedback) if args.counterfactual_weight else
        (None,{'disabled':True}))
    pair_generator=torch.Generator(device=args.device).manual_seed(args.seed+903)
    assert args.counterfactual_weight>=0 and args.magnitude_weight>=0
    quartets,magnitude_description=(build_quartets(feedback) if args.magnitude_weight else (None,{'disabled':True}))
    magnitude_generator=torch.Generator(device=args.device).manual_seed(args.seed+1903)
    base_sha=hashlib.sha256(Path(args.base).read_bytes()).hexdigest();frozen=model.frozen_digest()
    if args.initialize and args.resume:raise ValueError('Choose initialization or full optimizer resume')
    if args.initialize:
        ck=torch.load(args.initialize,map_location=args.device,weights_only=False)
        old_config=AdapterConfig(**ck['adapter_config'])
        old_description=asdict(old_config);new_description=asdict(config)
        old_description.pop('output_rank');new_description.pop('output_rank')
        old_description.pop('ignore_time');new_description.pop('ignore_time')
        if ck['base_sha256']!=base_sha or old_description!=new_description or config.output_rank<old_config.output_rank:raise ValueError('Initialization base/architecture mismatch')
        if set(ck['addon'])!=set(model.addon_state()):raise ValueError('Incomplete initialization adapter')
        addon=ck['addon']
        if config.output_rank!=old_config.output_rank:
            addon={k:v.clone() for k,v in model.addon_state().items()}
            for key,value in ck['addon'].items():
                if addon[key].shape==value.shape:addon[key].copy_(value)
                elif key in ['output_modulation.0.weight','output_modulation.0.bias']:addon[key][:len(value)].copy_(value)
                elif key=='output_modulation.2.weight':addon[key].zero_();addon[key][:,:value.shape[1]].copy_(value)
                else:raise ValueError(f'Unsupported initialization expansion: {key}')
        model.load_state_dict(addon,strict=False);encoder.load_state_dict(ck['encoder'])
        if 'discriminator' in ck:discriminator.load_state_dict(ck['discriminator'])
    start_step=0;best=-float('inf')
    if args.resume:
        ck=torch.load(args.resume,map_location=args.device,weights_only=False)
        if ck.get('stage')=='encoder_calibration' or 'optimizer' not in ck:
            raise ValueError('Use the generator-training latest.pt resume state, not compact or encoder-calibration weights')
        if ck['base_sha256']!=base_sha or asdict(AdapterConfig(**ck['adapter_config']))!=asdict(config):raise ValueError('Resume base/architecture mismatch')
        if set(ck['addon'])!=set(model.addon_state()):raise ValueError('Incomplete adapter')
        model.load_state_dict(ck['addon'],strict=False);encoder.load_state_dict(ck['encoder']);discriminator.load_state_dict(ck['discriminator'])
        optimizer.load_state_dict(ck['optimizer']);optimizer_d.load_state_dict(ck['optimizer_d'])
        start_step=ck['step'];best=ck['best_score'] if Path(args.resume).parent.resolve()==run.resolve() else -float('inf')
        if args.steps<=start_step:raise ValueError('--steps must extend beyond the resume checkpoint')
        torch.set_rng_state(ck['torch_rng'].cpu());np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        if args.counterfactual_weight:
            if 'counterfactual_rng' not in ck:raise ValueError('Use --initialize for a checkpoint without paired-loss sampler state')
            pair_generator.set_state(ck['counterfactual_rng'].cpu())
        if args.magnitude_weight:
            if 'magnitude_rng' not in ck:raise ValueError('Use --initialize for a checkpoint without magnitude sampler state')
            magnitude_generator.set_state(ck['magnitude_rng'].cpu())
        for group in optimizer.param_groups:group['lr']=args.base_lr if group.get('name')=='backbone' else args.lr
        for group in optimizer_d.param_groups:group['lr']=args.lr*.5
    info={'args':vars(args),'adapter_config':asdict(config),'base_sha256':base_sha,'data_sha256':hashlib.sha256((Path(args.data)/args.skill/'train.npz').read_bytes()).hexdigest(),
          'demonstrations_used':data.size,'normalization':'All original means/std buffers are copied unchanged and never fitted on new data.',
          'counts':{'frozen_generator':sum(p.numel() for p in model.base.parameters()),'frozen_original_encoder':sum(p.numel() for p in model.original_encoder.parameters()),
                    'trainable_generator':sum(p.numel() for p in model.trainable()),'new_encoder':sum(p.numel() for p in encoder.parameters()),'new_discriminator':sum(p.numel() for p in discriminator.parameters())},
          'auxiliary_supervision':'Known Gaussian expert codes supervise E and paired action reconstruction; set label/known-rec weights to zero for vanilla BicycleGAN.',
          'cycle_definition':'Generated actions at recorded teacher states; actual closed-loop recovery evaluated separately.',
          'gripper_reconstruction':'legacy probability-clamped BCE' if args.legacy_gripper_loss else 'Stable BCEWithLogits(2 * pre-tanh gripper)',
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
          'torch':torch.__version__,'gpu':torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu'}
    info['paired_own_teacher_differences']={'original_trainer_sha256':'339186ce4d4b7f96fe62e93bac705f3564f62cbc01a85c5e3ebcfdcbb4da26df','scope':'Only added own-teacher action-difference loss. Exact same state, own context and base code; no joint labels or new neural component. Separate sampler RNG preserves original draw stream.','pair_description':pair_description,'native_reset':'complete derived observation reset'}
    info['own_magnitude_supervision']={'parent_paired_trainer_sha256':'2e308107e6c121c22ee6c764562350c9b42029f638eccb76258da3104b1086bd','scope':'Original neural classes/inputs. Compare own teacher even action components for +/-small and +/-large at exactly sameS/c/base. No addedNN, physical axes, jointlabels or inference changes. Separate sampler preserves original RNG stream.','quartet_description':magnitude_description}
    (run/'config.json').write_text(json.dumps(info,indent=2)+'\n');print('CONFIG',json.dumps(info),flush=True)
    started=time.time();running={};last_log=start_step
    def save(step,metric,improved):
        assert model.frozen_digest()==frozen,'Immutable original parameters/buffers changed'
        ck={'step':step,'args':vars(args),'adapter_config':asdict(config),'addon':model.addon_state(),'encoder':encoder.state_dict(),
            'base_sha256':base_sha,'execute_steps':args.execute,'frozen_digest_before':frozen,'frozen_digest_after':model.frozen_digest(),'best_score':best,'validation':metric,
            'discriminator':discriminator.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_d':optimizer_d.state_dict(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None,
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'counterfactual_rng':pair_generator.get_state(),'magnitude_rng':magnitude_generator.get_state()}
        torch.save(ck,run/'latest.tmp');(run/'latest.tmp').replace(run/'latest.pt')
        if improved:
            compact={k:v for k,v in ck.items() if k not in ['optimizer','optimizer_d','torch_rng','cuda_rng','numpy_rng','python_rng']}
            torch.save(compact,run/'best.tmp');(run/'best.tmp').replace(run/'best.pt')
    with (run/'train.jsonl').open('a',buffering=1) as log:
        for step in range(start_step+1,args.steps+1):
            model.train();encoder.train();discriminator.train();b=data.sample(args.batch,args.chunks)
            state,ctx,target,mask=b['states'],b['contexts'],b['actions'],b['mask'];bz=expand(b['base_z'],args.chunks)
            zero=torch.zeros((),device=args.device);mu=lv=None
            if args.no_new_latent:code=torch.zeros_like(b['new_z'])
            else:
                mu,lv=encoder(b['episode_states'],b['episode_actions'],b['episode_contexts'],b['base_z'],b['lengths'])
                inferred=mu+torch.randn_like(mu)*(.5*lv).exp()
                # A disclosed warm start for the paired auxiliary experiment.
                mix=min(1.,step/500) if args.known_rec_weight else 1.
                code=(1-mix)*b['new_z']+mix*inferred
            reconstructed,rec_logits=model(state,bz,expand(code,args.chunks),ctx,return_logits=True)
            prior=torch.randn_like(b['new_z']);fake=model(state,bz,expand(prior,args.chunks),ctx)
            gan=args.adv_weight*min(1.,max(0.,(step-500)/300))
            dl=zero;adv=zero
            s=state.flatten(0,1);c=ctx.flatten(0,1);z=bz.flatten(0,1);m=mask.flatten(0,1)
            if gan:
                discriminator.requires_grad_(True);optimizer_d.zero_grad(set_to_none=True)
                real=discriminator(s.detach(),target.flatten(0,1),c.detach(),z.detach(),m)
                fakes=torch.cat([reconstructed.detach().flatten(0,1),fake.detach().flatten(0,1)])
                fake_log=discriminator(s.detach().repeat(2,1),fakes,c.detach().repeat(2,1),z.detach().repeat(2,1),m.repeat(2,1))
                dl=F.relu(1-real).mean()+F.relu(1+fake_log).mean();dl.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(),5);optimizer_d.step()
            discriminator.requires_grad_(False)
            if gan:
                adv=-discriminator(s.repeat(2,1),torch.cat([reconstructed.flatten(0,1),fake.flatten(0,1)]),c.repeat(2,1),z.repeat(2,1),m.repeat(2,1)).mean()
            optimizer.zero_grad(set_to_none=True)
            transport=(ctx[...,14]>.5)&(ctx[...,10]>.05)&(ctx[...,10]<.9)
            importance=1+(args.transport_weight-1)*transport.to(mask.dtype)
            rec_mask=mask*importance[...,None]
            rec=reconstruction(reconstructed,target,rec_mask,rec_logits,legacy=args.legacy_gripper_loss)
            known=zero
            if args.known_rec_weight and not args.no_new_latent:
                known_pred,known_logits=model(state,bz,expand(b['new_z'],args.chunks),ctx,return_logits=True)
                known=reconstruction(known_pred,target,rec_mask,known_logits,legacy=args.legacy_gripper_loss)
            label=F.l1_loss(mu,b['new_z']) if not args.no_new_latent else zero
            kl=kl_loss(mu,lv) if not args.no_new_latent else zero
            cycle=zero
            if not args.no_new_latent and args.latent_weight and step%args.cycle_every==0:
                cb=min(args.cycle_batch,args.batch);es=b['episode_states'][:cb];ec=b['episode_contexts'][:cb];t=es.shape[1]
                cz=torch.randn_like(b['new_z'][:cb])
                if args.cycle_execute>1:
                    sampled=es[:,::args.cycle_execute];sampled_context=ec[:,::args.cycle_execute];n=sampled.shape[1]
                    plans=model(sampled,expand(b['base_z'][:cb],n),expand(cz,n),sampled_context)
                    synthetic=plans[:,:,:args.cycle_execute].reshape(cb,-1,5)[:,:t]
                else:synthetic=model(es,expand(b['base_z'][:cb],t),expand(cz,t),ec)[:,:,0]
                # The deployed controller binarizes grip; do the same for the
                # cycle so analog gripper confidence cannot carry the code.
                if not args.legacy_cycle_grip:synthetic=torch.cat([synthetic[...,:4],torch.where(synthetic[...,4:]>=0,1.,-1.)],-1)
                recovered,_=encoder(es,synthetic,ec,b['base_z'][:cb],b['lengths'][:cb]);cycle=F.l1_loss(recovered,cz)
            with torch.no_grad():reference=model.base(state,bz)
            # Proposal's per-step UNSQUARED Euclidean action correction penalty.
            keep=(((fake-reference).norm(dim=-1)+(reconstructed-reference).norm(dim=-1))*.5*mask).sum()/mask.sum().clamp_min(1)
            fb=zero
            if feedback is not None:
                ids=torch.randint(len(feedback['states']),(args.batch*args.chunks,),device=args.device)
                f={k:v[ids] for k,v in feedback.items()}
                transporting=(f['contexts'][...,14]>.5)&(f['contexts'][...,10]>.05)&(f['contexts'][...,10]<.9)
                fm=f['mask']*(1+(args.transport_weight-1)*transporting.to(f['mask'].dtype))[...,None]
                fb_pred,fb_logits=model(f['states'],f['base_z'],f['new_z'],f['contexts'],return_logits=True)
                fb=reconstruction(fb_pred,f['actions'],fm,fb_logits,legacy=args.legacy_gripper_loss)
            cf=zero;cf_metrics={}
            if args.counterfactual_weight:
                cf,cf_metrics=action_difference_loss(model,feedback,pairs,pair_generator,args.counterfactual_batch,args.counterfactual_horizon)
            loss=args.rec_weight*rec+args.known_rec_weight*known+args.label_weight*label+args.kl_weight*kl+args.latent_weight*cycle+gan*adv+args.keep_weight*keep+args.feedback_weight*fb
            magnitude=zero;magnitude_metrics={}
            if args.magnitude_weight:
                magnitude,magnitude_metrics=magnitude_difference_loss(model,feedback,quartets,magnitude_generator,args.magnitude_batch,args.magnitude_horizon)
            loss=loss+args.counterfactual_weight*cf+args.magnitude_weight*magnitude
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            grad_actor=zero;grad_encoder=zero
            if args.separate_clip:
                grad_actor=torch.nn.utils.clip_grad_norm_(model.trainable(),5)
                if not args.no_new_latent:grad_encoder=torch.nn.utils.clip_grad_norm_(encoder.parameters(),5)
                grad=(grad_actor.square()+grad_encoder.square()).sqrt()
            else:grad=torch.nn.utils.clip_grad_norm_(params,5)
            optimizer.step()
            values={'loss':loss,'rec':rec,'known_rec':known,'label':label,'kl':kl,'teacher_cycle':cycle,'adv':adv,'d_loss':dl,'keep':keep,'feedback':fb,'grad':grad}
            if args.separate_clip:values.update(grad_actor=grad_actor,grad_encoder=grad_encoder)
            values.update(cf_metrics);values.update(magnitude_metrics)
            for k,v in values.items():running[k]=running.get(k,0)+float(v.detach())
            if step%100==0 or step==args.steps:
                row={'step':step,'elapsed_seconds':time.time()-started,**{k:v/(step-last_log) for k,v in running.items()}}
                log.write(json.dumps(row)+'\n');print('TRAIN',json.dumps(row),flush=True);running={};last_log=step
            if step%args.eval_every==0 or step==args.steps:
                model.eval();encoder.eval()
                if args.eval_workers>1:
                    from cube_addons.parallel_eval import evaluate_parallel
                    validation_actor=run/'validation_actor.pt'
                    torch.save({'adapter_config':asdict(config),'addon':model.addon_state(),'encoder':encoder.state_dict(),'base_sha256':base_sha,'execute_steps':args.execute},validation_actor)
                    metric,_=evaluate_parallel(validation_actor,args.base,val,args.skill,args.eval_codes,seed=867,workers=args.eval_workers)
                    validation_actor.unlink()
                else:
                    metric,_=evaluate(model,None if args.no_new_latent else encoder,val,args.skill,args.eval_codes,seed=867)
                score=metric['success']+.15*metric['both_successful_strategies_per_context']+.05*metric['strategy_matches_code']
                # Strongly prefer real task success; recovery only breaks ties.
                score-=.005*metric.get('physical_latent_rmse',0.)
                improved=score>best;best=max(best,score)
                row={'step':step,'score':score,**metric}
                with (run/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print('VALIDATION',json.dumps(row),flush=True);save(step,metric,improved)
    (run/'DONE.json').write_text(json.dumps({'step':args.steps,'best_score':best,'elapsed_seconds':time.time()-started,'frozen_unchanged':model.frozen_digest()==frozen},indent=2)+'\n')


if __name__=='__main__':main()
