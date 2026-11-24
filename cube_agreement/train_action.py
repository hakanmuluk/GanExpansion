"""Retrain the original StyleGAN action architecture on rich SINGLE-skill routes."""
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from cube_addons.losses import reconstruction
from .action_model import initialize,path_context,reference_context


def dataset(root,device):
    records=[];fields={k:[] for k in ['states','base_z','contexts','targets','mask','phase','validation']};sources=[]
    roots=root if isinstance(root,list) else [root]
    for path in sorted(p for r in roots for p in Path(r).glob('*/episode_*.npz')):
        with np.load(path) as f:trace={k:f[k] for k in f.files}
        n=len(trace['actions']);ref=reference_context(trace);cube=trace['poses'][:-1,3:];route=np.broadcast_to(trace['path'],(n,*trace['path'].shape)).copy()
        context=path_context(torch.tensor(ref),torch.tensor(cube),torch.tensor(route,dtype=torch.float32),torch.tensor(trace['path_indices'],dtype=torch.long)).numpy()
        padded=np.pad(trace['actions'],[(0,15),(0,0)]);target=np.stack([padded[i:i+16] for i in range(n)]);mask=(np.arange(n)[:,None]+np.arange(16)[None]<n).astype(np.float32)
        if bool(trace.get('is_feedback',False)):mask[:,1:]=0
        fields['states'].append(trace['states']);fields['base_z'].append(np.broadcast_to(trace['base_z'],(n,2)));fields['contexts'].append(context);fields['targets'].append(target);fields['mask'].append(mask);fields['phase'].append(trace['teacher_phase']);fields['validation'].append(np.full(n,int(trace['seed'])%5==0))
        sources.append({'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'skill':str(trace['skill']),'seed':int(trace['seed']),'steps':n})
    arrays={k:torch.as_tensor(np.concatenate(v),device=device) for k,v in fields.items()}
    return arrays,sources


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',nargs='+',default=['cube_agreement_artifacts/single_train_v1']);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=6000);p.add_argument('--batch',type=int,default=256);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed',type=int,default=42);p.add_argument('--ungated',action='store_true');p.add_argument('--resume');p.add_argument('--control-augmentation',type=float,default=0.);args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);device='cuda';actor=initialize(device=device,ungated=args.ungated);frozen=actor.frozen_digest();data,sources=dataset(args.data,device);optimizer=torch.optim.Adam(actor.trainable(),lr=args.lr,betas=(.5,.99))
    if args.resume:actor.load_state_dict(torch.load(args.resume,map_location=device,weights_only=False)['addon'],strict=False)
    groups=[torch.nonzero((~data['validation'])&(data['phase']==i)).flatten() for i in range(8)];groups=[g for g in groups if len(g)]
    validation=torch.nonzero(data['validation']).flatten();validation=validation[torch.linspace(0,len(validation)-1,min(4096,len(validation)),device=device).long()]
    config={'args':vars(args),'training_scope':'Independent single-skill native episodes and optional independent single-skill feedback labels only. No joint contexts, joint expert actions, composed-policy feedback or joint-task losses. Feedback labels supervise only the first control to avoid treating off-policy future labels as on-policy action chunks.','architecture':'Original StyleGAN backbone frozen, independent old encoder frozen, original waypoint adapter expanded with a 70-feature planned-route context and retrained. Full 16x5 action chunks including grasp/transport/release are predicted by the neural actor.','sources':sources,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    config['control_augmentation']='Optional training-only counterfactual local target perturbations at held transport states. The single-skill PathTeacher control rule labels only the first action; the network must respond to the local target rather than memorize the old route from observed state. These generic controller labels do not train either constraint decoder and use no joint scene/constraint pair.'
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');best=float('inf');started=time.perf_counter()
    def forward(ids):
        bz=data['base_z'][ids].float();ctx=data['contexts'][ids].float();s=data['states'][ids].float();z=torch.full((len(ids),1),.8,device=device)
        return actor(s,bz,z,ctx,return_logits=True)
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            actor.train();per=(args.batch+len(groups)-1)//len(groups);ids=torch.cat([g[torch.randint(len(g),(per,),device=device)] for g in groups])[:args.batch]
            pred,logits=forward(ids);target=data['targets'][ids].float();mask=data['mask'][ids].float()
            rec=reconstruction(pred,target,mask,logits)
            # The actually executed first control receives explicit supervision.
            first=F.l1_loss(pred[:,0,:4],target[:,0,:4])+.3*F.binary_cross_entropy_with_logits(2*logits[:,0,4],(target[:,0,4]+1)/2)
            loss=rec+first
            if args.control_augmentation:
                carry=ids[data['phase'][ids]==4]
                if len(carry):
                    c=data['contexts'][carry].float().clone();bz=data['base_z'][carry].float();s=data['states'][carry].float()
                    c[:,20:23]+=args.control_augmentation*torch.randn((len(carry),3),device=device)
                    cp,cl=actor(s,bz,torch.full((len(carry),1),.8,device=device),c,return_logits=True)
                    probability=.5*(1+torch.erf(bz[:,1]/2**.5));speed=torch.where(probability<.5,.65+.4*probability,.7+.3*probability)
                    limit=torch.minimum(speed,torch.full_like(speed,.65))[:,None]
                    label=(4*speed[:,None]*c[:,20:23]).clamp(-limit,limit)
                    loss=loss+2*F.mse_loss(cp[:,0,:3],label)+.1*cp[:,0,3].square().mean()+.1*F.binary_cross_entropy_with_logits(2*cl[:,0,4],torch.ones(len(carry),device=device))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.trainable(),5);optimizer.step()
            if step%500==0 or step==args.steps:
                actor.eval()
                with torch.no_grad():
                    vp,vl=forward(validation);vt=data['targets'][validation].float();vm=data['mask'][validation].float();val_rec=reconstruction(vp,vt,vm,vl)
                    mse=float((((vp[:,:,:4]-vt[:,:,:4])**2)*vm[:,:,None]).sum()/(4*vm.sum()));grip=float((((vp[:,:,4]>=0)==(vt[:,:,4]>=0))*vm).sum()/vm.sum())
                    first_mse=(vp[:,0,:4]-vt[:,0,:4]).square().mean()
                    value=val_rec+F.l1_loss(vp[:,0,:4],vt[:,0,:4])+.3*F.binary_cross_entropy_with_logits(2*vl[:,0,4],(vt[:,0,4]+1)/2)
                    phases={str(i):float((vp[data['phase'][validation]==i,0,:3]-vt[data['phase'][validation]==i,0,:3]).square().mean().sqrt()) for i in range(8) if (data['phase'][validation]==i).any()}
                metric={'step':step,'loss':float(loss),'validation_selection_loss':float(value),'validation_reconstruction':float(val_rec),'validation_first_action_RMSE':float(first_mse.sqrt()),'validation_phase_first_XYZ_RMSE':phases,'validation_action_RMSE':mse**.5,'validation_grip_accuracy':grip,'seconds':time.perf_counter()-started};log.write(json.dumps(metric)+'\n');print(metric,flush=True)
                assert actor.frozen_digest()==frozen
                ck={'adapter_config':asdict(actor.config),'addon':actor.addon_state(),'base_sha256':hashlib.sha256(Path('cube/artifacts/final/base_policy.pt').read_bytes()).hexdigest(),'step':step,'metric':metric,'config':config,'frozen_digest':frozen,'optimizer':optimizer.state_dict()}
                torch.save(ck,out/'latest.pt')
                if float(value)<best:best=float(value);torch.save(ck,out/'best.pt')
    (out/'DONE.json').write_text(json.dumps({'complete':True,'best_validation_reconstruction':best,'base_and_original_encoder_frozen':True}))


if __name__=='__main__':main()
