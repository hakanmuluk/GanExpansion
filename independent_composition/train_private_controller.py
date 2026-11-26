"""Train ONE skill's private path-conditioned action adapter on own data only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from .private_paths import initialize_controller,path_context
from .legacy import digest_file


def dataset(root,skill,device):
    rows=json.loads((Path(root)/'records.json').read_text())
    selected=[r for r in rows if r['skill']==skill and r['success']]
    fields={k:[] for k in ['states','base_z','contexts','targets','mask','phase','validation']};sources=[]
    for row in selected:
        path=Path(row['source'])
        with np.load(path) as d:trace={k:d[k] for k in d.files}
        assert str(trace['skill'])==skill and not bool(trace['is_feedback'])
        other=trace['geometry'][6:10] if skill=='keepout' else trace['geometry'][2:6]
        assert np.count_nonzero(other)==0
        n=len(trace['actions']);position=trace['poses'][:-1,3:]
        route=np.broadcast_to(trace['path'],(n,*trace['path'].shape)).copy()
        context=path_context(torch.tensor(trace['contexts']),torch.tensor(position),torch.tensor(route,dtype=torch.float32),
                             torch.tensor(trace['path_indices']),torch.tensor(trace['path_done'])).numpy()
        padded=np.pad(trace['actions'],[(0,15),(0,0)])
        fields['states'].append(trace['states']);fields['base_z'].append(np.broadcast_to(trace['base_z'],(n,2)))
        fields['contexts'].append(context);fields['targets'].append(np.stack([padded[i:i+16] for i in range(n)]))
        fields['mask'].append((np.arange(n)[:,None]+np.arange(16)[None]<n).astype(np.float32))
        fields['phase'].append(trace['teacher_phase']);fields['validation'].append(np.full(n,int(trace['seed'])%5==0))
        sources.append({'path':str(path),'sha256':digest_file(path),'seed':int(trace['seed']),'steps':n})
    assert sources,'No own-skill demonstrations'
    return {k:torch.as_tensor(np.concatenate(v),device=device) for k,v in fields.items()},sources


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=15000);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--control-augmentation',type=float,default=.2)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh private-controller output required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    actor,initial=initialize_controller(args.skill);frozen=actor.frozen_digest()
    data,sources=dataset(args.data,args.skill,'cuda')
    optimizer=torch.optim.Adam(actor.trainable(),lr=.0003,betas=(.5,.99))
    groups=[torch.nonzero(~data['validation']&(data['phase']==i)).flatten() for i in range(8)]
    groups=[g for g in groups if len(g)];validation=torch.nonzero(data['validation']).flatten()
    assert len(validation)
    validation=validation[torch.linspace(0,len(validation)-1,min(4096,len(validation)),device='cuda').long()]
    config={'args':vars(args),'initial_own_adapter':{'path':str(initial),'sha256':digest_file(initial)},
            'sources':sources,'scope':'Only own-skill native demonstrations. This action adapter is private to the named skill; no common actor checkpoint, joint data, joint loss or shared new learned parameters.',
            'loss':'Masked action reconstruction + first-control MAE/BCE. Generic own-state local target perturbations supervise only first XYZ/gripper controls during training.',
            'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    from cube_addons.losses import reconstruction
    def forward(ids):
        return actor(data['states'][ids].float(),data['base_z'][ids].float(),
                     torch.full((len(ids),1),.8,device='cuda'),data['contexts'][ids].float(),return_logits=True)
    best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            actor.train();per=(args.batch+len(groups)-1)//len(groups)
            ids=torch.cat([g[torch.randint(len(g),(per,),device='cuda')] for g in groups])[:args.batch]
            pred,logits=forward(ids);target=data['targets'][ids].float();mask=data['mask'][ids].float()
            first=F.l1_loss(pred[:,0,:4],target[:,0,:4])+.3*F.binary_cross_entropy_with_logits(2*logits[:,0,4],(target[:,0,4]+1)/2)
            loss=reconstruction(pred,target,mask,logits)+first
            carry=ids[data['phase'][ids]==4]
            if args.control_augmentation and len(carry):
                ctx=data['contexts'][carry].float().clone();bz=data['base_z'][carry].float()
                ctx[:,20:23]+=args.control_augmentation*torch.randn(len(carry),3,device='cuda')
                cp,cl=actor(data['states'][carry].float(),bz,torch.full((len(carry),1),.8,device='cuda'),ctx,return_logits=True)
                probability=.5*(1+torch.erf(bz[:,1]/2**.5))
                speed=torch.where(probability<.5,.65+.4*probability,.7+.3*probability)
                limit=torch.minimum(speed,torch.full_like(speed,.65))[:,None]
                label=(4*speed[:,None]*ctx[:,20:23]).clamp(-limit,limit)
                loss+=2*F.mse_loss(cp[:,0,:3],label)+.1*cp[:,0,3].square().mean()+.1*F.binary_cross_entropy_with_logits(2*cl[:,0,4],torch.ones(len(carry),device='cuda'))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.trainable(),5);optimizer.step()
            if step%500==0 or step==args.steps:
                actor.eval()
                with torch.no_grad():
                    vp,vl=forward(validation);vt=data['targets'][validation].float();vm=data['mask'][validation].float()
                    value=reconstruction(vp,vt,vm,vl)+F.l1_loss(vp[:,0,:4],vt[:,0,:4])+.3*F.binary_cross_entropy_with_logits(2*vl[:,0,4],(vt[:,0,4]+1)/2)
                    metric={'step':step,'loss':float(loss.detach()),'validation_loss':float(value),
                            'validation_first_action_rmse':float((vp[:,0,:4]-vt[:,0,:4]).square().mean().sqrt()),
                            'seconds':time.time()-started}
                assert actor.frozen_digest()==frozen
                log.write(json.dumps(metric)+'\n');print(json.dumps(metric),flush=True)
                ck={'addon':actor.addon_state(),'adapter_config':asdict(actor.config),'base_sha256':digest_file('cube/artifacts/final/base_policy.pt'),
                    'skill':args.skill,'step':step,'metrics':metric,'config':config,'frozen_digest':frozen}
                torch.save(ck,out/'latest.pt')
                if float(value)<best:best=float(value);torch.save(ck,out/'best.pt')
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_own_validation_loss':best,'frozen_base_and_encoder':True})+'\n')


if __name__=='__main__':main()
