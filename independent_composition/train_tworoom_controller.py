"""Train one private neural path follower around the frozen TwoRoom base.

Own-skill observed states supply the context distribution. Nearby target labels
are a training-only kinematic teacher. Inference executes the neural adapter.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .tworoom_paths import initialize_controller,path_context,canonical
from .legacy import digest_file


class Data:
    def __init__(self,path,device):
        with np.load(path) as d:
            arrays={k:d[k].copy() for k in ['states','contexts','base_z','lengths']}
        self.fingerprints=set();paths=[];indices=np.zeros(arrays['states'].shape[:2],np.int64)
        for i,length in enumerate(arrays['lengths']):
            state=arrays['states'][i];sign=1. if state[0,2]>=0 else -1.
            position=(state[:length+1,:2]+1)*112;position[:,0]=112+sign*(position[:,0]-112)
            goal=(state[0,2:4]+1)*112;goal[0]=112+sign*(goal[0]-112)
            full=np.concatenate([position,goal[None]],0)
            cumulative=np.r_[0,np.linalg.norm(np.diff(full,axis=0),axis=1).cumsum()]
            keep=np.r_[True,np.diff(cumulative)>1e-6]
            path=np.stack([np.interp(np.linspace(0,cumulative[-1],65),cumulative[keep],full[keep,j]) for j in range(2)],-1)
            paths.append(path.astype(np.float32))
            indices[i,:length+1]=np.maximum(1,np.ceil(cumulative[:length+1]/cumulative[-1]*64)).astype(np.int64).clip(0,64)
            import hashlib
            key=np.r_[state[0],arrays['contexts'][i,0],arrays['base_z'][i]].astype(np.float32)
            self.fingerprints.add(hashlib.sha256(key.tobytes()).hexdigest())
        arrays['paths']=np.stack(paths);arrays['indices']=indices
        self.data={k:torch.as_tensor(v,device=device) for k,v in arrays.items()}
        self.size=len(paths);self.device=device

    def sample(self,n,rng=None):
        d=self.data;ids=torch.randint(self.size,(n,),device=self.device,generator=rng)
        t=(torch.rand(n,device=self.device,generator=rng)*d['lengths'][ids]).long()
        state=d['states'][ids,t];own=d['contexts'][ids,t,:12];bz=d['base_z'][ids]
        canon,reflection=canonical(state);position=(canon[:,:2]+1)*112
        index=d['indices'][ids,t]
        context=path_context(own,position,d['paths'][ids],index,torch.zeros(n,device=self.device))
        counterfactual=torch.rand(n,device=self.device,generator=rng)<.5
        delta=30*(2*torch.rand(n,2,device=self.device,generator=rng)-1)
        context[:,12:14]=torch.where(counterfactual[:,None],delta/56,context[:,12:14])
        delta=context[:,12:14]*56;distance=delta.norm(dim=-1)
        from tworoom_gan.style import decode_style
        _,speed,_=decode_style(bz)
        target=delta/distance.clamp_min(1e-6)[:,None]*torch.minimum(speed,distance/5)[:,None]
        target=target*torch.stack([reflection,torch.ones_like(reflection)],-1)
        return state,bz,context,target


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh private follower directory required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    train=Data(args.train,'cuda');valid=Data(args.validation,'cuda')
    assert not train.fingerprints&valid.fingerprints
    actor=initialize_controller();frozen=actor.frozen_digest()
    optimizer=torch.optim.Adam(actor.trainable(),lr=.0005)
    rng=torch.Generator(device='cuda').manual_seed(8142901);validation=valid.sample(4096,rng)
    config={'args':vars(args),'train_sha256':digest_file(args.train),'validation_sha256':digest_file(args.validation),
            'train_contexts':len(train.fingerprints),'validation_contexts':len(valid.fingerprints),
            'scope':'One own-skill state corpus. Independent newly initialized LoRA/FiLM path follower, original base/encoder frozen. First-control supervised tracking only; no joint examples or shared new learned controller.',
            'loss':'First-control MSE + .1 MAE, using real route targets and independent nearby-target counterfactuals. Analytic target labels are used only in training.',
            'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            actor.train();s,bz,ctx,target=train.sample(args.batch)
            prediction=actor(s,bz,torch.full((len(s),1),.8,device='cuda'),ctx)[:,0]
            loss=(prediction-target).square().mean()+.1*(prediction-target).abs().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.trainable(),5);optimizer.step()
            if step%500==0 or step==args.steps:
                actor.eval()
                with torch.no_grad():
                    vs,vz,vc,vt=validation;vp=actor(vs,vz,torch.full((len(vs),1),.8,device='cuda'),vc)[:,0]
                    error=(vp-vt).square().mean().sqrt();value=(vp-vt).square().mean()+.1*(vp-vt).abs().mean()
                assert actor.frozen_digest()==frozen
                row={'step':step,'loss':float(loss.detach()),'validation_action_rmse':float(error),
                     'validation_loss':float(value),'seconds':time.time()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                ck={'addon':actor.addon_state(),'adapter_config':asdict(actor.config),'base_sha256':digest_file('artifacts/final/base_policy.pt'),
                    'skill':args.skill,'step':step,'metrics':row,'config':config}
                torch.save(ck,out/'latest.pt')
                if float(value)<best:best=float(value);torch.save(ck,out/'best.pt')
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_validation_loss':best,'frozen_base_and_encoder':True})+'\n')


if __name__=='__main__':main()
