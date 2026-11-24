"""Separately trained joint-demo behavioral cloning reference; never zero-shot."""
import argparse,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from tworoom_gan.style import canonical_state
from .runtime import observation
from .geometry import contexts,entry_fraction
from .models import FrozenBank
from .search import SearchConfig
from .rollout import rollout
from .evaluate import load_cases,summary


class JointPolicy(nn.Module):
    def __init__(self):
        super().__init__();layers=[nn.Linear(36,256),nn.LeakyReLU(.2)]
        for _ in range(3):layers.extend([nn.Linear(256,256),nn.LayerNorm(256),nn.LeakyReLU(.2)])
        layers.append(nn.Linear(256,32));self.net=nn.Sequential(*layers)

    def features(self,pos,case,visited,clock,previous=None):
        state,reflection=canonical_state(observation(pos,case['goal']))
        ctx=contexts(pos,case,visited,clock,previous)
        return torch.cat([state,case['base_z'],*ctx],-1),reflection

    def forward(self,pos,case,visited,clock,previous=None):
        x,reflection=self.features(pos,case,visited,clock,previous)
        return self.net(x).tanh().reshape(*x.shape[:-1],16,2)*torch.stack([reflection,torch.ones_like(reflection)],-1)[...,None,:]


def training_tokens(model,paths,device='cuda'):
    features=[];targets=[];masks=[]
    for path in paths:
        data=np.load(path);proof=np.load(Path(path).with_name(Path(path).stem+'_witnesses.npz'))
        for lo in range(0,len(data['start']),32):
            c={k:torch.as_tensor(data[k][lo:lo+32],device=device) for k in ['start','goal','base_z','keepout','waypoint']}
            p=torch.as_tensor(proof['positions'][lo:lo+32],device=device);a=torch.as_tensor(proof['actions'][lo:lo+32],device=device)
            lengths=torch.as_tensor(proof['lengths'][lo:lo+32],device=device);b=len(p)
            visit=torch.isfinite(entry_fraction(p[:,:-1],p[:,1:],c['waypoint'][:,None,:2],c['waypoint'][:,None,2]))
            before=torch.cat([visit.new_zeros((b,1)),visit[:,:-1].long().cumsum(1)>0],1)
            cc={k:v[:,None].expand(-1,240,*v.shape[1:]) for k,v in c.items()};clock=torch.arange(240,device=device)[None].expand(b,-1)
            valid=clock<lengths[:,None];x,reflection=model.features(p[:,:-1],cc,before,clock)
            indices=torch.arange(240,device=device)[:,None]+torch.arange(16,device=device)[None]
            chunks=a[:,indices.clamp_max(239)]*torch.stack([reflection,torch.ones_like(reflection)],-1)[:,:,None,:]
            mask=(indices[None]<lengths[:,None,None]).float()
            features.append(x[valid]);targets.append(chunks[valid]);masks.append(mask[valid])
    return torch.cat(features),torch.cat(targets),torch.cat(masks)


class JointPlanner:
    def __init__(self,model,bank):self.model=model;self.bank=bank
    def plan(self,pos,case,visited,clock,previous):
        torch.cuda.synchronize();start=time.perf_counter();self.bank.reset_counts()
        with torch.inference_mode():
            chunk=self.model(pos,case,visited,clock,previous);torch.cuda.synchronize();policy_seconds=time.perf_counter()-start
            z=pos.new_zeros((len(pos),2));ctx=contexts(pos,case,visited,clock,previous,self.bank.dimensions)
            a,b=self.bank.pair(observation(pos,case['goal']),case['base_z'],z,ctx);d=(a-b).square().mean((1,2));zero=pos.new_zeros(len(pos))
        torch.cuda.synchronize();self.bank.count('joint_reference',len(pos))
        return chunk,z,{'loss':zero,'disagreement':d,'prior':zero,'task_cost':zero,'tail_loss':zero,
            'seconds':time.perf_counter()-start,'policy_only_seconds':policy_seconds,'iterations':0,'backward_steps':0,
            'calls':{k:v.copy() for k,v in self.bank.calls.items()}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',nargs='+',required=True);p.add_argument('--val',nargs='+',required=True)
    p.add_argument('--test',nargs='+',required=True);p.add_argument('--out',default='composition_artifacts/joint_reference')
    p.add_argument('--seed',type=int,default=47);p.add_argument('--steps',type=int,default=3000);args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True);model=JointPolicy().cuda();bank=FrozenBank()
    from .dataset_audit import fingerprints
    groups=[]
    for paths in [args.train,args.val,args.test]:
        ids=set()
        for path in paths:
            with np.load(path) as d:ids|=fingerprints(d)
        groups.append(ids)
    assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    with torch.no_grad():train=training_tokens(model,args.train);val=training_tokens(model,args.val)
    (root/'config.json').write_text(json.dumps({'args':vars(args),'training_type':'Joint demonstration behavioral cloning; not zero-shot and not a strict upper bound.',
        'train_samples':len(train[0]),'val_samples':len(val[0]),'architecture':'36 inputs, 4x256 hidden, LayerNorm, 16x2 tanh outputs; canonical reflection',
        'train_val_test_exact_context_overlap':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'train_data_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.train},
        'frozen_files':bank.files},indent=2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5);history=[];best=float('inf');start=time.perf_counter()
    weights=torch.cat([torch.full((8,),2.,device='cuda'),torch.ones(8,device='cuda')])
    for step in range(1,args.steps+1):
        idx=torch.randint(len(train[0]),(512,),device='cuda');x,y,mask=[v[idx] for v in train]
        pred=model.net(x).tanh().reshape(-1,16,2);loss=(((pred-y).square().mean(-1)*mask*weights).sum()/(mask*weights).sum())
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%250==0 or step==args.steps:
            with torch.no_grad():
                errors=[]
                for lo in range(0,len(val[0]),4096):
                    x,y,m=[v[lo:lo+4096] for v in val];prediction=model.net(x).tanh().reshape(-1,16,2)
                    errors.append((((prediction-y).square().mean(-1)*m).sum(),m.sum()))
                validation=float(sum(x[0] for x in errors)/sum(x[1] for x in errors))
            history.append({'step':step,'train_loss':float(loss),'val_action_mse':validation,'seconds':time.perf_counter()-start})
            if validation<best:
                best=validation;torch.save({'model':model.state_dict(),'step':step,'args':vars(args),'validation_mse':validation},root/'best.pt')
            (root/'training.json').write_text(json.dumps(history,indent=2));print('TRAIN',history[-1],flush=True)
    model.load_state_dict(torch.load(root/'best.pt',weights_only=False)['model']);model.eval().requires_grad_(False)
    reference_hash=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest()
    for path in args.test:
        case=load_cases(path,'cuda');cfg=SearchConfig(method='fixed',family='sum',temporal='fixed');planner=JointPlanner(model,bank)
        rows,traj,info=rollout(bank,case,cfg,planner_override=planner)
        with np.load(path) as d:
            for row,name in zip(rows,d['kind']):row['kind']=str(name)
        target=root/'evaluation'/Path(path).stem/'joint_demo_bc';target.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target/'trajectories.npz',**traj,**{k:v.cpu().numpy() for k,v in case.items()})
        search=asdict(cfg);search['method']='joint_demo_bc';(target/'config.json').write_text(json.dumps({'args':vars(args),'search':search,'frozen_files':bank.files,
            'reference_sha256':reference_hash,'agreement_note':'Independent skill disagreement at zero modifier codes on the reference path; not the joint policy outputs.'},indent=2))
        (target/'records.json').write_text(json.dumps(rows,indent=2));(target/'planning.json').write_text(json.dumps(info,indent=2))
        measured=summary(rows,info);measured['policy_only_planning_batch_seconds_mean']=float(np.mean([m['policy_only_seconds'] for m in info['planning']]))
        (target/'summary.json').write_text(json.dumps(measured,indent=2));(target/'DONE.json').write_text(json.dumps({'complete':True,'not_zero_shot':True}));print('EVAL',path,measured['success'],flush=True)
    bank.assert_frozen();(root/'DONE.json').write_text(json.dumps({'complete':True,'trained_reference_only':True,'best_sha256':reference_hash}))


if __name__=='__main__':main()
