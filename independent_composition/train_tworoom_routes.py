"""Private full-route models, trained with own-skill labels and margins only.

The geometric functions below are training/evaluation diagnostics. The deployed
route decoder is the same generic MLP for either skill and never calls them.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .tworoom_paths import make_route,proposal
from .legacy import digest_file


def distances(path,center):
    a,b=path[:,:-1],path[:,1:];delta=b-a
    u=((center[:,None]-a)*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-8)
    return (a+u.clamp(0,1)[:,:,None]*delta-center[:,None]).norm(dim=-1)


def margins(predicted,condition,skill):
    """Differentiable own-only safety/order labels; not used at inference."""
    path=(predicted+1)*112;center=(condition[:,10:12]+1)*112;radius=condition[:,12]*40
    goal=(condition[:,2:4]+1)*112
    # Dense samples prevent most corner cutting between the 65 knots.
    a,b=path[:,:-1],path[:,1:]
    t=path.new_tensor([0.,.25,.5,.75,1.])
    dense=(a[:,:,None]*(1-t[None,None,:,None])+b[:,:,None]*t[None,None,:,None]).flatten(1,2)
    boxes=path.new_tensor([[107,117,14,35],[107,117,77,147],[107,117,189,210]])
    boxcenter=torch.stack([(boxes[:,0]+boxes[:,1])/2,(boxes[:,2]+boxes[:,3])/2],-1)
    half=torch.stack([(boxes[:,1]-boxes[:,0])/2,(boxes[:,3]-boxes[:,2])/2],-1)
    q=(dense[:,:,None]-boxcenter).abs()-half
    sdf=torch.relu(q).norm(dim=-1)+q.amax(-1).clamp_max(0)-7
    wallgap=sdf.amin((1,2));boundgap=torch.minimum(dense-21,203-dense).amin((1,2))
    wallpen=torch.relu(1.5-sdf).square().mean((1,2))/100
    boundpen=torch.relu(1.5-torch.minimum(dense-21,203-dense)).square().mean((1,2))/100
    d=distances(path,center);goald=distances(path,goal)
    if skill=='keepout':
        own_gap=d.amin(1)-radius-7
        ownpen=torch.relu(1.5-(d-radius[:,None]-7)).square().mean(1)/100
        orderpen=torch.zeros_like(ownpen)
        ownok=own_gap>=0
    else:
        own_gap=radius-d.amin(1)
        ownpen=torch.relu(1.5-own_gap).square()/100
        # A soft prefix visit indicator penalizes premature goal entry using
        # only this skill's own region and the base goal.
        visited=torch.sigmoid((radius[:,None]-d)/1.5).cummax(1).values
        orderpen=((1-visited)*torch.relu(17.5-goald).square()/100).mean(1)
        visit_index=(d<=radius[:,None]).to(torch.int64).argmax(1)
        goal_index=(goald<=16).to(torch.int64).argmax(1)
        ownok=(own_gap>=0)&(visit_index<=goal_index)
    penalty=wallpen+boundpen+ownpen+orderpen
    valid=(wallgap>=0)&(boundgap>=0)&ownok
    return penalty,valid,{'wall_gap':wallgap,'own_gap':own_gap}


class Data:
    def __init__(self,path):
        with np.load(path) as d:
            self.context=torch.as_tensor(d['condition'],device='cuda')
            self.code=torch.as_tensor(d['code'],device='cuda')
            self.path=torch.as_tensor(d['path'],device='cuda')
        self.size=len(self.code)

    def sample(self,n,rng):
        ids=torch.randint(self.size,(n,),device='cuda',generator=rng)
        return self.code[ids],self.context[ids],self.path[ids]


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=30000);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh route training directory required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    train,valid=Data(args.train),Data(args.validation)
    assert not set(map(bytes,train.context.cpu().numpy())).intersection(map(bytes,valid.context.cpu().numpy()))
    model=make_route().cuda();model.fit_normalization(train.context)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=1e-6)
    rng=torch.Generator(device='cuda').manual_seed(args.seed)
    vrng=torch.Generator(device='cuda').manual_seed(982001)
    vz,vc,vt=valid.sample(min(2048,valid.size),vrng)
    config={'args':vars(args),'model_config':asdict(model.config),
        'train_sha256':digest_file(args.train),'validation_sha256':digest_file(args.validation),
        'scope':'Independent own-skill trajectories with known Gaussian code10. No other skill region, joint demonstrations, shared new learned module, or final test access.',
        'architecture':'Private 3x512 SiLU residual trajectory MLP over fixed interpolation of five free XY control points. Exact endpoints. No geometric repair or own-skill predicate in the deployed forward function.',
        'loss':'Path MSE + .2 worst-point MSE + .05 own geometry penalty; additional .025 own geometry penalty on fresh Gaussian prior codes. Geometry includes base walls/bounds and only this adapter constraint, with goal ordering for waypoint.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            z,c,target=train.sample(args.batch,rng);pred=model(z,c,proposal(z,c))
            error=(pred-target).square();penalty,_,_=margins(pred,c,args.skill)
            fresh=torch.randn(z.shape,device='cuda',generator=rng).clamp(-3,3)
            prior=model(fresh,c,proposal(fresh,c));priorpen,_,_=margins(prior,c,args.skill)
            loss=error.mean()+.2*error.flatten(1).amax(1).mean()+.05*penalty.mean()+.025*priorpen.mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if step%1000==0 or step==args.steps:
                with torch.no_grad():
                    v=model(vz,vc,proposal(vz,vc));vp,ok,gaps=margins(v,vc,args.skill)
                    rmse=112*(v-vt).square().mean().sqrt();success=ok.float().mean()
                    score=1-float(success)+float(rmse)/112
                    row={'step':step,'loss':float(loss.detach()),'validation_path_rmse_px':float(rmse),
                         'validation_own_route_success':float(success),'validation_margin_penalty':float(vp.mean()),
                         'seconds':time.time()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                ck={'state':model.state_dict(),'model_config':asdict(model.config),'skill':args.skill,'step':step,'metrics':row,'config':config}
                torch.save(ck,out/'latest.pt')
                if score<best:best=score;torch.save(ck,out/'best.pt')
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_own_validation_score':best})+'\n')


if __name__=='__main__':main()
