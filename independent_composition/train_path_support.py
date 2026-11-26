"""Train a private route-domain model from own-only success/failure labels.

Exact swept geometry labels are computed only in this training program. The
deployed architecture consumes poses/context tensors and predicts a score.
No joint scene, other skill region, or shared learned model is used in training.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from .path_support import PathSupport
from .path_policy import load_route
from .tworoom_paths import proposal
from .train_tworoom_routes import Data
from .legacy import digest_file


@torch.no_grad()
def own_labels(path,condition,skill):
    from tworoom_addons.geometry import segment_distance
    from tworoom_composition.geometry import wall_clearance,entry_fraction
    pixels=(path+1)*112;a,b=pixels[:,:-1],pixels[:,1:]
    center=(condition[:,10:12]+1)*112;radius=condition[:,12]*40
    wall=wall_clearance(a,b).amin(1)
    distance=segment_distance(a,b,center[:,None])
    if skill=='keepout':margin=torch.minimum(wall,(distance-radius[:,None]-7).amin(1))
    else:
        goal=(condition[:,2:4]+1)*112
        wp=entry_fraction(a,b,center[:,None],radius[:,None])
        gg=entry_fraction(a,b,goal[:,None],torch.full_like(wp,16.))
        length=(b-a).norm(dim=-1);start=torch.cat([length.new_zeros(len(length),1),length.cumsum(1)[:,:-1]],1)
        infinity=torch.full_like(wp,float('inf'))
        first_wp=torch.where(torch.isfinite(wp),start+wp.nan_to_num(posinf=0.)*length,infinity).amin(1)
        first_goal=torch.where(torch.isfinite(gg),start+gg.nan_to_num(posinf=0.)*length,infinity).amin(1)
        order=(first_goal-first_wp).nan_to_num(nan=-50.,neginf=-50.,posinf=50.).clamp(-50,50)
        margin=torch.minimum(torch.minimum(wall,radius-distance.amin(1)),order)
    return margin>=-1e-5,margin.clamp(-50,50)


@torch.no_grad()
def examples(data,route,skill,n,rng):
    z,c,teacher=data.sample(n,rng)
    code=torch.randn(z.shape,device='cuda',generator=rng).clamp(-3,3)
    code[:n//2]=6*torch.rand((n//2,10),device='cuda',generator=rng)-3
    raw=proposal(code,c);generated=route(code,c,raw)
    knots=torch.linspace(0,1,7,device='cuda');t=torch.linspace(0,1,65,device='cuda')
    weights=(1-(t[:,None]-knots[None]).abs()*6).clamp_min(0)
    offsets=torch.randn((n,7,2),device='cuda',generator=rng)
    sigma=.005+.15*torch.rand((n,1,1),device='cuda',generator=rng)
    perturb=torch.einsum('pa,bad->bpd',weights,offsets)*sigma*torch.sin(torch.pi*t)[None,:,None]
    alpha=torch.rand((n,1,1),device='cuda',generator=rng)
    choices=torch.stack([teacher,generated,teacher+perturb,raw,alpha*teacher+(1-alpha)*generated],1)
    ids=torch.randint(5,(n,),device='cuda',generator=rng)
    path=choices[torch.arange(n,device='cuda'),ids]
    # Generic time warping prevents the score relying on one visit phase.
    power=.6+1.2*torch.rand((n,1),device='cuda',generator=rng)
    index=64*t[None].pow(power);lo=index.floor().long().clamp_max(64);hi=(lo+1).clamp_max(64)
    fraction=(index-lo)[:,:,None]
    path=path.gather(1,lo[:,:,None].expand(-1,-1,2))*(1-fraction)+path.gather(1,hi[:,:,None].expand(-1,-1,2))*fraction
    path[:,0]=c[:,:2];path[:,-1]=c[:,2:4]
    # Off-policy own-region randomization supplies matched paths with varied
    # own labels. It never introduces a second skill or a joint scene.
    changed=torch.rand(n,device='cuda',generator=rng)<.5
    center=28+168*torch.rand((n,2),device='cuda',generator=rng)
    radius=(10+4*torch.rand(n,device='cuda',generator=rng) if skill=='keepout'
            else 7+3*torch.rand(n,device='cuda',generator=rng))
    modified=c.clone();delta=(c[:,:2]+1)*112-center;distance=delta.norm(dim=-1)
    modified[:,10:12]=center/112-1;modified[:,12]=radius/40
    modified[:,15:17]=delta/56;modified[:,17]=(distance-radius-(7 if skill=='keepout' else 0))/56
    modified[:,20:22]=delta/distance.clamp_min(1e-6)[:,None]
    c=torch.where(changed[:,None],modified,c)
    valid,margin=own_labels(path,c,skill)
    return path,c,valid,margin


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--route',required=True);p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--seed',type=int,default=42);args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh support-model training directory required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    route,meta=load_route(args.route);assert meta['skill']==args.skill
    train,valid=Data(args.train),Data(args.validation)
    assert not set(map(bytes,train.context.cpu().numpy())).intersection(map(bytes,valid.context.cpu().numpy()))
    config={'context_dim':25,'coordinates':2,'hidden':128,'points':65};model=PathSupport(**config).cuda()
    model.context_mean.copy_(route.context_mean);model.context_std.copy_(route.context_std)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-6)
    rng=torch.Generator(device='cuda').manual_seed(args.seed+4131)
    vrng=torch.Generator(device='cuda').manual_seed(973102)
    vp,vc,vy,vm=examples(valid,route,args.skill,4096,vrng)
    provenance={'args':vars(args),'route_sha256':digest_file(args.route),
        'train_sha256':digest_file(args.train),'validation_sha256':digest_file(args.validation),
        'scope':__doc__,'architecture':'Generic private segment MLP, two temporal convolutions, mean/max pooling, and a two-output head. Identical architecture for both skill roles.',
        'loss':'Class-balanced own-validity BCE + .5 signed-own-margin SmoothL1, with5xnear-boundaryweights. Own teacher/generated/corrupted/interpolated/free-curve routes; generic temporal warping and randomized own-region contexts. Other skill is absent.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(provenance,indent=2)+'\n');best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            path,c,label,margin=examples(train,route,args.skill,512,rng)
            pred=model(path,c);positive=label.float();fraction=positive.mean().clamp(.05,.95)
            balanced=torch.where(label,.5/fraction,.5/(1-fraction))
            boundary=1+4*torch.exp(-margin.abs()/3)
            bce=torch.nn.functional.binary_cross_entropy_with_logits(pred[:,0],positive,reduction='none')
            regression=torch.nn.functional.smooth_l1_loss(pred[:,1],margin/10,reduction='none')
            loss=((bce+.5*regression)*balanced*boundary).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if step%1000==0 or step==args.steps:
                model.eval()
                with torch.no_grad():
                    value=model(vp,vc);guess=value[:,0]>=0
                    fp=float(guess[~vy].float().mean());fn=float((~guess[vy]).float().mean())
                    error=float((10*value[:,1]-vm).square().mean().sqrt());score=2*fp+fn+.001*error
                    row={'step':step,'loss':float(loss.detach()),'validation_accuracy':float((guess==vy).float().mean()),
                         'validation_false_positive_rate':fp,'validation_false_negative_rate':fn,
                         'validation_margin_rmse_px':error,'validation_positive_fraction':float(vy.float().mean()),
                         'seconds':time.time()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                saved={'state':model.state_dict(),'model_config':config,'skill':args.skill,'step':step,
                       'metrics':row,'config':provenance}
                torch.save(saved,out/'latest.pt')
                if score<best:best=score;torch.save(saved,out/'best.pt')
                model.train()
    assert all(p.grad is None and not p.requires_grad for p in route.parameters())
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_validation_score':best,'route_frozen':True})+'\n')


if __name__=='__main__':main()
