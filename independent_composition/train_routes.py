"""Fit a generic private trajectory network from own-skill path examples.

Analytic own-skill teachers/labels are confined to this training program. The
deployed decoder has the same generic architecture for either skill and never
calls these functions or a simulator.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time
import torch
from .private_paths import RouteConfig,RouteDecoder,spline_proposal,own_geometry
from .legacy import digest_file


def gap(path,own,skill):
    center=torch.cat([own[:,2:4],.2+own[:,4:5]],-1)
    a,b=path[:,:-1],path[:,1:];delta=b-a
    fraction=((center[:,None]-a)*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-12)
    distance=(a+fraction.clamp(0,1)[:,:,None]*delta-center[:,None]).norm(dim=-1).amin(1)
    radius=own[:,5]+(math.sqrt(3)*.02 if skill=='keepout' else 0.)
    return distance-radius


@torch.no_grad()
def sample(skill,n,device,rng):
    from cube_agreement.neural import sample_training
    from cube_agreement.manifolds import oracle
    z,g=sample_training(skill,n,device,rng)
    other=g[:,6:10] if skill=='keepout' else g[:,2:6]
    assert not bool(torch.count_nonzero(other))
    own=own_geometry(g,skill);transverse=oracle(skill,z,g)
    t=torch.linspace(0,1,65,device=device)
    target=torch.cat([(t[None]*own[:,0,None])[:,:,None],transverse[:,:,:1],.2+transverse[:,:,1:]],-1)
    return z,own,spline_proposal(z,own),target


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--batch',type=int,default=256)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh private route directory required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    rng=torch.Generator(device='cuda').manual_seed(args.seed)
    vrng=torch.Generator(device='cuda').manual_seed(7921801)
    vz,vc,vp,vt=sample(args.skill,4096,'cuda',vrng)
    config=RouteConfig(vz.shape[-1]);model=RouteDecoder(config).cuda()
    _,norm,_,_=sample(args.skill,4096,'cuda',rng);model.fit_normalization(norm)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=1e-6)
    provenance={'args':vars(args),'model_config':asdict(config),
        'scope':'Fresh own-skill contexts and path codes only. Other skill geometry is absent. The controller for this skill is separately trained on own native trajectories. No newly shared learned weights.',
        'architecture':'Generic residual MLP over a fixed free-curve prior; no keepout radial push, mandatory waypoint insertion or analytic projector inside the deployed network.',
        'loss':'Normalized trajectory MSE + .2 worst-point MSE + 2 own-skill margin penalty, used only in training.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')},
        'teacher_sha256':{str(f):digest_file(f) for f in [Path('cube_agreement/neural.py'),Path('cube_agreement/manifolds.py')]}}
    (out/'config.json').write_text(json.dumps(provenance,indent=2)+'\n')
    best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            z,context,proposal,target=sample(args.skill,args.batch,'cuda',rng)
            predicted=model(z,context,proposal);error=(predicted-target)/model.coordinate_scale
            distance=gap(predicted,context,args.skill)
            penalty=(torch.relu(.003-distance)/.1).square().mean() if args.skill=='keepout' else (torch.relu(distance+.002)/.1).square().mean()
            loss=error.square().mean()+.2*error.square().flatten(1).amax(1).mean()+2*penalty
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if step%500==0 or step==args.steps:
                with torch.no_grad():
                    value=model(vz,vc,vp);vgap=gap(value,vc,args.skill)
                    valid=vgap>=-1e-7 if args.skill=='keepout' else vgap<=0
                    rmse=float((value-vt).square().mean().sqrt())
                    score=1-float(valid.float().mean())+rmse
                    metric={'step':step,'loss':float(loss.detach()),'validation_path_rmse_m':rmse,
                            'validation_own_geometry_success':float(valid.float().mean()),
                            'worst_validation_gap_m':float(vgap.min() if args.skill=='keepout' else vgap.max()),
                            'seconds':time.time()-started}
                log.write(json.dumps(metric)+'\n');print(json.dumps(metric),flush=True)
                checkpoint={'state':model.state_dict(),'model_config':asdict(config),'skill':args.skill,
                            'step':step,'metrics':metric,'config':provenance}
                torch.save(checkpoint,out/'latest.pt')
                if score<best:best=score;torch.save(checkpoint,out/'best.pt')
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_own_validation_score':best})+'\n')


if __name__=='__main__':main()
