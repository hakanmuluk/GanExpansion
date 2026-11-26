"""Own-only domain-model finetuning with continuous missed-waypoint margins.

Binary success labels stay identical to v11. If a path misses its waypoint,
the distance deficit supplies the regression target instead of an undefined
entry time becoming a -50px target. This also emphasizes near-boundary misses.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import torch
from .path_support import PathSupport
from .path_policy import load_route
from .train_tworoom_routes import Data
from .train_path_support import examples
from .legacy import digest_file


@torch.no_grad()
def continuous_margin(path,c,skill,margin):
    if skill=='waypoint':
        from tworoom_addons.geometry import segment_distance
        from tworoom_composition.geometry import wall_clearance,entry_fraction
        p=(path+1)*112;a,b=p[:,:-1],p[:,1:]
        center=(c[:,10:12]+1)*112;radius=c[:,12]*40;goal=(c[:,2:4]+1)*112
        own=radius-segment_distance(a,b,center[:,None]).amin(1)
        wall=wall_clearance(a,b).amin(1)
        wp=entry_fraction(a,b,center[:,None],radius[:,None])
        gg=entry_fraction(a,b,goal[:,None],torch.full_like(wp,16.))
        length=(b-a).norm(dim=-1);start=torch.cat([length.new_zeros(len(length),1),length.cumsum(1)[:,:-1]],1)
        infinity=torch.full_like(wp,float('inf'))
        wparc=torch.where(torch.isfinite(wp),start+wp.nan_to_num(posinf=0.)*length,infinity).amin(1)
        garc=torch.where(torch.isfinite(gg),start+gg.nan_to_num(posinf=0.)*length,infinity).amin(1)
        order=torch.where(torch.isfinite(wparc),garc-wparc,torch.full_like(wparc,50.))
        order=order.nan_to_num(nan=-50.,neginf=-50.,posinf=50.).clamp(-50,50)
        margin=torch.minimum(torch.minimum(wall,own),order).clamp(-50,50)
    return margin


@torch.no_grad()
def continuous_examples(data,route,skill,n,rng):
    path,c,label,margin=examples(data,route,skill,n,rng)
    return path,c,label,continuous_margin(path,c,skill,margin)


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--initial',required=True);p.add_argument('--route',required=True)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=15000);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh domain finetuning directory required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    initial=Path(args.initial).read_bytes();ck=torch.load(io.BytesIO(initial),map_location='cuda',weights_only=False)
    assert ck['skill']==args.skill and ck['config']['route_sha256']==digest_file(args.route)
    (out/'initial.pt').write_bytes(initial)
    model=PathSupport(**ck['model_config']).cuda();model.load_state_dict(ck['state'])
    route,meta=load_route(args.route);assert meta['skill']==args.skill
    train,valid=Data(args.train),Data(args.validation)
    assert not set(map(bytes,train.context.cpu().numpy())).intersection(map(bytes,valid.context.cpu().numpy()))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=1e-6)
    rng=torch.Generator(device='cuda').manual_seed(args.seed+9131)
    vrng=torch.Generator(device='cuda').manual_seed(973102)
    vp,vc,vy,vm=continuous_examples(valid,route,args.skill,4096,vrng)
    provenance={'args':vars(args),'initial_step':ck['step'],'initial_sha256':hashlib.sha256(initial).hexdigest(),
        'route_sha256':digest_file(args.route),'train_sha256':digest_file(args.train),
        'validation_sha256':digest_file(args.validation),'scope':__doc__,
        'loss':'Same class-balanced validity BCE + .5 signed-margin SmoothL1 with near-boundary weighting; corrected continuous distance deficit when waypoint is missed. Only own paths, randomized own regions and own labels.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(provenance,indent=2)+'\n');best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            path,c,label,margin=continuous_examples(train,route,args.skill,512,rng)
            pred=model(path,c);positive=label.float();fraction=positive.mean().clamp(.05,.95)
            balanced=torch.where(label,.5/fraction,.5/(1-fraction));boundary=1+4*torch.exp(-margin.abs()/3)
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
                saved={'state':model.state_dict(),'model_config':ck['model_config'],'skill':args.skill,'step':step,
                       'metrics':row,'config':provenance}
                torch.save(saved,out/'latest.pt')
                if score<best:best=score;torch.save(saved,out/'best.pt')
                model.train()
    assert all(p.grad is None and not p.requires_grad for p in route.parameters())
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_validation_score':best,'route_frozen':True})+'\n')


if __name__=='__main__':main()
