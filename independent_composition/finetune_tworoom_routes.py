"""Own-only margin finetuning for private TwoRoom trajectory models.

The first route regressor can average across geometric branches. This training
experiment emphasizes the worst violation in each complete route, including
fresh codes covering the code-search box. No geometry is queried at inference.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import torch
from .private_paths import RouteConfig,RouteDecoder
from .tworoom_paths import proposal
from .train_tworoom_routes import Data,margins,distances
from .legacy import digest_file


def worst_margin(predicted,condition,skill):
    average,valid,gaps=margins(predicted,condition,skill)
    path=(predicted+1)*112
    boundary=torch.minimum(path-21,203-path).amin((1,2))
    penalty=(torch.relu(3-gaps['wall_gap'])/10).square()
    penalty=penalty+(torch.relu(2-boundary)/10).square()
    penalty=penalty+(torch.relu((3 if skill=='keepout' else 2)-gaps['own_gap'])/10).square()
    if skill=='waypoint':
        center=(condition[:,10:12]+1)*112;radius=condition[:,12]*40
        goal=(condition[:,2:4]+1)*112
        d=distances(path,center);goald=distances(path,goal)
        visited=torch.sigmoid((radius[:,None]-d-.5)/.5).cummax(1).values
        penalty=penalty+((1-visited)*(torch.relu(17-goald)/10).square()).amax(1)
    return penalty+.1*average,valid


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--initial',required=True);p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=15000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--batch',type=int,default=256)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh margin experiment directory required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    # Preserve the exact loaded bytes even when the upstream trainer continues.
    for attempt in range(5):
        try:
            initial=Path(args.initial).read_bytes()
            ck=torch.load(io.BytesIO(initial),map_location='cuda',weights_only=False);break
        except (EOFError,RuntimeError):
            if attempt==4:raise
            time.sleep(1)
    assert ck['skill']==args.skill
    (out/'initial.pt').write_bytes(initial)
    model=RouteDecoder(RouteConfig(**ck['model_config'])).cuda();model.load_state_dict(ck['state'])
    train,valid=Data(args.train),Data(args.validation)
    assert not set(map(bytes,train.context.cpu().numpy())).intersection(map(bytes,valid.context.cpu().numpy()))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-6)
    rng=torch.Generator(device='cuda').manual_seed(args.seed+301)
    vrng=torch.Generator(device='cuda').manual_seed(982001)
    vz,vc,vt=valid.sample(min(2048,valid.size),vrng)
    config={'args':vars(args),'initial_step':ck['step'],'initial_sha256':hashlib.sha256(initial).hexdigest(),
        'train_sha256':digest_file(args.train),'validation_sha256':digest_file(args.validation),
        'scope':'Own single-skill paths/contexts only. Worst-route margins are training losses; no analytic predicate is part of the deployed generic decoder or composition optimizer. No newly shared learned network.',
        'loss':'Trajectory MSE + .2 worst-point MSE + .25 own worst-margin loss on labeled codes + .25 on fresh codes (half Gaussian, half uniform code box [-3,3]); own margin also includes base walls/bounds and own waypoint-before-goal ordering.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            z,c,target=train.sample(args.batch,rng);pred=model(z,c,proposal(z,c))
            error=(pred-target).square();own,_=worst_margin(pred,c,args.skill)
            fresh=torch.randn(z.shape,device='cuda',generator=rng).clamp(-3,3)
            fresh[:len(z)//2]=6*torch.rand((len(z)//2,z.shape[-1]),device='cuda',generator=rng)-3
            prior=model(fresh,c,proposal(fresh,c));priorloss,_=worst_margin(prior,c,args.skill)
            loss=error.mean()+.2*error.flatten(1).amax(1).mean()+.25*(own.mean()+priorloss.mean())
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if step%1000==0 or step==args.steps:
                with torch.no_grad():
                    value=model(vz,vc,proposal(vz,vc));pen,ok,gaps=margins(value,vc,args.skill)
                    rmse=112*(value-vt).square().mean().sqrt();score=1-float(ok.float().mean())+float(rmse)/112
                    row={'step':step,'loss':float(loss.detach()),'validation_path_rmse_px':float(rmse),
                        'validation_own_route_success':float(ok.float().mean()),
                        'validation_wall_clear':float((gaps['wall_gap']>=0).float().mean()),
                        'validation_own_region_margin_valid':float((gaps['own_gap']>=0).float().mean()),
                        'seconds':time.time()-started}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                saved={'state':model.state_dict(),'model_config':ck['model_config'],'skill':args.skill,
                       'step':step,'metrics':row,'config':config}
                torch.save(saved,out/'latest.pt')
                if score<best:best=score;torch.save(saved,out/'best.pt')
    (out/'DONE.json').write_text(json.dumps({'steps':args.steps,'best_own_validation_score':best})+'\n')


if __name__=='__main__':main()
