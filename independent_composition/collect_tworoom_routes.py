"""Independent own-skill trajectory labels with ten Gaussian route controls.

This collector alone uses an analytic geometric teacher. The learned decoder's
forward function has no teacher, constraint projection, or environment query.
"""
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from .legacy import digest_file


def init_worker():torch.set_num_threads(1)


def label(job):
    from tworoom_composition.planner import astar,clear_segment,clear_segments
    from .tworoom_paths import control_points
    skill,context,code=job
    points=(control_points(torch.tensor(code)[None],torch.tensor(context)[None])[0].numpy()+1)*112
    start,goal=points[0].copy(),points[-1].copy()
    spec=np.r_[(context[10:12]+1)*112,context[12]*40]
    obstacle=spec if skill=='keepout' else np.array([-1000.,-1000.,0.])
    z0,z1=context[22:24]
    route=int(z0>=0);offset=24*math.erf(abs(float(z0))/2**.5)-12
    door=np.array([112.,[56.,168.][route]+offset])
    grid=np.arange(24.,201.,4.);grid=np.stack(np.meshgrid(grid,grid),-1).reshape(-1,2)
    safe=clear_segments(grid,grid,obstacle,goal,route,margin=1.5);grid=grid[safe]
    if not len(grid):return None
    targets=[]
    for i,point in enumerate(points[1:-1]):
        if skill=='waypoint' and i==4:
            v=np.asarray(code[-2:]);point=spec[:2]+.6*spec[2]*v/np.sqrt(1+np.square(v).sum())
            if not clear_segment(point,point,obstacle,goal,route):point=spec[:2]
            if not clear_segment(point,point,obstacle,goal,route):return None
        elif not clear_segment(point,point,obstacle,goal,route):
            point=grid[np.linalg.norm(grid-point,axis=1).argmin()]
        targets.append(point)
    targets.append(goal);previous=start;paths=[]
    for i,target in enumerate(targets):
        avoid=goal if i<len(targets)-1 else None
        if (previous[0]>112)!=(target[0]>112):
            first=astar(previous,door,obstacle,avoid,route)
            last=astar(door,target,obstacle,avoid,route)
            if first is None or last is None:return None
            path=np.concatenate([first,last[1:]])
        else:path=astar(previous,target,obstacle,avoid,route)
        if path is None:return None
        paths.append(path);previous=target
    nodes=np.concatenate([paths[0],*[p[1:] for p in paths[1:]]])
    cumulative=np.r_[0,np.linalg.norm(np.diff(nodes,axis=0),axis=1).cumsum()]
    probability=.5*(1+math.erf(float(z1)/2**.5))
    speed=.55+.46*probability if probability<.5 else .78+.44*(probability-.5)
    if cumulative[-1]>min(560.,160*5*speed):return None
    keep=np.r_[True,np.diff(cumulative)>1e-6]
    path=np.stack([np.interp(np.linspace(0,cumulative[-1],65),cumulative[keep],nodes[keep,j]) for j in range(2)],-1)
    return (path/112-1).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--contexts',type=int,default=2048);p.add_argument('--validation-contexts',type=int,default=256)
    p.add_argument('--codes',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=1619200001);args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh own-route data directory required')
    out.mkdir(parents=True)
    from tworoom_gan.evaluate import load_checkpoint
    from tworoom_addons.data import sample_contexts
    from tworoom_gan.env import observation
    from tworoom_addons.geometry import skill_features
    from .tworoom_paths import condition
    base,_,_=load_checkpoint('artifacts/final/base_policy.pt','cuda')
    base.eval().requires_grad_(False)
    config={'args':vars(args),'scope':__doc__,
            'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');report={};started=time.time()
    for j,skill in enumerate(['keepout','waypoint']):
        (out/skill).mkdir();report[skill]={}
        for split,count,offset in [('train',args.contexts,0),('val',args.validation_contexts,20000000)]:
            seed=args.seed+j*10000+offset;rng=np.random.default_rng(seed+91)
            contexts,_=sample_contexts(base,count,seed,skill,'cuda')
            # Broaden only the named region, independently of any other skill.
            direction=torch.where(contexts['goal'][:,0]>=112,1.,-1.)
            x=rng.uniform(43,181,count);x=np.where(x<112,np.minimum(x,84),np.maximum(x,140))
            proposed=torch.tensor(np.stack([x,rng.uniform(38,186,count),
                rng.uniform(10,14,count) if skill=='keepout' else rng.uniform(7,10,count)],-1),device='cuda',dtype=torch.float32)
            proposed[:,0]=112+direction*(proposed[:,0]-112)
            keep=torch.arange(count,device='cuda')%2==1
            keep&=(proposed[:,:2]-contexts['start']).norm(dim=-1)>proposed[:,2]+12
            keep&=(proposed[:,:2]-contexts['goal']).norm(dim=-1)>proposed[:,2]+20
            contexts['spec']=torch.where(keep[:,None],proposed,contexts['spec'])
            c=skill_features(contexts['start'],contexts['goal'],contexts['start'],contexts['spec'],
                torch.zeros(count,device='cuda',dtype=torch.bool),0,skill)
            features=condition(observation(contexts['start'],contexts['goal']),c,contexts['base_z']).cpu().numpy()
            codes=np.clip(rng.standard_normal((count,args.codes,10)),-3,3).astype(np.float32)
            jobs=[(skill,features[i],codes[i,k]) for i in range(count) for k in range(args.codes)]
            labels=[]
            with mp.get_context('spawn').Pool(args.workers,initializer=init_worker) as pool:
                for index,path in enumerate(pool.imap(label,jobs,chunksize=4)):
                    labels.append(path)
                    if (index+1)%512==0:print(json.dumps({'skill':skill,'split':split,'processed':index+1,
                        'accepted':sum(p is not None for p in labels),'seconds':time.time()-started}),flush=True)
            accepted=np.asarray([p is not None for p in labels]);n=int(accepted.sum())
            assert n>0
            data={'condition':np.repeat(features,args.codes,axis=0)[accepted],
                  'code':codes.reshape(-1,10)[accepted],'path':np.stack([p for p in labels if p is not None]),
                  'context_id':np.repeat(np.arange(count),args.codes)[accepted]}
            np.savez_compressed(out/skill/f'{split}.npz',**data)
            report[skill][split]={'requested_contexts':count,'attempted_paths':len(jobs),'accepted_paths':n,
                'accepted_contexts':len(np.unique(data['context_id'])),'seconds':time.time()-started}
            (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'skill':skill,'split':split,**report[skill][split]}),flush=True)
    (out/'DONE.json').write_text(json.dumps({'skills':report,'seconds':time.time()-started},indent=2)+'\n')


if __name__=='__main__':main()
