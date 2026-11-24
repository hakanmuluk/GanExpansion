"""Small held-object ARX surrogate fit only to old single-skill TRAIN episodes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def positions(context):
    center=context[...,:3]*np.array([.2,.3,.2])+np.array([.425,0,.2])
    return center+.15*context[...,4:7],center+.15*context[...,7:10]


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    xs=[];ys=[];labels=[];sources={};episodes=[]
    for skill in ['keepout','waypoint']:
        path=Path(f'cube_addon_artifacts/physical/{skill}_s{args.seed}/episodes.npz');sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        provenance=json.loads((path.parent/'summary.json').read_text());assert Path(provenance['args']['contexts']).name=='train_contexts.npz'
        with np.load(path) as f:
            arrays={k:f[k] for k in ['contexts','states','actions','lengths']}
            for i,length in enumerate(arrays['lengths']):
                c=arrays['contexts'][i,:length];states=arrays['states'][i,:length];actions=arrays['actions'][i,:length];cube,ee=positions(c)
                mask=(c[:-1,13]>.5)&(c[1:,13]>.5)&(cube[:-1,2]>.10)&(cube[1:,2]>.10)
                x=np.concatenate([actions[:-1,:3],states[:-1,-3:]*.01,np.ones((length-1,1))],-1)
                y=np.concatenate([np.diff(ee,axis=0),np.diff(cube,axis=0)],-1)
                xs.append(x[mask]);ys.append(y[mask]);labels.extend([i%5==0]*int(mask.sum()));episodes.append((x,y,mask,i%5==0))
    x=np.concatenate(xs);y=np.concatenate(ys);val=np.array(labels);ridge=1e-6*np.eye(7)
    b=np.linalg.solve(x[~val].T@x[~val]+ridge,x[~val].T@y[~val]);pred=x[val]@b;error=pred-y[val];roll_errors=[]
    for xx,yy,mask,heldout in episodes:
        if not heldout:continue
        for t in range(0,len(mask)-8,8):
            if not mask[t:t+8].all():continue
            velocity=xx[t,3:6];total=np.zeros(6)
            for u in range(t,t+8):
                delta=np.r_[xx[u,:3],velocity,1.]@b;total+=delta;velocity=delta[:3]
            roll_errors.append(total-yy[t:t+8].sum(0))
    out=Path(f'cube_composition_artifacts/dynamics/seed{args.seed}.json');out.parent.mkdir(parents=True,exist_ok=True)
    result={'model':'Held-object linear ARX: [actionXYZ, previousEEdisplacement,1] -> [nextEEdisplacement,nextCubedisplacement]',
        'coefficients':b.tolist(),'source_single_skill_training_sha256':sources,'joint_training_episodes':0,
        'train_transitions':int((~val).sum()),'validation_transitions':int(val.sum()),'split':'Every fifth original episode held out separately within each skill; no joint contexts.',
        'one_step_validation_rmse_m':np.sqrt(np.mean(error**2,axis=0)).tolist(),
        'eight_step_validation_endpoint_rmse_m':np.sqrt(np.mean(np.array(roll_errors)**2,axis=0)).tolist(),
        'eight_step_validation_segments':len(roll_errors),'limitations':'Only bilateral-held regime. No learned contact, grasp, release, or robot collision model. Native physics remains final verifier.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
