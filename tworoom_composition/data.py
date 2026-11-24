"""New geometry-stratified contexts, split before any method evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tworoom_addons.data import sample_contexts
from tworoom_addons.geometry import frame,unframe
from tworoom_gan.style import decode_style
from .planner import witness,segment_distance


def build(base,count,seed,kind,device):
    # Original skill distributions are used directly for the supported bank.
    raw,_=sample_contexts(base,count*30,seed,'keepout',device)
    data={k:v.cpu().numpy() for k,v in raw.items()};rng=np.random.default_rng(seed+71)
    mode=(data['base_z'][:,0]>=0)
    x=rng.uniform(155,175,len(mode));y=rng.uniform(160,176,len(mode));y=np.where(mode,224-y,y)
    direction=np.where(data['goal'][:,0]>=112,1,-1)
    wp=np.stack([112+direction*(x-112),y,rng.uniform(7,9,len(mode))],-1).astype(np.float32)
    if kind=='coupled_ood':
        # Deliberately outside the keep-out adapter's trained first-room support.
        cx=x-rng.uniform(20,30,len(x));cy=y+rng.uniform(-8,8,len(x))
        data['spec']=np.stack([112+direction*(cx-112),cy,rng.uniform(10,13,len(x))],-1).astype(np.float32)
    elif kind=='infeasible':wp[:,:2]=data['spec'][:,:2]
    accepted=[];proofs=[];attempts=[]
    for i in range(len(mode)):
        case={k:data[k][i] for k in ['start','goal','base_z']};case.update(keepout=data['spec'][i],waypoint=wp[i],seed=np.int64(seed+i))
        # Geometry-only difficulty descriptor: does one nominal approach cross the goal?
        canonical_goal=np.array([112+direction[i]*(case['goal'][0]-112),case['goal'][1]])
        exit=np.array([142.,[56.,168.][int(mode[i])]])
        approaches=[]
        for side in [-1,1]:
            via=np.array([x[i]+side*22,y[i]])
            approaches.append(min(segment_distance(exit,via,canonical_goal),segment_distance(via,[x[i],y[i]],canonical_goal)))
        conflict=min(approaches)<17 and max(approaches)>18
        if kind=='supported_easy' and min(approaches)<20:continue
        if kind=='supported_conflict' and not conflict:continue
        start=time.perf_counter();proof=None if kind=='infeasible' else witness(case)
        attempts.append({'candidate':i,'witness_found':proof is not None,'planner_seconds':time.perf_counter()-start})
        if kind!='infeasible' and proof is None:continue
        case.update(kind=np.array(kind),geometric_conflict=np.bool_(conflict),feasible=np.bool_(proof is not None))
        accepted.append(case);proofs.append(proof)
        if len(accepted)==count:break
    if len(accepted)<count:raise RuntimeError(f'{kind}: only {len(accepted)}/{count}; {len(attempts)} eligible planner queries from {len(mode)} geometry candidates')
    return accepted,proofs,attempts


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='composition_artifacts/tasks');p.add_argument('--count',type=int,default=64)
    p.add_argument('--splits',nargs='+',default=['dev','val','test']);p.add_argument('--kinds',nargs='+',default=['supported_easy','supported_conflict','coupled_ood','infeasible'])
    p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(2)
    from tworoom_gan.evaluate import load_checkpoint
    base,encoder,_=load_checkpoint('artifacts/final/base_policy.pt',args.device)
    base.requires_grad_(False);encoder.requires_grad_(False)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    report={};started=time.time()
    for split,offset in [('dev',2100001),('val',3100001),('test',4100001),('train',5100001)]:
        if split not in args.splits:continue
        for kind in args.kinds:
            index=['supported_easy','supported_conflict','coupled_ood','infeasible'].index(kind)
            cases,proofs,attempts=build(base,args.count,offset+index*100000,kind,args.device)
            arrays={key:np.stack([case[key] for case in cases]) for key in cases[0]}
            path=root/f'{split}_{kind}.npz';np.savez_compressed(path,**arrays)
            positions=np.zeros((len(cases),241,2),np.float32);actions=np.zeros((len(cases),240,2),np.float32);lengths=np.zeros(len(cases),np.int64)
            for i,proof in enumerate(proofs):
                if proof is None:continue
                lengths[i]=proof['steps'];positions[i]=proof['positions'][-1];positions[i,:len(proof['positions'])]=proof['positions'];actions[i,:proof['steps']]=proof['actions']
            np.savez_compressed(root/f'{split}_{kind}_witnesses.npz',positions=positions,actions=actions,lengths=lengths)
            record={'contexts':len(cases),'feasible':int(arrays['feasible'].sum()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'generation_seed':offset+index*100000,'candidate_attempts':attempts,'witness_training_use':'None: geometry witnesses certify feasibility only; skill weights stay frozen.'}
            if kind=='infeasible':record['analytic_certificate']='Required waypoint disk lies wholly inside forbidden disk inflated by agent radius.'
            report[split+'_'+kind]=record;(root/'manifest.json').write_text(json.dumps({'tasks':report,'args':vars(args),'elapsed_seconds':time.time()-started,
                'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('planner.py')]}},indent=2)+'\n')
            print(split,kind,'saved',len(cases),'elapsed',time.time()-started,flush=True)


if __name__=='__main__':main()
