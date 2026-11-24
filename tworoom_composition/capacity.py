"""Finite-domain capability checks; these do not prove manifold nonintersection."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .models import FrozenBank
from .evaluate import load_cases
from .search import SearchConfig,repeat_case,LatentSearch
from .rollout import rollout


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=32);p.add_argument('--codes',type=int,default=16)
    p.add_argument('--keepout',default='addon_artifacts/final/keepout.pt');p.add_argument('--waypoint',default='addon_artifacts/final/waypoint.pt')
    args=p.parse_args();torch.set_num_threads(2);bank=FrozenBank(args.keepout,args.waypoint);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({'args':vars(args),'frozen_files':bank.files},indent=2))
    values=torch.linspace(-3,3,args.codes,device='cuda');results=[]
    for task in args.tasks:
        case=load_cases(task,'cuda',args.count);n=len(case['start']);record={'tasks':task,'contexts':n,'codes':values.cpu().tolist(),'families':{}}
        for family in ['keepout','waypoint','sum']:
            codes=torch.stack([values,torch.zeros_like(values)],-1) if family=='keepout' else torch.stack([torch.zeros_like(values),values],-1)
            if family=='sum':codes=torch.cartesian_prod(values,values)
            allrows=[]
            # Bound GPU batch memory and save every finite code success, not only the best.
            for lo in range(0,n,4):
                c={k:v[lo:lo+4] for k,v in case.items()};b=len(c['start']);cc=repeat_case(c,len(codes))
                z=codes.repeat(b,1);cfg=SearchConfig(method='fixed',family=family,objective='agreement',regularizer='none',temporal='fixed')
                rows,traj,info=rollout(bank,cc,cfg,initial_latents=z)
                for row in rows:row['context']=lo+row['context']//len(codes)
                allrows.extend(rows)
            success=np.array([r['success'] for r in allrows]).reshape(n,len(codes))
            record['families'][family]={'finite_grid_oracle_success':float(success.any(1).mean()),'per_context_any_success':success.any(1).tolist(),
                'per_context_success_count':success.sum(1).tolist(),'fixed_code_success_matrix':success.tolist(),
                'strict_failures_despite_physical_witness':int((~success.any(1)).sum())}
            target=out/(Path(task).stem+'_'+family+'_records.json');target.write_text(json.dumps(allrows,indent=2))
            print(Path(task).stem,family,record['families'][family]['finite_grid_oracle_success'],flush=True)
        results.append(record);(out/'capacity.json').write_text(json.dumps(results,indent=2))
    bank.assert_frozen();(out/'DONE.json').write_text(json.dumps({'complete':True,'caveat':'Finite fixed-code grid and specified executors only; not a mathematical nonintersection proof.'}))


if __name__=='__main__':main()
