"""Original frozen base on the unchanged single-skill test banks."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=['tworoom','cube'],required=True)
    p.add_argument('--skill',choices=['keepout','waypoint'],required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    if args.environment=='tworoom':
        from tworoom_addons.evaluate import load,contexts,evaluate,preservation
        torch.set_num_threads(2);base='artifacts/final/base_policy.pt'
        model,_,_=load(f'addon_artifacts/final/{args.skill}.pt','cuda',base)
        d=contexts(f'addon_artifacts/data/{args.skill}/test_contexts.npz','cuda')
        metric,result=evaluate(model,None,d,args.skill,8,gate=0)
        metric['preservation']=preservation(model,base,d)
        torch.save(result,out/'private_rollouts.pt')
    else:
        from cube_addons.parallel_eval import evaluate_parallel
        from cube_addons.evaluate import load_contexts
        cases=load_contexts(f'cube_addon_artifacts/data_force/{args.skill}/test_contexts.npz')
        metric,_=evaluate_parallel(None,'cube/artifacts/final/base_policy.pt',cases,args.skill,8,
            seed=123,workers=4,out=out/'native_rollouts')
    (out/'summary.json').write_text(json.dumps(metric,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'complete':True,'actor':'Original frozen base'})+'\n')
    print(json.dumps(metric),flush=True)


if __name__=='__main__':main()
