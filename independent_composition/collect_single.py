"""Generate fresh single-skill data at the remote host; never joint supervision."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--contexts',type=int,default=128)
    p.add_argument('--validation-contexts',type=int,default=64)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=910100001)
    args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);started=time.time()
    if (out/'DONE.json').exists():raise FileExistsError('Use a new collection directory')
    (out/'config.json').write_text(json.dumps({'args':vars(args),
        'scope':'New own-skill contexts and teacher trajectories generated directly on Colab. No local dataset upload or joint constraint pairs.'},indent=2)+'\n')
    report={}
    if args.env=='tworoom':
        from tworoom_gan.evaluate import load_checkpoint
        from tworoom_addons.data import sample_contexts,collect
        base,_,_=load_checkpoint('artifacts/final/base_policy.pt','cuda')
        for j,skill in enumerate(['keepout','waypoint']):
            folder=out/skill;folder.mkdir(exist_ok=True);report[skill]={}
            for split,count,offset in [('train',args.contexts,0),('val',args.validation_contexts,20000000)]:
                seed=args.seed+offset+10000*j
                ctx,audit=sample_contexts(base,count,seed,skill,'cuda')
                data=collect(base,ctx,seed+313,skill)
                np.savez_compressed(folder/f'{split}.npz',**data)
                np.savez_compressed(folder/f'{split}_contexts.npz',**{k:v.cpu().numpy() for k,v in ctx.items()})
                row={'contexts':count,'episodes':len(data['lengths']),'expert_success':float(data['success'].mean()),'base':audit}
                report[skill][split]=row;print(json.dumps({'skill':skill,'split':split,**row}),flush=True)
    else:
        from cube_addons.data import context_worker,pack
        for j,skill in enumerate(['keepout','waypoint']):
            report[skill]={}
            for split,count,offset in [('train',args.contexts,0),('val',args.validation_contexts,20000000)]:
                folder=out/skill/f'{split}_raw';folder.mkdir(parents=True,exist_ok=True)
                n=int(count*1.2)+8 if split=='train' else count
                seeds=list(range(args.seed+offset+j*10000,args.seed+offset+j*10000+n))
                jobs=[(s,skill,True,str(folder),True,None,skill=='waypoint',8) for s in seeds]
                rows=[]
                with mp.get_context('spawn').Pool(args.workers) as pool:
                    for row in pool.imap_unordered(context_worker,jobs,chunksize=1):
                        rows.append(row)
                        if len(rows)%8==0:print(json.dumps({'skill':skill,'split':split,'finished':len(rows),'requested':n,'seconds':time.time()-started}),flush=True)
                selected=sorted(r['seed'] for r in rows if r['paired_success'])[:count] if split=='train' else seeds
                if len(selected)<count:raise RuntimeError(f'Only {len(selected)}/{count} complete successful single-skill teacher contexts')
                pack(out,skill,split,selected,True,split=='train')
                report[skill][split]={'contexts':count,'attempted_contexts':n,'expert_failed_contexts':sum(not r['paired_success'] for r in rows)}
                (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'args':vars(args),'seconds':time.time()-started,'skills':report},indent=2)+'\n')


if __name__=='__main__':main()
