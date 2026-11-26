"""Broader SINGLE-skill teacher trajectories; no learned shared controller.

Reuses the older analytic single-skill path teacher strictly for data collection.
Its path representation is NOT an inference input or policy architecture here.
No actor checkpoint is passed to the collector. New models see current state,
own 20D context, base code, and frozen-base actions only.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--contexts',type=int,default=256);p.add_argument('--codes',type=int,default=2)
    p.add_argument('--seed',type=int,default=1218100001);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(2);out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh broad-data output required')
    out.mkdir(parents=True,exist_ok=True)
    from cube_agreement.collect_single import init,collect
    settings={'code_scale':.7,'actor':None}
    config={'args':vars(args),'settings':settings,
            'scope':'One own region per episode. Analytic teacher only during collection; no joint scene, witness, or learned shared actor supplies training actions. No path or latent-anchor representation is supplied to the learned policy at inference.',
            'source_sha256':{str(p):digest_file(p) for p in [Path('cube_agreement/collect_single.py'),Path('cube_agreement/physical.py'),Path('cube_agreement/manifolds.py')]}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    jobs=[(skill,args.seed+j*1000000+i,i,str(out/'raw'),args.codes)
          for j,skill in enumerate(['keepout','waypoint']) for i in range(args.contexts)]
    rows=[];contexts=[];started=time.time()
    with mp.get_context('spawn').Pool(args.workers,initializer=init,initargs=(settings,)) as pool:
        for record in pool.imap(collect,jobs,chunksize=1):
            contexts.append(record);rows.extend(record['episodes'])
            (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
            if len(contexts)%16==0:print(json.dumps({'contexts':len(contexts),'episodes':len(rows),
                'successful_own_skill':sum(r['success'] for r in rows),'seconds':time.time()-started}),flush=True)
    report={}
    for skill in ['keepout','waypoint']:
        report[skill]={};(out/skill).mkdir()
        for split in ['train','val']:
            selected=[r for r in rows if r['skill']==skill and r['success'] and
                      ((r['seed']%5==0)==(split=='val'))]
            if not selected:raise RuntimeError(f'No successful own-skill demonstrations for {skill}/{split}')
            examples=[]
            for row in selected:
                with np.load(row['source']) as d:
                    assert str(d['skill'])==skill and not bool(d['is_feedback'])
                    geometry=d['geometry']
                    other=geometry[6:10] if skill=='keepout' else geometry[2:6]
                    assert np.count_nonzero(other)==0,'Other skill geometry entered training data'
                    examples.append({k:d[k] for k in ['states','contexts','actions','base_z','new_z','seed']})
            fields={}
            for key in ['states','contexts','actions']:
                a=np.zeros((len(examples),240,*examples[0][key].shape[1:]),np.float32)
                for i,e in enumerate(examples):a[i,:len(e[key])]=e[key]
                fields[key]=a
            fields['lengths']=np.asarray([len(e['actions']) for e in examples],np.int64)
            for key in ['base_z','new_z','seed']:fields[key]=np.stack([e[key] for e in examples])
            np.savez_compressed(out/skill/f'{split}.npz',**fields)
            report[skill][split]={'successful_episodes':len(examples),'contexts':len(np.unique(fields['seed'])),
                'source_hashes':{r['source']:digest_file(r['source']) for r in selected}}
    (out/'context_records.json').write_text(json.dumps(contexts,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'seconds':time.time()-started,'collected_episodes':len(rows),
        'retained_own_successes':sum(r['success'] for r in rows),'skills':report,
        'validation_context_banks':'Environment validation and final test banks remain unfiltered; only training/inverse-model demonstration targets are filtered by own success.'},indent=2)+'\n')


if __name__=='__main__':main()
