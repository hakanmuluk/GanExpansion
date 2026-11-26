"""Evaluate private action-manifold adapters with the unchanged task monitors."""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch

from .cvae import load_manifold, ManifoldComposer
from .evaluate_diffusion import evaluate_tworoom, evaluate_cube, summarize
from .train_diffusion import load_base
from .legacy import digest_file


def factory(args,models,base,seed):
    return ManifoldComposer(models,base,args.method,args.cycle,args.reconstruction,args.iterations,
                            args.starts,seed,args.prior,args.continuity,args.lr)


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--tasks',required=True);p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=16)
    p.add_argument('--method',choices=['keepout','waypoint','compose','fixed'],default='compose')
    p.add_argument('--cycle',type=float,default=.25);p.add_argument('--reconstruction',type=float,default=1.)
    p.add_argument('--iterations',type=int,default=32);p.add_argument('--starts',type=int,default=8)
    p.add_argument('--prior',type=float,default=.01);p.add_argument('--continuity',type=float,default=.01)
    p.add_argument('--lr',type=float,default=.05);p.add_argument('--execute',type=int,default=4)
    p.add_argument('--seed',type=int,default=41);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(2)
    if Path(args.tasks).name.startswith('test'):raise ValueError('Development validation only')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh evaluation output required')
    out.mkdir(parents=True)
    if len(args.checkpoints)==1:
        if args.method not in ['keepout','waypoint']:raise ValueError('Composition requires both private models')
        slots=[(int(args.method=='waypoint'),args.method)]
    elif len(args.checkpoints)==2:slots=[(0,'keepout'),(1,'waypoint')]
    else:raise ValueError('Expected one own-skill checkpoint or two composition checkpoints')
    base,source=load_base(args.env,args.device);models=[None,None];provenance=[]
    for path,(slot,skill) in zip(args.checkpoints,slots):
        model,saved=load_manifold(path,args.device)
        assert saved['config']['args']['skill']==skill and saved['config']['args']['env']==args.env
        assert saved['config']['base']==source
        models[slot]=model;provenance.append({'path':path,'sha256':digest_file(path),'step':saved['step']})
    config={'args':vars(args),'models':provenance,'base':source,'task_sha256':digest_file(args.tasks),
            'scope':'Frozen original base and two private action generators/encoders. Only latent/action consistency and Gaussian code priors are optimized. No simulator, hand geometry cost, projection, shared new learned network or joint supervision in inference.',
            'source_sha256':{p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');(out/'source').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    started=time.time()
    rows=evaluate_tworoom(args,models,base,out,factory) if args.env=='tworoom' else evaluate_cube(args,models,base,out,factory)
    result=summarize(rows);result['total_seconds']=time.time()-started
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    assert digest_file(source['path'])==source['sha256']
    assert all(p.grad is None and not p.requires_grad for m in [base,*models] if m is not None for p in m.parameters())
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'inference_environment_queries':0,'models_frozen':True})+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
