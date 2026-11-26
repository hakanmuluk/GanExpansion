"""Physical validation of original base plus denoising LoRA/FiLM adapters."""
import argparse
import json
from pathlib import Path
import time
import torch
from .action_adapter_diffusion import load
from .diffusion import ScoreComposer
from .evaluate_diffusion import evaluate_tworoom,evaluate_cube,summarize
from .inference_guard import GuardedPolicy
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--checkpoints',nargs=2,required=True);p.add_argument('--tasks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['keepout','waypoint','ratio','mean'],default='ratio')
    p.add_argument('--count',type=int,default=16);p.add_argument('--execute',type=int,default=8)
    p.add_argument('--sampling-steps',type=int,default=20);p.add_argument('--cold-time',type=int,default=70)
    p.add_argument('--warm-time',type=int,default=70);p.add_argument('--guidance',type=float,default=1.)
    p.add_argument('--seed',type=int,default=241);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(2)
    if Path(args.tasks).name.startswith(('final','test')):raise ValueError('Development banks only')
    if args.execute not in [1,2,4,8,16]:raise ValueError('Invalid execution interval')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    models=[];records=[]
    for skill,path in zip(['keepout','waypoint'],args.checkpoints):
        model,saved=load(path,args.device)
        assert saved['environment']==args.env and saved['skill']==skill
        assert all(name.startswith('actor.') for name,_ in model.named_parameters()),'Auxiliary learned model detected'
        models.append(model);records.append({'path':path,'sha256':digest_file(path),'step':saved['step']})
    from tworoom_addons.models import state_digest
    before=state_digest(*models);base=models[0].actor.base
    assert state_digest(base)==state_digest(models[1].actor.base)
    out.mkdir(parents=True)
    settings={'args':vars(args),'models':records,'task_sha256':digest_file(args.tasks),
        'scope':'Original frozen base and private LoRA/FiLM adapters only. No additional learned layers except extending existing adapter context inputs. No route, judge, inverse encoder or simulator query in policy inference.',
        'sampling':'Heuristic noisy density-ratio score combination from conditional/context-dropped forwards of the same private adapters; no exact product-distribution guarantee.',
        'learned_parameter_names':[[name for name,_ in model.named_parameters() if not name.startswith(('actor.base.','actor.original_encoder.'))] for model in models],
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
    def factory(args,models,base,seed):
        return GuardedPolicy(ScoreComposer(models,base,args.method,args.guidance,args.sampling_steps,
                             seed,args.warm_time,args.execute,cold_time=args.cold_time))
    started=time.time()
    rows=(evaluate_tworoom if args.env=='tworoom' else evaluate_cube)(args,models,base,out,factory)
    summary=summarize(rows);summary['total_seconds']=time.time()-started
    assert state_digest(*models)==before
    assert all(not p.requires_grad and p.grad is None for model in models for p in model.parameters())
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'inference_environment_queries':0,
        'models_frozen':True,'new_auxiliary_learned_networks':0,'inference_tripwires_enabled':True})+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
