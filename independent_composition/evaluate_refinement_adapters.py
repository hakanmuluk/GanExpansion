"""Physical validation of base action stream plus independent edit adapters."""
import argparse
import json
from pathlib import Path
import time
import torch
from .refinement_adapters import load
from .refinement_policy import RefinementPolicy
from .evaluate_diffusion import evaluate_tworoom,evaluate_cube,summarize
from .legacy import digest_file
from .inference_guard import GuardedPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--checkpoints',nargs=2,required=True);p.add_argument('--tasks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['base','keepout','waypoint','compose','mean','residual'],default='compose')
    p.add_argument('--sweeps',type=int,default=2);p.add_argument('--order',choices=['wk','kw'],default='wk')
    p.add_argument('--base-period',type=int,default=8);p.add_argument('--count',type=int,default=16)
    p.add_argument('--seed',type=int,default=241);p.add_argument('--device',default='cuda');args=p.parse_args()
    args.execute=1;torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if Path(args.tasks).name.startswith(('final','test')):raise ValueError('Development validation only')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    models=[];provenance=[]
    for skill,path in zip(['keepout','waypoint'],args.checkpoints):
        model,saved=load(path,args.device);assert model.environment==args.env and model.skill==skill
        assert all(n.startswith('actor.') for n,_ in model.named_parameters())
        models.append(model);provenance.append({'path':path,'sha256':digest_file(path),'step':saved['step']})
    from tworoom_addons.models import state_digest
    before=state_digest(*models);assert state_digest(models[0].actor.base)==state_digest(models[1].actor.base)
    out.mkdir(parents=True)
    config={'args':vars(args),'models':provenance,'task_sha256':digest_file(args.tasks),
        'scope':'Original frozen action base plus two independent classical LoRA/FiLM adapters only. Candidate action appended to each own context. No route, judge, inverse model, shared learned executor or inference task cost.',
        'control':'Base retains its original eight-control stream; adapters refine the actual next control from each fresh observation. Only first adapter head is trained/executed. Cube yaw/gripper remain exactly from base stream.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    def factory(args,models,base,seed):
        return GuardedPolicy(RefinementPolicy(models,args.method,args.sweeps,args.order,args.base_period,seed))
    started=time.time();rows=(evaluate_tworoom if args.env=='tworoom' else evaluate_cube)(args,models,models[0].actor.base,out,factory)
    summary=summarize(rows);summary['total_seconds']=time.time()-started
    assert before==state_digest(*models)
    assert all(not p.requires_grad and p.grad is None for m in models for p in m.parameters())
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'models_frozen':True,'new_auxiliary_learned_networks':0,
        'inference_environment_queries':0,'inference_tripwires_enabled':True,'cube_yaw_gripper_preserved':True})+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
