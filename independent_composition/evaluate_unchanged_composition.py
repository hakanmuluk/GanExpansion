"""Evaluate composition rules over unchanged original action generators only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import torch
from .legacy import digest_file
from .evaluate_diffusion import evaluate_tworoom,evaluate_cube,summarize
from .inference_guard import GuardedPolicy
from .unchanged_composition import CompositionConfig,OriginalComposer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--tasks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--config',required=True);p.add_argument('--count',type=int,default=16)
    p.add_argument('--execute',type=int,default=8);p.add_argument('--seed',type=int,default=241)
    p.add_argument('--model-seed',type=int,default=42);p.add_argument('--device',default='cuda')
    p.add_argument('--space',choices=['z','w'],default='z')
    p.add_argument('--keepout-checkpoint');p.add_argument('--waypoint-checkpoint')
    p.add_argument('--base-checkpoint',default='cube/artifacts/final/base_policy.pt')
    args=p.parse_args();torch.set_num_threads(2)
    if Path(args.tasks).name.startswith(('test','final')):raise ValueError('Development diagnostics only')
    if not 1<=args.execute<=16:raise ValueError('Execution exceeds original chunk length')
    if args.space=='w':
        if args.env!='cube':raise ValueError('Current W-space study is Cube only')
        from .style_composition import StyleCompositionConfig,StyleComposer
        config_type,policy_type=StyleCompositionConfig,StyleComposer
    else:config_type,policy_type=CompositionConfig,OriginalComposer
    c=config_type(**json.loads(Path(args.config).read_text()))
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    if args.env=='tworoom':
        from tworoom_composition.models import FrozenBank
        paths={} if args.model_seed==42 else {s:f'addon_artifacts/runs/{s}_s{args.model_seed}/best.pt' for s in ['keepout','waypoint']}
        bank=FrozenBank(device=args.device,**paths)
    else:
        if args.keepout_checkpoint or args.waypoint_checkpoint:
            if not args.keepout_checkpoint or not args.waypoint_checkpoint:raise ValueError('Pass both adapter checkpoint paths')
            from .cube_checkpoint_bank import CheckpointBank
            bank=CheckpointBank(args.keepout_checkpoint,args.waypoint_checkpoint,args.base_checkpoint,args.device)
        else:
            from cube_composition.models import FrozenBank
            bank=FrozenBank(seed=args.model_seed,device=args.device,base=args.base_checkpoint)
    assert all(m.config.new_z_dim==1 for m in [bank.k,bank.w])
    out.mkdir(parents=True)
    source={name:digest_file(Path(__file__).parent/name) for name in
            ['unchanged_composition.py','evaluate_unchanged_composition.py','evaluate_diffusion.py','inference_guard.py']}
    if args.space=='w':source['style_composition.py']=digest_file(Path(__file__).parent/'style_composition.py')
    if args.keepout_checkpoint:source['cube_checkpoint_bank.py']=digest_file(Path(__file__).parent/'cube_checkpoint_bank.py')
    (out/'config.json').write_text(json.dumps({'args':vars(args),'composition':asdict(c),'original_model_files':bank.files,
        'source_sha256':source,'task_sha256':digest_file(args.tasks),
        'scope':'Original base and adapter architectures; checkpoint weights frozen during this evaluation. Explicit checkpoint arguments may select newly trained original-architecture models. Output algebra and optional optimization of existing Z or user-authorized private mapped W. No simulator or inverse encoder during inference.',
        'history':'At most one actual issued control at each observed replanning state. No imagined states.',
        'optimization_space':args.space,
        'W_scope':'Private adapter mapping outputs only; base style/weights fixed; bounded style perturbations. No W+ or new learned module.' if args.space=='w' else None},indent=2)+'\n')
    (out/'source').mkdir()
    for name in source:shutil.copy2(Path(__file__).parent/name,out/'source'/name)
    def factory(args,unused,base,seed):return GuardedPolicy(policy_type(bank,c,seed))
    started=time.time()
    rows=(evaluate_tworoom if args.env=='tworoom' else evaluate_cube)(args,[],bank.k.base,out,factory)
    summary=summarize(rows);summary['total_seconds']=time.time()-started
    bank.assert_frozen()
    assert all(digest_file(path)==sha for path,sha in bank.files.items())
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'original_models_unchanged':True,
        'original_skill_code_dimensions':[1,1],'inference_environment_queries':0,
        'inference_tripwires_enabled':True,'new_learned_modules':0})+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
