"""Run an original trainer, optionally tracing/replaying its sampled batches.

All trainer arguments come from a frozen existing configuration. Architecture
ablation hooks are installed separately; loss implementations remain original.
"""
import argparse
import importlib
import hashlib
import json
from pathlib import Path
import sys
from .rng_replay import TrainingRNG


def arguments(parser, config):
    defaults=vars(parser.parse_args(_required_only(parser,config)))
    result=[]
    known={action.dest:action for action in parser._actions}
    for name,value in config.items():
        if name not in known:
            raise ValueError(f'Original trainer does not accept recorded argument: {name}')
        if value is None:continue
        flag='--'+name.replace('_','-');action=known[name]
        if action.nargs==0:
            if value!=defaults[name]:result.append(flag)
        else:result.extend([flag,str(value)])
    return result


def _required_only(parser, config):
    out=[]
    for action in parser._actions:
        if action.required:
            out.extend([action.option_strings[0],str(config[action.dest])])
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--module',required=True)
    p.add_argument('--config',required=True);p.add_argument('--rng-trace',required=True)
    p.add_argument('--rng-mode',choices=['record','replay'],required=True)
    p.add_argument('--rank',type=int);p.add_argument('--reference-rank',type=int)
    p.add_argument('--lora-only',action='store_true');args=p.parse_args()
    module=importlib.import_module(args.module);config=json.loads(Path(args.config).read_text())
    implementation_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path(module.__file__),Path(__file__),Path(__file__).with_name('rng_replay.py'),
         Path(__file__).with_name('model_factory.py'),Path(__file__).with_name('evaluation.py')]}
    if args.rank is not None:
        from .model_factory import install_adapter_factory
        install_adapter_factory(module,args.rank,args.reference_rank,args.lora_only)
        config['rank']=args.rank
    elif args.lora_only:raise ValueError('LoRA-only requires explicit rank')
    trace=TrainingRNG(args.rng_trace,args.rng_mode,args.module)
    trace.install(module.EpisodeDataset)
    if args.rank is not None:
        # Add explicit variant metadata to NEW checkpoints only. State tensors
        # and the original training/saving schedule are not changed.
        import torch
        original_save=torch.save
        def save(value,*positional,**keywords):
            if isinstance(value,dict) and 'addon' in value and 'adapter_config' in value:
                value={**value,'ablation_config':{'lora_only':args.lora_only,
                    'rank':args.rank,'reference_rank':args.reference_rank}}
            return original_save(value,*positional,**keywords)
        torch.save=save
        if args.module.startswith('cube'):
            from .evaluation import install_cube_variant_loader
            install_cube_variant_loader()
    if hasattr(module,'parser'):cli=arguments(module.parser(),config)
    else:
        cli=[]
        for name,value in config.items():
            if value is None or value is False:continue
            cli.append('--'+name.replace('_','-'))
            if value is not True:cli.append(str(value))
    sys.argv=[module.__file__,*cli]
    try:module.main()
    finally:trace.flush()
    run=Path(config['run'])
    receipt={'module':args.module,'rank':args.rank,'reference_rank':args.reference_rank,
        'lora_only':args.lora_only,'rng_mode':args.rng_mode,'rng_trace':args.rng_trace,
        'sampled_batches_verified':len(trace.observed),'first_step':min(trace.observed),'last_step':max(trace.observed),
        'implementation_sha256_at_launch':implementation_hashes}
    (run/'controlled_ablation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (run/f'controlled_ablation_steps_{min(trace.observed)}_{max(trace.observed)}.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
