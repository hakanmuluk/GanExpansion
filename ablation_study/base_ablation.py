"""Precisely remove the specified latent-enforcing scheme from base training.

The effective trainer is an auditable mechanical edit of the original source.
G, D, data, optimizers, GAN warmup/R1, EMA and selection logic are unchanged.
The unused E is retained only for initialization RNG/checkpoint compatibility;
it receives no training gradient. Its recovery metrics are not valid results.
"""
import argparse
import difflib
import importlib
import json
from pathlib import Path
import sys
import types
from .rng_replay import TrainingRNG
from .run_original import arguments


def replace_block(source,start,end,value):
    assert source.count(start)==1,(start,source.count(start))
    a=source.index(start);b=source.index(end,a)
    return source[:a]+value+source[b:]


def transform(environment,source):
    if environment=='tworoom':
        source=source.replace('semantic_style=args.style_weight > 0 or args.prior_rec_weight > 0)',
                              'semantic_style=True)')
        source=replace_block(source,'            mu, logvar = encoder(',
            '            state, target, mask =',
            '            mu = logvar = torch.zeros((len(b["lengths"]), config.z_dim), device=device)\n'
            '            torch.randn_like(mu)  # Match original posterior-noise RNG draw; its value is unused.\n'
            '            style_aux = torch.zeros((), device=device)\n')
        source=replace_block(source,'            z_per_chunk =','            gan_scale =',
            '            prior_z = torch.randn((len(b["lengths"]), config.z_dim), device=device)\n'
            '            z_per_chunk = prior_z[:, None].expand(-1, args.chunks, -1)\n'
            '            random_actions = generator(state, z_per_chunk)\n'
            '            reconstructed = random_actions  # No posterior branch; GAN uses the prior alone.\n')
        source=replace_block(source,'            rec = masked_l1(','            adv = torch.zeros',
            '            rec = kl = lat = prior_rec = torch.zeros((), device=device)\n')
        # Original smoothness survives, applied to the remaining prior proposal.
        # Feedback depends on E(demonstration) and paired latent-labeled expert
        # targets, so it cannot operate after removing that pathway.
    else:
        source=replace_block(source,'            mu,logvar=e(','            state,target,mask=',
            '            mu=logvar=torch.zeros((len(b["lengths"]),config.z_dim),device=device)\n'
            '            torch.randn_like(mu)  # Preserve the original prior-code draw stream.\n'
            '            known_z=torch.zeros_like(mu)\n')
        source=replace_block(source,'            rec_actions=g(','            gan_scale=',
            '            prior_z=torch.randn((len(b["lengths"]),config.z_dim),device=device)\n'
            '            prior_actions=g(state,expand(prior_z))\n'
            '            rec_actions=prior_actions  # No posterior branch.\n')
        source=replace_block(source,'            rec=reconstruction(','            adv=torch.zeros',
            '            rec=kl=lat=torch.zeros((),device=device)\n')
        old='            loss.backward();torch.nn.utils.clip_grad_norm_(list(g.parameters())+list(e.parameters()),5);opt_ge.step()'
        assert source.count(old)==1
        source=source.replace(old,
            '            if loss.requires_grad:\n'
            '                loss.backward();torch.nn.utils.clip_grad_norm_(list(g.parameters())+list(e.parameters()),5);opt_ge.step()\n'
            '            # Original GAN warmup can leave no active G loss: take no artificial update.')
    return source


def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=['tworoom','cube'],required=True)
    p.add_argument('--config',required=True);p.add_argument('--rng-trace',required=True)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());run=Path(config['run'])
    run.mkdir(parents=True,exist_ok=False)
    name='tworoom_gan.train' if args.environment=='tworoom' else 'cube.cube_gan.train'
    original=importlib.import_module(name);source=Path(original.__file__).read_text()
    effective=transform(args.environment,source)
    path=run/'effective_train.py';path.write_text(effective)
    (run/'only_requested_loss_removal.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True),effective.splitlines(True),fromfile=name,tofile='adversarial_only')))
    disabled=['rec_weight','latent_weight','kl_weight','style_weight','prior_rec_weight','known_rec_weight',
              'prior_match_weight','rollout_cycle_weight','mode_weight']
    # This E-dependent term has no valid encoder-free counterpart. No substitute.
    disabled+=['feedback_weight','physical_style_weight']
    changes={key:{'original':value,'ablation':0.} for key,value in config.items() if key in disabled and value!=0}
    for key in disabled:
        if key in config:config[key]=0.
    assert all(config.get(key,0)==0 for key in disabled),'A requested latent/imitation objective remains active'
    (run/'loss_removal.json').write_text(json.dumps({'changes':changes,
        'retained':'Original conditional hinge GAN, R1, warmup and applicable action smoothness. No replacement latent losses.',
        'posterior_branch':'Removed; prior proposal used alone (duplicating it in existing mean reductions leaves the same mathematical GAN objective).',
        'encoder':'Unused checkpoint-compatible placeholder; no encoder recovery results are interpreted.',
        'random_numbers':'Consume the removed posterior-noise draw without using its value, so the retained augmentation and prior-code draws match the reference.',
        'feedback':'Original TwoRoom feedback requires encoded demonstrations and known-style targets; removed with that latent pathway.',
        'warmup':'If Cube has no active generator loss before original GAN warmup, skip backward/update rather than add a replacement loss.'},indent=2)+'\n')
    module=types.ModuleType(name+'.adversarial_only');module.__file__=str(path)
    module.__package__=original.__package__;sys.modules[module.__name__]=module
    exec(compile(effective,str(path),'exec'),module.__dict__)
    trace=TrainingRNG(args.rng_trace,'replay',module.__name__);trace.install(module.EpisodeDataset)
    if hasattr(module,'parser'):
        cli=arguments(module.parser(),config)
    else:
        cli=[]
        for key,value in config.items():
            if value is None or value is False:continue
            cli.append('--'+key.replace('_','-'))
            if value is not True:cli.append(str(value))
    sys.argv=[str(path),*cli]
    try:module.main()
    finally:trace.flush()
    (run/'controlled_ablation.json').write_text(json.dumps({'kind':'base_adversarial_only',
        'sampled_batches_exactly_match_reference':len(trace.observed),'trace':args.rng_trace},indent=2)+'\n')


if __name__=='__main__':main()
