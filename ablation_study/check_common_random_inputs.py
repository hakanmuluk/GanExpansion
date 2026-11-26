"""Verify retained stochastic inputs using the ACTUAL original/edited prefixes.

Only neural outputs in the removed posterior branch are stubbed: their values
do not affect augmentation or the retained independent prior-code sampling.
Both prefixes use the real dataset sampler, augmentation, and expert relabeling.
"""
import argparse
import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .base_ablation import transform


def prefix(source):
    tree=ast.parse(source)
    main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    loop=next(x for x in ast.walk(main) if isinstance(x,ast.For) and ast.get_source_segment(source,x.target)=='step')
    start=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='b' for t in x.targets))
    stop=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gan_scale' for t in x.targets))
    return compile(ast.fix_missing_locations(ast.Module(body=loop.body[start:stop],type_ignores=[])),'original_or_ablation_prefix','exec')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    results=[]
    for env in ['tworoom','cube']:
        name='tworoom_gan.train' if env=='tworoom' else 'cube.cube_gan.train'
        module=importlib.import_module(name);source=Path(module.__file__).read_text()
        configuration='ablation_study/configs/tworoom_base_full_s42.json' if env=='tworoom' else 'ablation_study/configs/cube_original_base_full_s42.json'
        cfg=json.loads(Path(configuration).read_text());ablated=deepcopy(cfg)
        for key in ['style_weight','rec_weight','latent_weight','kl_weight','prior_rec_weight','feedback_weight','rollout_cycle_weight']:
            if key in ablated:ablated[key]=0.
        ds=module.EpisodeDataset(Path(cfg['data'])/'train.npz','cuda')
        def encoder(states,actions,lengths):
            x=torch.zeros((len(states),2),device=states.device);return x,x
        action_dim=2 if env=='tworoom' else 5
        def generator(states,z):return torch.zeros((*states.shape[:-1],16,action_dim),device=states.device)
        common={**module.__dict__,'train_data':ds,'ds':ds,'encoder':encoder,'e':encoder,
            'generator':generator,'g':generator,'config':SimpleNamespace(z_dim=2),'device':'cuda'}
        full_code=prefix(source);ablation_code=prefix(transform(env,source))
        for step in [1,1001,1500,2000,5000]:
            torch.manual_seed(2145+step);cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
            full={**common,'args':SimpleNamespace(**cfg),'step':step};exec(full_code,full)
            torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
            reduced={**common,'args':SimpleNamespace(**ablated),'step':step};exec(ablation_code,reduced)
            fields=['state','target','mask','prior_z']
            assert all(torch.equal(full[k],reduced[k]) for k in fields),(env,step)
            results.append({'environment':env,'step':step,'retained_augmented_states_exact':True,
                'retained_real_targets_exact':True,'retained_masks_exact':True,'retained_prior_codes_exact':True})
    Path(args.out).write_text(json.dumps({'checks':results,'all_passed':True},indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
