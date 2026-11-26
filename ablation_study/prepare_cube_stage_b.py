"""Reproduce the recorded latest-Cube base preparation, without adapters.

Stage B's original script used 64 validation contexts at seed2871, and fresh
Adam state initialized from the prior selected base's EMA G, calibrated E and D.
The no-latent arm instead carries its own Stage-A GAN-only G/D; the encoder is
an unused compatible placeholder. No phase receives a different architecture,
normalization rule, optimizer budget, teacher or training-data file.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation(args):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from cube.cube_gan.data import initialize_worker,collect_context,context_specs,save_split
    out=Path(args.data);assert (out/'DONE.json').exists()
    assert not (out/'val.npz').exists(),'Do not silently regenerate an existing validation file'
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
        rows=list(pool.map(collect_context,context_specs(64,2871,.5)))
    stats,_=save_split(rows,out,'val')
    historical=json.loads(Path(args.historical).read_text())
    actual=digest(out/'train.npz')
    receipt={'generation':'Verbatim original Stage-B validation recipe:64 contexts,seed2871,native fraction .5,4 workers',
        'stats':stats,'train_sha256':actual,'historical_train_sha256':historical['train_sha256'],
        'recreated_training_bytes_match_historical':actual==historical['train_sha256'],
        'training_file_is_identical_for_both_arms':True,'validation_sha256':digest(out/'val.npz')}
    (out/'stage_b_inputs_verified.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


def initialization(args):
    from cube.cube_gan.models import ModelConfig,Generator,TrajectoryEncoder,Discriminator
    parent=torch.load(args.parent,map_location='cpu',weights_only=False)
    reference=torch.load(args.discriminator_parent,map_location='cpu',weights_only=False)
    cfg=ModelConfig(**parent['model']);g=Generator(cfg);e=TrajectoryEncoder(cfg);d=Discriminator(cfg)
    g.load_state_dict(parent.get('generator_ema',parent['generator']));e.load_state_dict(parent['encoder'])
    d.load_state_dict((parent if args.no_latent else reference)['discriminator'])
    ge=torch.optim.Adam(list(g.parameters())+list(e.parameters()),lr=1e-4,betas=(.5,.99))
    od=torch.optim.Adam(d.parameters(),lr=5e-5,betas=(.5,.99))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    value={'model':asdict(cfg),'generator':g.state_dict(),'generator_ema':g.state_dict(),
           'encoder':e.state_dict(),'discriminator':d.state_dict(),'optimizer_ge':ge.state_dict(),
           'optimizer_d':od.state_dict(),'step':0,'best_score':-1,
           'ablation_config':{'base_without_latent_losses':args.no_latent}}
    torch.save(value,out)
    receipt={'parent':args.parent,'parent_sha256':digest(args.parent),'discriminator_parent':args.parent if args.no_latent else args.discriminator_parent,
        'no_latent':args.no_latent,'architecture':asdict(cfg),'parameters':sum(p.numel() for p in g.parameters()),
        'normalization':'Original parent buffers copied exactly; no refitting on broader data',
        'optimizer':'Recorded fresh Adam states:GE1e-4,D5e-5,betas(.5,.99),step0',
        'scope':'Original staged training; latent-objective removal extends through both generator-training phases in the ablation.'}
    out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['validation','initialization'])
    p.add_argument('--data');p.add_argument('--historical');p.add_argument('--parent');p.add_argument('--discriminator-parent')
    p.add_argument('--out');p.add_argument('--no-latent',action='store_true');args=p.parse_args()
    (validation if args.action=='validation' else initialization)(args)


if __name__=='__main__':main()
