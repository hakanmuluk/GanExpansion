"""Existing action-agreement planning plus private Gaussian/cycle regularization."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import torch
from .regularization_models import load_bank,load_cycle_encoders,digest_file
from .regularized_latent_composition import RegularizedConfig,RegularizedComposer
from .inference_guard import GuardedPolicy
from .evaluate_diffusion import evaluate_tworoom,evaluate_cube,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['env','base','keepout','waypoint','tasks','config','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--encoders',nargs=2);p.add_argument('--runtime')
    p.add_argument('--count',type=int,default=16);p.add_argument('--execute',type=int,default=8)
    p.add_argument('--seed',type=int,default=241);p.add_argument('--device',default='cuda')
    p.add_argument('--freeze');p.add_argument('--bank-manifest');p.add_argument('--method-name')
    a=p.parse_args();torch.set_num_threads(2)
    if a.freeze:
        frozen=json.loads(Path(a.freeze).read_text());manifest=json.loads(Path(a.bank_manifest).read_text())
        assert a.count==frozen['scene_count'] and a.execute==frozen['execute'] and a.seed==frozen['evaluation_seed']
        assert digest_file(a.config)==frozen['configuration_sha256'][a.method_name]
        assert manifest['protocol_sha256']==digest_file(a.freeze) and manifest['created']>frozen['created']
        assert any(v['sha256']==digest_file(a.tasks) and v['count']==a.count for v in manifest['strata'].values())
        for path,sha in frozen['source_files'].items():assert digest_file(path)==sha,path
        for path,sha in frozen['models'].items():assert digest_file(path)==sha,path
    else:assert not Path(a.tasks).name.startswith(('test','final')),'Development only without a freeze'
    c=RegularizedConfig(**json.loads(Path(a.config).read_text()))
    assert a.execute==c.agreement_horizon==8
    bank=load_bank(a.env,a.base,a.keepout,a.waypoint,a.device,a.runtime)
    if c.cycle_weight:
        assert a.encoders;load_cycle_encoders(bank,a.env,a.encoders,c.cycle_horizon)
    else:assert not a.encoders,'Do not change encoder weights for a prior-only/baseline control'
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    names=['regularized_latent_composition.py','regularization_models.py','latent_action_composition.py',
           'unchanged_composition.py','inference_guard.py','evaluate_regularized_latents.py','evaluate_diffusion.py']
    sources={name:digest_file(Path(__file__).parent/name) for name in names}
    (out/'source').mkdir()
    for name in names:shutil.copy2(Path(__file__).parent/name,out/'source'/name)
    (out/'config.json').write_text(json.dumps({'args':vars(a),'composition':asdict(c),'models':bank.files,
        'source_sha256':sources,'task_sha256':digest_file(a.tasks),'scope':__doc__,
        'gaussian_prior':'0.5*(private_z_K**2+private_z_W**2)',
        'cycle':'mean abs(E(current-state repeated, issued proposal, own context repeated, own base)-private_z)',
        'encoders_used_online':bool(c.cycle_weight),'cycle_interface':getattr(bank,'cycle_interface',None)},indent=2)+'\n')
    if a.freeze:
        assert bank.files==frozen['method_models'][a.method_name]
        (out/'frozen_proof.json').write_text(json.dumps({'source_models_config_bank_verified':True,'freeze_sha256':digest_file(a.freeze)},indent=2)+'\n')
    def factory(args,unused,base,seed):return GuardedPolicy(RegularizedComposer(bank,c,seed))
    started=time.time()
    rows=(evaluate_tworoom if a.env=='tworoom' else evaluate_cube)(a,[],bank.k.base,out,factory)
    summary=summarize(rows);summary['total_seconds']=time.time()-started
    bank.assert_frozen();assert all(digest_file(path)==sha for path,sha in bank.files.items())
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'all_weights_frozen':True,'inference_environment_queries':0,
        'inference_tripwires_enabled':True,'new_neural_modules':0,'only_existing_private_encoders_used':bool(c.cycle_weight),
        'private_prior_weight':c.private_prior_weight,'cycle_weight':c.cycle_weight,'cycle_horizon':c.cycle_horizon},indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
