"""Run the original base evaluation CLI and label unavailable encoder measures.

The no-latent model retains an inert E only for checkpoint compatibility. Its
legacy E-related diagnostic fields are not interpreted or used as evidence.
Task success, native audits, mode coverage and physical code-control formulas
remain the original evaluation procedures.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=['tworoom','cube'],required=True)
    p.add_argument('--config',required=True);args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    assert not Path(cfg['out']).exists(),'Use an unused output directory; preserve completed evaluations'
    checkpoint_hash=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()
    name='tworoom_gan.evaluate' if args.environment=='tworoom' else 'cube.cube_gan.evaluate'
    module=importlib.import_module(name);argv=[]
    for key,value in cfg.items():
        if value is None or value is False:continue
        argv.append('--'+key.replace('_','-'))
        if value is not True:argv.append(str(value))
    sys.argv=[module.__file__,*argv];module.main()
    out=Path(cfg['out']);filename='metrics.json' if args.environment=='tworoom' else 'summary.json'
    raw=json.loads((out/filename).read_text())
    ck=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    assert hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()==checkpoint_hash,'Checkpoint changed during evaluation'
    training=ck.get('train_config',ck.get('args',{}));available=training.get('rec_weight',1)!=0
    primary=raw.get('independent_gaussian_latents',raw).copy()
    if not available:
        for key in ['rollout_latent_rmse_all','rollout_latent_rmse_successful','rollout_latent_correlations',
                    'closed_loop_latent_rmse','offline']:
            primary.pop(key,None)
    result={'environment':args.environment,'encoder_recovery_applicable':available,
        'checkpoint_sha256':checkpoint_hash,
        'primary_original_gaussian_protocol':primary,'original_cli_configuration':cfg,
        'original_results_file':filename,'native_reference_audit':raw.get('reference_audit'),
        'scope':'Original scores/settings. Legacy placeholder-encoder diagnostics excluded from interpretation for the loss ablation.'}
    (out/'study_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'original_evaluation_complete':True})+'\n')


if __name__=='__main__':main()
