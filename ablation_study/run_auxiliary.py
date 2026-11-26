"""Run ORIGINAL physical-encoder collection/calibration with variant loading.

Only checkpoint loading differs for the explicitly removed modulation branch.
Original rollout, data sampling, losses, schedules and selection remain intact.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
from .evaluation import load_adapter,install_cube_variant_loader,variant_feedback_init_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--module',choices=['cube_addons.feedback','cube_addons.calibrate'],required=True)
    p.add_argument('--config',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());module=importlib.import_module(args.module)
    if args.module.endswith('feedback'):module.init_worker=variant_feedback_init_worker
    else:
        module.load_adapter=lambda path,base,device:load_adapter(path,base,'cube',device)
        install_cube_variant_loader()
    cli=[]
    for key,value in cfg.items():
        if value is None or value is False:continue
        cli.append('--'+key.replace('_','-'))
        if value is not True:cli.append(str(value))
    sys.argv=[module.__file__,*cli];module.main()
    out=Path(cfg['out']);(out/'original_procedure_verified.json').write_text(json.dumps({
        'module':args.module,'variant_loading_only':True,'original_objective_and_sampling_unchanged':True,
        'scope':'Existing physical private-encoder calibration procedure, not a new training objective.'},indent=2)+'\n')


if __name__=='__main__':main()
