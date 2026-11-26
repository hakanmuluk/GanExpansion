"""Freeze configurations and references to the existing, unmodified task banks."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import torch
from independent_composition.regularized_latent_composition import RegularizedConfig

KINDS=['supported_easy','supported_conflict','coupled_ood']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--original-root',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();root=Path(args.original_root).resolve();out=Path(args.out).resolve()
    assert not out.exists();out.mkdir(parents=True)
    record={'created_utc_epoch':time.time(),'original_root':str(root),'environments':{},'jobs':[],
      'scope':'Controlled existing-model composition ablations; all neural weights frozen, no training or regenerated scenes.',
      'execute':8,'optimization_seed':241,'random_search_seeds':[241,1241,2241],
      'uncertainty_unit':'Independent task context, not optimizer iterations, replans or repeated random seeds.',
      'reuse_note':'These banks were used in the earlier frozen regularization study; new analyses are exploratory reuse, not a fresh confirmatory test.',
      'zero_steps_note':'Exact original coarse-grid argmin, including the previous candidate; bypass only the original bound-logit round-trip when no gradient step is requested.',
      'random_budget_note':'Chosen outcome-blind by measured neural arithmetic FLOPs relative to original20-stepAdam; actual latency and forward/backward work reported separately.',
      'original_hashes':{},'source_hashes':{},'configurations':{},
      'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                     'gpu':torch.cuda.get_device_name(0),'native_packages':{}}}
    from importlib.metadata import version
    for name in ['numpy','mujoco','gymnasium','ogbench','dm-control','scipy','matplotlib']:
        record['environment']['native_packages'][name]=version(name)
    for env in ['tworoom','cube']:
        old=root/'independent_composition_artifacts'/f'regularization_v142_{env}_protocol'
        frozen=json.loads((old/'freeze.json').read_text())
        for key in ['source_files','models']:
            for name,digest in frozen[key].items():
                path=root/name if not Path(name).is_absolute() else Path(name)
                assert sha(path)==digest,('Frozen original changed',path)
                record['original_hashes'][str(path)]=digest
        reference=json.loads((old/'prior.json').read_text());c=RegularizedConfig(**reference)
        assert c.agreement_horizon==8 and c.steps==20 and c.private_prior_weight==.01 and c.cycle_weight==0
        assert frozen['scene_count']==64 and frozen['evaluation_seed']==241 and frozen['execute']==8
        bank=root/'independent_composition_artifacts'/f'regularization_v142_{env}_bank'
        manifest=json.loads((bank/'manifest.json').read_text())
        models={k:str(root/v) for k,v in frozen['role_model_paths'].items()}
        paths={kind:str(bank/f'final_{kind}.npz') for kind in KINDS}
        for kind,path in paths.items():
            assert sha(path)==manifest['strata'][kind]['sha256']
            assert manifest['strata'][kind]['count']==64
            record['original_hashes'][path]=sha(path)
        record['original_hashes'][str(old/'freeze.json')]=sha(old/'freeze.json')
        record['original_hashes'][str(old/'prior.json')]=sha(old/'prior.json')
        record['environments'][env]={'models':models,'tasks':paths,'reference_config':reference,
            'old_freeze':str(old/'freeze.json'),'old_evaluations':str(root/'independent_composition_artifacts'/f'regularization_v142_{env}_eval/prior'),
            'bank_sha256':{kind:sha(path) for kind,path in paths.items()},'contexts_per_category':64,
            'executor':'clip(A_W+A_K-A_B,-1,1)' if env=='tworoom' else 'A_W; issue gripper threshold +/-1'}
        settings=[(f'h{h}_n20',{'agreement_horizon':h}) for h in [4,8,12,16]]
        settings += [(f'h8_n{n}',{'steps':n}) for n in [0,5,10,40]]
        settings += [('h8_n0_parameterized',{'steps':0})]
        for name,change in settings:
            cfg={**reference,**change};delta={k:[reference[k],v] for k,v in cfg.items() if v!=reference[k]}
            assert set(delta)<=({'agreement_horizon'} if name.endswith('n20') else {'steps'})
            path=out/'configs'/env/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(cfg,indent=2)+'\n')
            key=env+'/'+name;record['configurations'][key]={'path':str(path),'sha256':sha(path),'delta':delta}
            for kind in KINDS:
                record['jobs'].append({'id':f'{env}_{name}_{kind}','environment':env,'setting':name,'category':kind,
                    'search':'adam','seed':241,'config':str(path),'out':str(out/'runs'/env/name/kind),
                    'zero_mode':'original_parameterization' if name.endswith('parameterized') else 'exact_grid',
                    'reference_control':name=='h8_n20'})
        for skill in ['keepout','waypoint']:
            setting='diagnostic_single_'+skill
            for kind in KINDS:
                record['jobs'].append({'id':f'{env}_{setting}_{kind}','environment':env,'setting':setting,'category':kind,
                    'search':'single','skill':skill,'seed':241,'config':record['configurations'][env+'/h8_n20']['path'],
                    'out':str(out/'runs'/env/setting/kind),'reference_control':False,
                    'scope':'Additional fixed-u=-0.8 original single-skill control; not part of H/N ablation or a policy-capacity oracle.'})
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ['.py','.txt','.json']:record['source_hashes'][str(path.resolve())]=sha(path)
    (out/'study.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'out':str(out),'initial_adam_cells':len(record['jobs']),'originals_verified':len(record['original_hashes'])}))

if __name__=='__main__':main()
