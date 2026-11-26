"""Freeze a source/weight capsule before broader validation or final testing."""
import argparse
import json
from pathlib import Path
import shutil
from .legacy import digest_file


BOOTSTRAP='''"""Run frozen experiment code from the GanExpansion repository root."""
from pathlib import Path
import runpy
import sys

snapshot=Path(__file__).resolve().parent
repository=Path.cwd().resolve()
if not (repository/'cube').is_dir():
    raise SystemExit('Run from the GanExpansion repository root')
sys.path.insert(0,str(snapshot))
sys.path.insert(1,str(repository))
if len(sys.argv)<2 or '.' in sys.argv[1] or '/' in sys.argv[1]:
    raise SystemExit('Pass an independent_composition module basename')
module='independent_composition.'+sys.argv[1]
sys.argv=[module,*sys.argv[2:]]
runpy.run_module(module,run_name='__main__',alter_sys=True)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--env',choices=['cube','tworoom'],default='cube')
    p.add_argument('--agreement',choices=['pointwise','chamfer','hybrid'],default='pointwise')
    p.add_argument('--executor',type=int,choices=[0,1],default=0)
    p.add_argument('--encoders',nargs=2);p.add_argument('--supports',nargs=2)
    p.add_argument('--cycle-mode',default='none');p.add_argument('--cycle-weight',type=float,default=0.)
    p.add_argument('--support-weight',type=float,default=0.);p.add_argument('--code-bound',type=float,default=3.)
    p.add_argument('--routes',nargs=2,required=True);p.add_argument('--controllers',nargs=2,required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Use a fresh immutable candidate directory')
    (out/'independent_composition').mkdir(parents=True);(out/'models').mkdir()
    source={}
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ['.py','.json']:
            target=out/'independent_composition'/path.name;shutil.copy2(path,target)
            source[path.name]=digest_file(target)
    (out/'run_frozen.py').write_text(BOOTSTRAP)
    weights={}
    for kind,paths in [('route',args.routes),('controller',args.controllers),
                       ('encoder',args.encoders or []),('support',args.supports or [])]:
        for skill,path in zip(['keepout','waypoint'],paths):
            original=Path(path)
            assert (original.parent/'DONE.json').exists(),'Training has not completed'
            target=out/'models'/f'{skill}_{kind}.pt';shutil.copy2(original,target)
            assert digest_file(original)==digest_file(target)
            weights[f'{skill}_{kind}']={'path':str(target),'sha256':digest_file(target),'original':str(original)}
    record={'scope':'Frozen candidate for expanded development validation. Final tests remain unopened.',
            'source_sha256':source,'bootstrap_sha256':digest_file(out/'run_frozen.py'),'models':weights,
            'environment':args.env,
            'base_sha256':digest_file('cube/artifacts/final/base_policy.pt' if args.env=='cube' else 'artifacts/final/base_policy.pt'),
            'selected_configuration':{'iterations':120,'starts':8,'executor':args.executor,'execute':1,'prior':.0001,'lr':.12,'guard_inference':True,'agreement':args.agreement,
                'cycle_mode':args.cycle_mode,'cycle_weight':args.cycle_weight,'code_bound':args.code_bound,'support_weight':args.support_weight},
            'model_selection':'Pilot validation only; no per-scene simulator search or joint task-cost selection.',
            'privacy':'This capsule contains Python source, configurations and model weights. It contains no raw trajectory arrays or training data.'}
    (out/'freeze.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'capsule':str(out),'models':len(weights),'source_files':len(source)}))


if __name__=='__main__':main()
