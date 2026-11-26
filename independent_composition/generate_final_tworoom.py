"""Fresh feasible TwoRoom scenes, native-certified after a complete model freeze."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=64);args=p.parse_args();torch.set_num_threads(2)
    protocol=json.loads(Path(args.protocol).read_text())
    assert protocol['environment']=='tworoom' and protocol['created']<time.time()
    assert args.count>=protocol['minimum_scenes_per_stratum']
    assert args.count==protocol['scene_count']
    assert digest_file('artifacts/final/base_policy.pt')==protocol['base_sha256']
    root=Path(args.out)
    if root.exists():raise FileExistsError('Fresh final scenes required')
    root.mkdir(parents=True)
    from tworoom_gan.evaluate import load_checkpoint
    from tworoom_gan.env import make_reference,reset_reference
    from tworoom_composition.data import build
    from tworoom_composition.audit import monitor
    base,encoder,_=load_checkpoint('artifacts/final/base_policy.pt','cuda')
    base.eval().requires_grad_(False);encoder.eval().requires_grad_(False)
    manifest={'protocol_sha256':digest_file(args.protocol),'created':time.time(),
        'selection':'Geometry and independent native witness only; no learned-policy outcome filtering',
        'construction_note':protocol['scene_construction'],'generator_sha256':digest_file(__file__),'strata':{}}
    env=make_reference()
    try:
        for k,kind in enumerate(['supported_easy','supported_conflict','coupled_ood']):
            offset=protocol['fresh_scene_seed']+k*100000
            # Predeclared reserve pool; selection sees only expert feasibility.
            cases,proofs,attempts=build(base,2*args.count,offset,kind,'cuda')
            accepted=[];certification=[];folder=root/'witnesses'/kind;folder.mkdir(parents=True)
            for case,proof in zip(cases,proofs):
                reset_reference(env,case['start'],case['goal']);positions=[case['start']];premature=False
                for t,action in enumerate(proof['actions']):
                    observed,_,done,_,_=env.step(action);positions.append(np.asarray(observed[:2]).copy())
                    premature|=bool(done and t+1<len(proof['actions']))
                actual=np.asarray(positions);measured=monitor(actual,case)
                error=float(np.max(np.abs(actual-proof['positions'])))
                valid=bool(measured['success'] and not premature and error<5e-5)
                certification.append({'seed':int(case['seed']),'native_witness_success':valid,'maximum_position_error_px':error})
                if not valid:continue
                index=len(accepted);accepted.append(case)
                np.savez_compressed(folder/f'{index:04d}.npz',positions=actual,actions=proof['actions'],nodes=proof['nodes'])
                if len(accepted)==args.count:break
            assert len(accepted)==args.count,'Insufficient independently native-certified scenes'
            arrays={name:np.stack([case[name] for case in accepted]) for name in accepted[0]}
            target=root/f'final_{kind}.npz';np.savez_compressed(target,**arrays)
            manifest['strata'][kind]={'path':str(target),'count':len(accepted),'sha256':digest_file(target),
                'generation_seed':offset,'geometry_attempts':attempts,'native_certification':certification,
                'reserve_pool':2*args.count,'selection':'First64 native-certified expert-feasible cases; no evaluated composition outcome involved.'}
            (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps({'stratum':kind,'native_certified_scenes':len(accepted)}),flush=True)
    finally:env.close()
    assert digest_file('artifacts/final/base_policy.pt')==protocol['base_sha256']
    (root/'DONE.json').write_text(json.dumps({'complete':True,'strata':3})+'\n')


if __name__=='__main__':main()
