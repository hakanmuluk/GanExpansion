"""Freeze all three Cube training pairs before generating fresh final scenes."""
import argparse
import json
from pathlib import Path
import shutil
import time
from .legacy import digest_file
from .freeze_candidate import BOOTSTRAP


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    root=Path('independent_composition_artifacts')
    original=root/'frozen_candidates/cube_private_paths_v7'
    for name in ['path_policy.py','private_paths.py']:
        assert digest_file(Path(__file__).parent/name)==digest_file(original/'independent_composition'/name)
    validation=[]
    for seed in [42,43,44]:
        directory=root/('private_paths_cube_extension_v7b' if seed==42 else f'private_paths_cube_replication_val_s{seed}_v7')
        status=json.loads((directory/'status.json').read_text())
        assert all(j['status']=='complete' for j in status['jobs'])
        for kind in ['supported_easy','supported_conflict','coupled_ood']:
            matches=list((directory/'results').glob(f'*{kind}*/summary.json'))
            assert len(matches)==1,(directory,kind,matches)
            summary=json.loads(matches[0].read_text())
            validation.append({'seed':seed,'kind':kind,'summary_path':str(matches[0]),
                               'summary_sha256':digest_file(matches[0]),**summary})
    for kind,target in [('supported_easy',.9),('supported_conflict',.9),('coupled_ood',.8)]:
        rows=[r for r in validation if r['kind']==kind]
        assert len(rows)==3 and sum(r['success'] for r in rows)/3>=target,(kind,rows)
    (out/'independent_composition').mkdir(parents=True);(out/'models').mkdir()
    source={}
    for file in Path(__file__).parent.iterdir():
        if file.is_file() and file.suffix in ['.py','.json']:
            target=out/'independent_composition'/file.name;shutil.copy2(file,target)
            source[file.name]=digest_file(target)
    (out/'run_frozen.py').write_text(BOOTSTRAP)
    pairs=[]
    for seed in [42,43,44]:
        copied=[];originals=[]
        for role in ['route','controller']:
            for skill in ['keepout','waypoint']:
                if seed==42:path=original/'models'/f'{skill}_{role}.pt'
                else:
                    directory=root/f'private_paths_cube_retrain_s{seed}_v7/models'/f'{role}s'/skill
                    assert (directory/'DONE.json').exists();path=directory/'best.pt'
                target=out/'models'/f's{seed}_{skill}_{role}.pt';shutil.copy2(path,target)
                assert digest_file(target)==digest_file(path)
                copied.append(str(target));originals.append(str(path))
        pairs.append({'training_seed':seed,'ordered_weight_paths':copied,
                      'ordered_weight_sha256':[digest_file(p) for p in copied],
                      'original_paths':originals})
    protocol={'environment':'cube','created':time.time(),'scope':'Frozen replicated candidate for fresh final testing; all three training seeds retained, with no test-dependent model selection.',
        'source_sha256':source,'bootstrap_sha256':digest_file(out/'run_frozen.py'),
        'model_pairs':pairs,'validation_before_freeze':validation,
        'base_sha256':digest_file('cube/artifacts/final/base_policy.pt'),
        'inference_configuration':{'iterations':120,'starts':8,'executor':0,'execute':1,'prior':.0001,'lr':.12,'guard_inference':True},
        'methods':['compose','fixed','keepout','waypoint','base'],'evaluation_seed':7141,
        'fresh_scene_seed':1731500001,
        'evaluation_plan':'64 fresh scene identities per stratum, shared across training pairs. Compose, fixed mean, and individual skills evaluated for each pair. Base is identical across pairs and evaluated once. No simulator calls or own/joint geometry scores during policy inference.',
        'control_rate_note':'All new methods execute one control per observation. Original released baselines used eight controls; exact disabled base NN output is retained, but closed-loop rates differ.',
        'privacy':'Only source, weights, metadata and aggregate statistics may be backed up off Colab. Raw arrays and witness trajectories stay on Colab.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'capsule':str(out),'training_pairs':3,'weights':12,'protocol_sha256':digest_file(out/'protocol.json')}))


if __name__=='__main__':main()
