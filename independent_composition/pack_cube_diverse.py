"""Pack independent native training episodes into the ORIGINAL dataset format."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import time
import numpy as np
from cube_addons.data import atomic_npz
from cube_addons.env import make_env,close_env
from cube_addons.probe import case
from .cube_diverse_demonstrations import independent_spec
from .legacy import digest_file


CONTEXT_KEYS=['initial_state','initial_cube','goal','base_z','spec','seed','force_contacts','macro_waypoint','planning_interval']
EPISODE_KEYS=['states','contexts','actions','poses','phase','lengths','new_z']


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True)
    p.add_argument('--val-contexts',type=int,default=32);p.add_argument('--val-seed',type=int,default=1801001)
    args=p.parse_args();raw=Path(args.raw);out=Path(args.out);assert not out.exists();assert (raw/'DONE.json').exists()
    config=json.loads((raw/'config.json').read_text());n=config['args']['contexts'];out.mkdir(parents=True)
    train_seeds=set(range(config['args']['seed'],config['args']['seed']+n))
    val_seeds=set(range(args.val_seed,args.val_seed+args.val_contexts));assert train_seeds.isdisjoint(val_seeds)
    started=time.time();report={'scope':'Same original EpisodeDataset fields; own-success training filter only; validation contexts unfiltered. No joint data/labels.',
        'raw_source_sha256':digest_file(raw/'config.json'),'skills':{},'train_validation_seed_overlap':0}
    env=make_env()
    try:
        for skill in ['keepout','waypoint']:
            folder=out/skill;folder.mkdir();parts={k:[] for k in EPISODE_KEYS+CONTEXT_KEYS};contexts={k:[] for k in CONTEXT_KEYS}
            requested=0;kept=0;successful_bins=np.zeros(config['args']['codes'],dtype=int)
            for path in sorted((raw/skill/'raw').glob('*.npz')):
                meta=json.loads(path.with_suffix('.json').read_text());select=np.array([r['success'] for r in meta['records']],dtype=bool)
                requested+=len(select);kept+=int(select.sum());successful_bins+=select
                with np.load(path) as data:
                    for k in CONTEXT_KEYS:
                        value=data[k];contexts[k].append(value.copy())
                        if select.any():parts[k].append(np.repeat(value[None],int(select.sum()),axis=0))
                    if select.any():
                        for k in EPISODE_KEYS:parts[k].append(data[k][select])
            assert len(contexts['seed'])==n and kept>0
            assert (successful_bins>0).all(),'Lost an entire style stratum; inspect the expert'
            atomic_npz(folder/'train.npz',**{k:np.concatenate(v) for k,v in parts.items()})
            atomic_npz(folder/'train_contexts.npz',**{k:np.stack(v) for k,v in contexts.items()})
            val={k:[] for k in CONTEXT_KEYS}
            for seed in sorted(val_seeds):
                c,style=case(env,seed);c['spec']=independent_spec(c,style,skill,seed)
                c.update(force_contacts=True,macro_waypoint=skill=='waypoint',planning_interval=8)
                for k in CONTEXT_KEYS:val[k].append(np.asarray(c[k]))
            atomic_npz(folder/'val_contexts.npz',**{k:np.stack(v) for k,v in val.items()})
            report['skills'][skill]={'requested_episodes':requested,'successful_training_episodes':kept,
                'own_expert_success_rate':kept/requested,'successful_code_strata':successful_bins.tolist(),
                'train_contexts':n,'validation_contexts':args.val_contexts,
                'train_sha256':digest_file(folder/'train.npz'),'validation_sha256':digest_file(folder/'val_contexts.npz')}
    finally:close_env(env)
    report['seconds']=time.time()-started
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'original_dataset_interface':True,'independent_skills_only':True})+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
