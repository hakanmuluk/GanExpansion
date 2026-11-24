"""Paired native old/new executor equivalence on original held-out single tasks."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from cube_addons.env import make_env,close_env
from cube_addons.evaluate import load_contexts
from cube_addons.rollout import rollout as original_rollout
from .models import FrozenBank
from .rollout import rollout
from .search import SearchConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--count',type=int,default=2)
    p.add_argument('--out',default='cube_composition_artifacts/executor_parity.json');args=p.parse_args();torch.set_num_threads(2);records=[]
    aenv=make_env();benv=make_env()
    try:
        for model_seed in args.seeds:
            bank=FrozenBank(model_seed,'cuda')
            for skill in ['keepout','waypoint']:
                path=Path(f'cube_addon_artifacts/data_force/{skill}/test_contexts.npz')
                for c in load_contexts(path,args.count):
                    case=dict(c);case['kind']=np.array('supported_easy');middle=(c['initial_cube'][:2]+c['goal'][:2])/2
                    case['keepout']=c['spec'] if skill=='keepout' else np.r_[middle,.20,.022]
                    case['waypoint']=c['spec'] if skill=='waypoint' else np.r_[middle,.20,.022]
                    for value in [-.8,.8]:
                        codes=(value,.8) if skill=='keepout' else (.8,value)
                        original,ot=original_rollout(aenv,bank.k if skill=='keepout' else bank.w,c,c['spec'],skill,[value],force_contacts=True)
                        row,nt,_=rollout(benv,bank,case,SearchConfig(method='fixed',family=skill,fixed_codes=codes,temporal='fixed'))
                        errors={key:float(np.max(np.abs(ot[key]-nt[key]))) for key in ['states','actions','poses']}
                        assert all(error==0 for error in errors.values()),errors
                        assert original['constraint_success']==row[skill+'_satisfied'] and original['released_success']==row['released_success']
                        records.append({'model_seed':model_seed,'skill':skill,'context_seed':int(c['seed']),'code':value,'max_errors':errors,'original_full_skill_success':original['success'],
                            'source_context_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'frozen_files':bank.files})
            bank.assert_frozen()
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'args':vars(args),'pairs':len(records),'native_executions':2*len(records),'records':records,'frozen_unchanged':True},indent=2)+'\n');print('Native executor parity pairs:',len(records),'maximum error:',max(max(r['max_errors'].values()) for r in records))
    finally:close_env(aenv);close_env(benv)


if __name__=='__main__':main()
