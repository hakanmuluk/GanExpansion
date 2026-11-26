"""Run unchanged native evaluators with a frozen, observed composition planner."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from independent_composition.regularization_models import load_bank,digest_file
from independent_composition.regularized_latent_composition import RegularizedConfig
from independent_composition.inference_guard import GuardedPolicy
from independent_composition.evaluate_diffusion import evaluate_tworoom,evaluate_cube,summarize
from .planning import ObservedComposer,SingleSkillControl


def verify(study):
    for group in ['original_hashes','source_hashes']:
        for path,digest in study[group].items():assert digest_file(path)==digest,path


def reference_check(folder,reference,environment,count):
    old=json.loads((reference/'records.json').read_text());new=json.loads((folder/'records.json').read_text())
    keys=['success','goal_success','keepout_satisfied','waypoint_satisfied','steps']
    assert len(old)==len(new)==count
    assert all(a[k]==b[k] for a,b in zip(old,new) for k in keys),'Original reference outcomes changed'
    maxima={};pairs=[('trajectories.npz',['positions','actions','lengths'])] if environment=='tworoom' else [
        (f'rollout_{i:04d}.npz',['states','contexts','poses','actions','continuous_actions','micro_cube']) for i in range(count)]
    for name,fields in pairs:
        with np.load(folder/name) as a,np.load(reference/name) as b:
            for key in fields:
                assert a[key].shape==b[key].shape,(name,key)
                error=float(np.max(np.abs(a[key]-b[key])))
                maxima[key]=max(maxima.get(key,0),error)
                assert error<=1e-6,('Reference action/state changed',name,key,error)
    return {'episodes':count,'outcomes_and_lengths_exact':True,'maximum_absolute_differences':maxima,
            'scope':'Entire original reference rollouts, not only initial action probes.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--job',required=True)
    args=p.parse_args();study=json.loads(Path(args.study).read_text());verify(study)
    job=next(r for r in study['jobs'] if r['id']==args.job);env=job['environment'];e=study['environments'][env]
    out=Path(job['out']);assert not out.exists();out.mkdir(parents=True)
    torch.set_num_threads(2)
    c=RegularizedConfig(**json.loads(Path(job['config']).read_text()))
    bank=load_bank(env,**e['models'],device='cuda')
    options=SimpleNamespace(env=env,tasks=e['tasks'][job['category']],count=64,execute=8,seed=job['seed'],device='cuda')
    (out/'config.json').write_text(json.dumps({'job':job,'composition':vars(c),'models':bank.files,
        'task_sha256':digest_file(options.tasks),'study_sha256':digest_file(args.study),
        'original_evaluator_sha256':digest_file(__import__('independent_composition.evaluate_diffusion',fromlist=['x']).__file__),
        'scope':'Unchanged original evaluators, constraints, termination and executor; detached planner diagnostics only.'},indent=2)+'\n')
    def factory(a,unused,base,seed):
        if job['search']=='single':return GuardedPolicy(SingleSkillControl(bank,job['skill']))
        return GuardedPolicy(ObservedComposer(bank,c,seed,search=job['search'],random_candidates=job.get('random_candidates',0),
                                             zero_mode=job.get('zero_mode','exact_grid')))
    start=time.time()
    rows=(evaluate_tworoom if env=='tworoom' else evaluate_cube)(options,[],bank.k.base,out,factory)
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    measured=summarize(rows);measured['total_seconds']=time.time()-start
    (out/'summary.json').write_text(json.dumps(measured,indent=2)+'\n')
    proof=None
    if env=='tworoom':
        from tworoom_gan.env import make_reference
        from tworoom_composition.audit import replay
        native=make_reference()
        try:result=replay(out,native)
        finally:native.close()
        assert result['all_match'] and result['episodes']==64
        proof={'episodes':64,'all_positions_and_outcomes_match':True,
            'maximum_position_error_pixels':max(r['max_position_error'] for r in result['rows'])}
        (out/'native_replay_aggregate.json').write_text(json.dumps(proof,indent=2)+'\n')
    if job.get('reference_control'):
        r=reference_check(out,Path(e['old_evaluations'])/job['category'],env,64)
        (out/'original_reference_replay.json').write_text(json.dumps(r,indent=2)+'\n')
    bank.assert_frozen();verify(study)
    (out/'DONE.json').write_text(json.dumps({'episodes':64,'all_weights_frozen':True,
        'original_sources_models_banks_unchanged':True,'inference_tripwires_enabled':True,
        'inference_environment_queries':0,'native_tworoom_replay':proof},indent=2)+'\n')
    print(json.dumps({'job':job['id'],'summary':measured}),flush=True)

if __name__=='__main__':main()
