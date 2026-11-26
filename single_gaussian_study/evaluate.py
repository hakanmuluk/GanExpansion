"""Original native closed loops with the single-start planner."""
import argparse,time,json
from pathlib import Path
from types import SimpleNamespace
import torch
from independent_composition.regularization_models import load_bank
from independent_composition.regularized_latent_composition import RegularizedConfig
from independent_composition.inference_guard import GuardedPolicy
from independent_composition.evaluate_diffusion import evaluate_tworoom,evaluate_cube,summarize
from .planning import SingleGaussianComposer
from .freeze import read,write,sha,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--job',required=True);a=p.parse_args()
    s=read(a.study);verify(s);j=next(j for j in s['jobs'] if j['id']==a.job);e=s['environments'][j['environment']]
    out=Path(j['out']);assert not out.exists();out.mkdir(parents=True)
    torch.set_num_threads(2);c=RegularizedConfig(**read(j['config']));bank=load_bank(j['environment'],**e['models'],device='cuda')
    options=SimpleNamespace(tasks=e['tasks'][j['category']],device='cuda',count=64,execute=8,seed=j['seed'])
    write(out/'config.json',{'job':j,'composition':vars(c),'models':bank.files,'study_sha256':sha(a.study),
        'task_sha256':sha(options.tasks),'original_evaluator_sha256':sha(__import__('independent_composition.evaluate_diffusion',fromlist=['x']).__file__)})
    def factory(a,unused,base,seed):return GuardedPolicy(SingleGaussianComposer(bank,c,seed))
    started=time.perf_counter()
    rows=(evaluate_tworoom if j['environment']=='tworoom' else evaluate_cube)(options,[],bank.k.base,out,factory)
    write(out/'records.json',rows);summary=summarize(rows);summary['total_seconds']=time.perf_counter()-started;write(out/'summary.json',summary)
    native=None
    if j['environment']=='tworoom':
        from tworoom_gan.env import make_reference
        from tworoom_composition.audit import replay
        env=make_reference()
        try:result=replay(out,env)
        finally:env.close()
        assert result['all_match'] and result['episodes']==64
        native={'episodes':64,'all_positions_and_outcomes_match':True,'maximum_position_error_pixels':max(r['max_position_error'] for r in result['rows'])}
        write(out/'native_replay_aggregate.json',native)
    from .audit import audit_cell
    validation=audit_cell(s,j,require_done=False)
    bank.assert_frozen();verify(s)
    write(out/'DONE.json',{'episodes':64,'models_frozen':True,'originals_unchanged':True,'inference_environment_queries':0,
        'single_gaussian_initialization_and_warm_start_audited':True,'validation':validation,'native_replay':native})
    print(json.dumps({'job':j['id'],'completed_episodes':64}),flush=True)

if __name__=='__main__':main()
