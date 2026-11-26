"""Uncontended matched-observation batch1 timing, after all90 cells finish."""
import argparse,time,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from independent_composition.regularization_models import load_bank
from independent_composition.regularized_latent_composition import RegularizedConfig
from independent_composition.inference_guard import tripwires
from composition_ablation_study.timing import probes as original_probes
from .planning import SingleGaussianComposer
from .freeze import read,write,sha,verify,STEPS,KINDS
from .analyze import csvfile,BOOT,BOOT_SEED

def probes(s,env,seed):
    # Reuse the verified original observed-state/context reconstruction routine.
    selected=[dict(j,setting='h8_n20') for j in s['jobs'] if j['steps']==20 and j['seed']==seed]
    shadow=dict(s,jobs=selected)
    data=original_probes(shadow,env);result=[]
    for q in data:
        kind,i,_,*_=q
        # Full production sampler timing, including the real Gaussian draw.
        # The same per-probe seed makes the batch1 draw identical across N.
        # TwoRoom's native batch64 RNG layout need not match this batch1 draw.
        result.append((q,seed+i*9973))
    return result

def call(policy,probe):
    q,probe_seed=probe;_,_,_,state,ctx,bz,previous=q
    policy.previous=None if previous is None else previous.detach().clone()
    policy.initial_gaussian_override=None
    policy.rng.manual_seed(probe_seed)
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();t=time.perf_counter();start.record()
    with tripwires():action=policy.sample(state,ctx,bz)
    end.record();end.synchronize();return time.perf_counter()-t,start.elapsed_time(end)/1000,action

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    source_hash=sha(__file__);s=read(a.study);verify(s);root=Path(a.study).parent
    assert read(root/'queue/status.json')['state']=='complete' and read(root/'final_audit.json')['complete']
    assert all((Path(j['out'])/'DONE.json').exists() for j in s['jobs'])
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True);torch.set_num_threads(2)
    rng=np.random.default_rng(BOOT_SEED);result=[];fidelity=[];overheads=[]
    for env,e in s['environments'].items():
        bank=load_bank(env,**e['models'],device='cuda');data={seed:probes(s,env,seed) for seed in s['seeds']}
        policies={n:SingleGaussianComposer(bank,replace(RegularizedConfig(**e['reference_config']),starts=1,steps=n),241,record=False) for n in STEPS}
        identities=0
        for n in [0,20,40]:
            recorded=SingleGaussianComposer(bank,policies[n].config,241,record=True)
            for pi in [0,1]:
                _,_,x=call(policies[n],data[s['seeds'][0]][pi]);_,_,y=call(recorded,data[s['seeds'][0]][pi])
                assert torch.equal(x,y) and torch.equal(policies[n].gaussian,recorded.gaussian);identities+=1
        fidelity.append({'environment':env,'actual_observation_action_identity_checks':identities,'all_bitwise_equal':True})
        for policy in policies.values():
            for _ in range(3):call(policy,data[s['seeds'][0]][0])
        measurements={n:[] for n in STEPS}
        for repeat in range(3):
            for seed in s['seeds']:
                for probe in data[seed]:
                    for n in rng.permutation(STEPS):
                        wall,cuda,_=call(policies[n],probe);kind,i,warm,*_=probe[0]
                        measurements[n].append((kind,i,seed,warm,repeat,wall,cuda))
        for n,values in measurements.items():
            wall=np.array([x[5] for x in values]);cuda=np.array([x[6] for x in values]);groups=[]
            for kind in KINDS:
                for i in range(4):
                    groups.append([[x[5] for x in values if x[0]==kind and x[1]==i and x[2]==seed] for seed in s['seeds']])
            array=np.array(groups);assert array.shape==(12,3,6)
            rr=np.random.default_rng(BOOT_SEED);ctx=np.concatenate([rr.integers(4,size=(BOOT,4)),rr.integers(4,size=(BOOT,4))+4,rr.integers(4,size=(BOOT,4))+8],1);seeds=rr.integers(3,size=(BOOT,3))
            boot=np.median(array[ctx[:,:,None],seeds[:,None,:]],axis=(1,2,3))
            result.append({'environment':env,'steps':int(n),'batch_size':1,'independent_contexts':12,'initialization_seeds':3,
                'observed_phases_per_context_seed':2,'repeats_per_phase':3,'warmup_calls':3,'measured_calls':len(values),
                'median_wall_seconds':float(np.median(wall)),'p95_wall_seconds':float(np.quantile(wall,.95)),
                'median_cuda_event_elapsed_seconds':float(np.median(cuda)),
                'median_context_seed_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
                'actor_forward_candidate_equivalents_per_call':2*(int(n)+2),'base_forward_equivalents_per_call':1,
                'input_backward_candidate_equivalents_per_call':int(n),'coarse_grid_candidates':0,'starts':1,
                'scope':'Same actual initial/midpoint N20 observation and previous-code input for every N. Cold calls include production randn with a common per-probe seed; batch1 TwoRoom draws need not equal native batch64 RNG layout. Outcome-independent first4 contexts/category.'})
        recorded=SingleGaussianComposer(bank,policies[20].config,241,record=True);plain=[];logged=[]
        for probe in data[s['seeds'][0]]:
            for which in rng.permutation([0,1]):
                wall,_,_=call(recorded if which else policies[20],probe)
                (logged if which else plain).append(wall)
        overheads.append({'environment':env,'matched_probes':24,'plain_median_seconds':float(np.median(plain)),
            'recorded_median_seconds':float(np.median(logged)),'recorded_to_plain_median_ratio':float(np.median(logged)/np.median(plain))})
        bank.assert_frozen();verify(s)
    old=Path(s['reference_study']).parent/'latency/summary.json';historical=read(old)
    record={'complete':True,'rows':result,'actual_GPU_fidelity_checks':fidelity,'recorder_overhead':overheads,
        'historical_grid_latency_sha256':sha(old),'historical_grid_latency':[r for r in historical['rows'] if r['setting'] in [f'h8_n{n}' for n in STEPS]],
        'historical_reference_note':'Previously reported matched batch1 timing on same L4/software; not remeasured or retuned in this experiment. Observed-state sets differ between studies.',
        'study_sha256':sha(a.study),'timing_source_sha256':source_hash,'raw_episode_data_exported':False}
    assert sha(__file__)==source_hash;write(out/'summary.json',record);csvfile(out/'latency.csv',result)
    print(json.dumps({'complete':True,'rows':len(result),'out':str(out)}))

if __name__=='__main__':main()
