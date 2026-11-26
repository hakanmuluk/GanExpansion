"""Uncontended batch1 latency on fixed, actually observed reference states."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from independent_composition.regularization_models import load_bank,digest_file
from independent_composition.regularized_latent_composition import RegularizedConfig,RegularizedComposer
from independent_composition.inference_guard import tripwires
from .planning import ObservedComposer
from .analyze import KINDS,BOOT,SEED,flat_csv
from .evaluate import verify


class RuntimeComposer(ObservedComposer):
    """Same sampler, without the detached recorder and timing-event wrapper."""
    def sample(self,state,contexts,base_z):
        if self.uses_original_sample():return RegularizedComposer.sample(self,state,contexts,base_z)
        return self._select_without_adam(state,contexts,base_z)


def probes(study,environment):
    result=[]
    e=study['environments'][environment]
    for kind in KINDS:
        job=next(j for j in study['jobs'] if j['environment']==environment and j['setting']=='h8_n20' and j['category']==kind)
        folder=Path(job['out']);plans=json.loads((folder/'planning.json').read_text())
        rows=json.loads((folder/'records.json').read_text())
        if environment=='tworoom':
            from tworoom_gan.env import observation
            from tworoom_composition.geometry import contexts,update
            with np.load(folder/'trajectories.npz') as data:
                for i in range(4):
                    active=[p for p in plans if p['step']<rows[i]['steps']]
                    for pi in [0,len(active)//2]:
                        t=active[pi]['step'];device='cuda'
                        tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
                        case={k:tensor(data[k][i:i+1]) for k in ['start','goal','keepout','waypoint','base_z']}
                        visited=torch.zeros(1,dtype=torch.bool,device=device)
                        violated=visited.clone();wall=visited.clone();early=visited.clone()
                        for step in range(t):
                            visited,violated,wall,early,_=update(tensor(data['positions'][i:i+1,step]),
                                tensor(data['positions'][i:i+1,step+1]),case,visited,violated,wall,early)
                        pos=tensor(data['positions'][i:i+1,t]);previous=tensor(data['actions'][i:i+1,t-1]) if t else torch.zeros_like(pos)
                        ctx=contexts(pos,case,visited,torch.full((1,),float(t),device=device),previous)
                        s=observation(pos,case['goal']);bz=case['base_z']
                        prev=tensor(active[pi-1]['selected_existing_inputs'][i:i+1]) if pi else None
                        result.append((kind,i,pi>0,s,ctx,bz,prev))
        else:
            for i in range(4):
                active=plans[i]
                with np.load(folder/f'rollout_{i:04d}.npz') as data:
                    for pi in [0,len(active)//2]:
                        t=active[pi]['step'];tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cuda')
                        s=tensor(data['states'][t:t+1]);ctx=tuple(tensor(data['contexts'][t:t+1,j]) for j in [0,1]);bz=tensor(data['base_z'][None])
                        prev=tensor(active[pi-1]['selected_existing_inputs']) if pi else None
                        result.append((kind,i,pi>0,s,ctx,bz,prev))
    assert len(result)==24
    return result


def time_call(policy,probe):
    _,_,_,state,ctx,bz,previous=probe
    policy.previous=None if previous is None else previous.detach().clone()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();started=time.perf_counter();start.record()
    with tripwires():actions=policy.sample(state,ctx,bz)
    end.record();end.synchronize();wall=time.perf_counter()-started
    return wall,start.elapsed_time(end)/1000,actions


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();source_sha256=digest_file(__file__);study=json.loads(Path(a.study).read_text());verify(study)
    queue=json.loads((Path(a.study).parent/'queue/status.json').read_text())
    assert queue['state']=='complete' and all((Path(j['out'])/'DONE.json').exists() for j in study['jobs'])
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True);torch.set_num_threads(2)
    output=[];overhead=[];work_validation=[];rng=np.random.default_rng(SEED)
    for env,e in study['environments'].items():
        bank=load_bank(env,**e['models'],device='cuda');reference=RegularizedConfig(**e['reference_config'])
        data=probes(study,env)
        settings=['h8_n0','h8_n5','h8_n10','h8_n20','h8_n40','h4_n20','h12_n20','h16_n20','h8_n0_parameterized','random_budget20']
        policies={}
        for setting in settings:
            random=setting=='random_budget20';key='h8_n20' if random else setting
            c=RegularizedConfig(**json.loads(Path(study['configurations'][env+'/'+key]['path']).read_text()))
            policies[setting]=RuntimeComposer(bank,c,241,search='random' if random else 'adam',
                random_candidates=e['random_compute_budget']['random_candidates_after_unchanged_grid'] if random else 0,
                record=False,zero_mode='original_parameterization' if setting.endswith('parameterized') else 'exact_grid')
        # Actual runtime-vs-recorder identity checks for representative inputs.
        for key in ['h8_n20','h8_n0','random_budget20']:
            runtime=policies[key];probe=data[0]
            observed=ObservedComposer(bank,runtime.config,241,search=runtime.search,
                random_candidates=runtime.random_candidates,zero_mode=runtime.zero_mode)
            _,_,x=time_call(runtime,probe);_,_,y=time_call(observed,probe)
            assert torch.equal(x,y)
            # Reset random generators after the fidelity probe.
            runtime.rng.manual_seed(241)
        for policy in policies.values():
            for _ in range(3):time_call(policy,data[0])
        observations={key:[] for key in settings}
        for repetition in range(3):
            for pi,probe in enumerate(data):
                for key in rng.permutation(settings):
                    wall,cuda,_=time_call(policies[key],probe)
                    observations[key].append((probe[0],probe[1],probe[2],wall,cuda))
        for key,values in observations.items():
            wall=np.array([v[3] for v in values]);cuda=np.array([v[4] for v in values])
            # Resample whole contexts, retaining both phases and all repeats.
            groups=[np.array([v[3] for v in values if v[0]==kind and v[1]==i]) for kind in KINDS for i in range(4)]
            boot=[];brng=np.random.default_rng(SEED)
            for _ in range(BOOT):
                indices=np.r_[brng.integers(0,4,4),brng.integers(4,8,4),brng.integers(8,12,4)]
                boot.append(np.median(np.concatenate([groups[i] for i in indices])))
            output.append({'environment':env,'setting':key,'batch_size':1,'independent_contexts':12,
                'measured_calls':len(values),'phases_per_context':2,'timing_repeats_per_phase':3,'warmup_calls':3,
                'median_wall_seconds':float(np.median(wall)),'p95_wall_seconds':float(np.quantile(wall,.95)),
                'mean_wall_seconds':float(wall.mean()),'median_cuda_event_elapsed_seconds':float(np.median(cuda)),
                'median_context_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
                'probe_selection':'First4 contexts/category, initial and midpoint observed reference replans; no outcome filtering.',
                'history':'Same reference previous-code candidate at a matched warm observation; actual rollouts retain each method own previous solution.'})
        original=RegularizedComposer(bank,reference,241);observed=ObservedComposer(bank,reference,241)
        plain=[];recorded=[]
        for probe in data:
            for first in rng.permutation([0,1]):
                policy=[original,observed][first];wall,_,_=time_call(policy,probe)
                if first:recorded.append(wall);_ = observed.last_plan
                else:plain.append(wall)
        overhead.append({'environment':env,'matched_observations':24,'plain_median_seconds':float(np.median(plain)),
            'recorded_median_seconds':float(np.median(recorded)),
            'median_ratio_recorded_to_plain':float(np.median(recorded)/np.median(plain)),
            'note':'Detached trace conversion occurs after the timing boundary in native evaluators. CUDA-event elapsed time includes host launch gaps, not only kernel busy time.'})
        from .preflight import linear_flops
        pairs=[]
        for index in [0,1,8,9,16,17]:
            probe=data[index]
            f_adam,_=linear_flops(lambda:time_call(policies['h8_n20'],probe))
            f_random,_=linear_flops(lambda:time_call(policies['random_budget20'],probe))
            pairs.append((f_adam,f_random,abs(f_adam-f_random)/f_adam))
        work_validation.append({'environment':env,'observed_probes_profiled':6,'categories':3,'cold_and_midpoint_phases':True,
            'maximum_relative_matrix_work_mismatch':max(v[2] for v in pairs),
            'all_within_one_percent':all(v[2]<.01 for v in pairs),
            'adam20_matrix_flops_min':min(v[0] for v in pairs),'adam20_matrix_flops_max':max(v[0] for v in pairs),
            'random_matrix_flops_min':min(v[1] for v in pairs),'random_matrix_flops_max':max(v[1] for v in pairs)})
        bank.assert_frozen();verify(study)
    record={'complete':True,'rows':output,'instrumentation_overhead':overhead,'study_sha256':digest_file(a.study),
        'source_sha256':source_sha256,'raw_episode_data_exported':False,'matched_work_validation':work_validation,
        'measurement_scope':'No native evaluation process concurrently active; paired observed-state batch1 benchmark, separate from native batch64 TwoRoom evaluation.'}
    assert digest_file(__file__)==source_sha256,'Timing source changed while executing'
    (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n');flat_csv(out/'batch1_latency.csv',output)
    print(json.dumps({'complete':True,'settings':len(output),'out':str(out)}))

if __name__=='__main__':main()
