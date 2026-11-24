"""Synchronized batch-one planning latency on common held-out control states."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import torch
from .models import FrozenBank
from .search import SearchConfig,LatentSearch
from .action_search import ActionSearch
from .geometry import entry_fraction
from .geometry import contexts
from .runtime import observation


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True);p.add_argument('--contexts',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(2);bank=FrozenBank();priors=json.loads(Path('composition_artifacts/priors/seed42.json').read_text())['skills']
    data=np.load(Path(args.reference)/'trajectories.npz');jobs=json.loads(Path(args.plan).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True);report=[]
    for job in jobs:
        cfg=SearchConfig(**job['search']);samples=[]
        factory=ActionSearch if cfg.method=='action' else LatentSearch
        warm=factory(bank,cfg,priors,41)
        c={k:torch.as_tensor(data[k][0:1],device='cuda') for k in ['start','goal','base_z','keepout','waypoint','seed']}
        # Warm allocator and kernels; these calls are explicitly excluded from latency samples.
        warmup=[]
        for _ in range(3):
            _,_,info=warm.plan(c['start'],c,torch.zeros(1,dtype=torch.bool,device='cuda'),torch.zeros(1,device='cuda'),torch.zeros(1,2,device='cuda'))
            warmup.append(info['seconds'])
        for i in range(min(args.contexts,len(data['start']))):
            planner=warm;planner.previous=None;planner.tail=None;planner.rng.manual_seed(41+i)
            if hasattr(planner,'previous_actions'):planner.previous_actions=None
            c={k:torch.as_tensor(data[k][i:i+1],device='cuda') for k in ['start','goal','base_z','keepout','waypoint','seed']}
            length=int(data['lengths'][i]);indices=sorted(set([0,length//4,length//2,3*length//4]))
            pp=torch.as_tensor(data['positions'][i],device='cuda');visited=torch.zeros(1,dtype=torch.bool,device='cuda')
            for t in indices:
                if t:visited=torch.isfinite(entry_fraction(pp[:t],pp[1:t+1],c['waypoint'][0,:2],c['waypoint'][0,2])).any()[None]
                previous=torch.as_tensor(data['actions'][i,t-1:t],device='cuda') if t else torch.zeros(1,2,device='cuda')
                _,z,info=planner.plan(pp[t:t+1],c,visited,torch.full((1,),float(t),device='cuda'),previous)
                samples.append({'context':i,'step':t,'seconds':info['seconds'],'iterations':info['iterations'],
                    'backward_steps':info['backward_steps'],'calls':info['calls']})
        times=np.array([r['seconds'] for r in samples]);result={'name':job['name'],'config':job['search'],'batch_size':1,
            'samples':samples,'seconds_p50':float(np.median(times)),'seconds_p95':float(np.quantile(times,.95)),
            'seconds_mean':float(times.mean()),'first_call_after_warmup_seconds':samples[0]['seconds'],
            'excluded_warmup_calls':3,'warmup_seconds':warmup,'warmup_total_seconds':sum(warmup),
            'common_state_reference':args.reference,'reference_states_are_not_online_success_evaluation':True}
        report.append(result);(out/'latency.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'cpu_threads':torch.get_num_threads(),'timing':'CUDA synchronize immediately before and after complete planner.plan, including Python overhead.',
            'results':report},indent=2));print(job['name'],result['seconds_p50'],result['seconds_p95'],flush=True)
    # Diagnostic pair forwards are unnecessary for deployment of individual baselines.
    # Measure their minimal actual inference paths separately and keep both timings.
    c={k:torch.as_tensor(data[k][0:1],device='cuda') for k in ['start','goal','base_z','keepout','waypoint','seed']}
    flags=torch.zeros(1,dtype=torch.bool,device='cuda');clock=torch.zeros(1,device='cuda');previous=torch.zeros(1,2,device='cuda');z=torch.full((1,2),.4,device='cuda')
    direct={}
    for family in ['base','keepout','waypoint','sum']:
        samples=[]
        with torch.inference_mode():
            for iteration in range(25):
                torch.cuda.synchronize();start=time.perf_counter();state=observation(c['start'],c['goal'])
                if family=='base':bank.k.base(state,c['base_z'])
                else:
                    ctx=contexts(c['start'],c,flags,clock,previous,bank.dimensions)
                    if family=='keepout':bank.k(state,c['base_z'],z[:,:1],ctx[0])
                    elif family=='waypoint':bank.w(state,c['base_z'],z[:,1:],ctx[1])
                    else:bank.additive(state,c['base_z'],z,ctx)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if iteration>=5:samples.append(elapsed)
        direct[family]={'p50':float(np.median(samples)),'p95':float(np.quantile(samples,.95)),'samples':samples,'batch_size':1,'includes_observation_and_required_context_features':True}
    record=json.loads((out/'latency.json').read_text());record['minimal_frozen_policy_inference_seconds']=direct
    (out/'latency.json').write_text(json.dumps(record,indent=2))
    bank.assert_frozen();(out/'DONE.json').write_text(json.dumps({'complete':True,'methods':len(jobs)}))


if __name__=='__main__':main()
