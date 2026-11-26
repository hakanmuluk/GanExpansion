"""Measure scalar tensor-only planning and repeated controller-call latency."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from .path_policy import load_route,PrivatePathPolicy
from .private_paths import load_controller
from .inference_guard import GuardedPolicy
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['cube','tworoom'],required=True)
    p.add_argument('--routes',nargs=2,required=True);p.add_argument('--controllers',nargs=2,required=True)
    p.add_argument('--supports',nargs=2);p.add_argument('--support-weight',type=float,default=0.)
    p.add_argument('--executor',type=int,choices=[0,1]);p.add_argument('--protocol')
    p.add_argument('--observations',required=True);p.add_argument('--out',required=True)
    p.add_argument('--repeats',type=int,default=12);p.add_argument('--require-exclusive-gpu',action='store_true')
    args=p.parse_args();torch.set_num_threads(2)
    if args.repeats<1:p.error('At least one measured repeat is required')
    if args.executor is None:args.executor=0 if args.env=='cube' else 1
    if args.env=='cube' and (args.supports or args.support_weight):p.error('Cube has no domain heads')
    if args.env=='tworoom' and (not args.supports or not args.protocol):
        p.error('Selected TwoRoom timing requires both private domain heads and the frozen protocol')
    weights=[*args.routes,*args.controllers,*(args.supports or [])]
    if args.protocol:
        protocol=json.loads(Path(args.protocol).read_text())
        assert protocol['environment']==args.env
        assert [digest_file(path) for path in weights] in [x['ordered_weight_sha256'] for x in protocol['model_pairs']]
        expected={'iterations':120,'starts':8,'executor':args.executor,'prior':.0001,'lr':.12}
        if args.env=='tworoom':
            expected.update(agreement='pointwise',cycle_mode='none',cycle_weight=0.,code_bound=3.,support_weight=args.support_weight)
        assert all(protocol['inference_configuration'][key]==value for key,value in expected.items())
        for name in ['path_policy.py','private_paths.py','tworoom_paths.py','tworoom_path_policy.py','path_support.py','inference_guard.py']:
            assert digest_file(Path(__file__).parent/name)==protocol['source_sha256'][name],('Policy source changed',name)
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    if args.env=='cube':
        controller_loader=load_controller;Policy=PrivatePathPolicy
        with np.load(args.observations) as d:
            state=torch.as_tensor(d['states'][:1],device='cuda',dtype=torch.float32)
            contexts=tuple(torch.as_tensor(d['contexts'][:1,j],device='cuda',dtype=torch.float32) for j in [0,1])
            bz=torch.as_tensor(d['base_z'],device='cuda',dtype=torch.float32)[None]
    else:
        from .tworoom_paths import load_controller as controller_loader
        from .tworoom_path_policy import TwoRoomPathPolicy as Policy
        from tworoom_gan.env import observation
        from tworoom_composition.geometry import contexts as make_contexts
        with np.load(args.observations) as d:
            case={k:torch.as_tensor(d[k][:1],device='cuda',dtype=torch.float32) for k in ['start','goal','keepout','waypoint','base_z']}
        state=observation(case['start'],case['goal']);bz=case['base_z']
        contexts=make_contexts(case['start'],case,torch.zeros(1,device='cuda',dtype=torch.bool),0)
    routes=[load_route(path)[0] for path in args.routes]
    controllers=[controller_loader(path)[0] for path in args.controllers]
    supports=[]
    if args.supports:
        from .path_support import load_path_support
        for skill,path,route in zip(['keepout','waypoint'],args.supports,args.routes):
            model,saved=load_path_support(path)
            assert saved['skill']==skill and saved['config']['route_sha256']==digest_file(route)
            supports.append(model)
    process_query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                                 check=True,capture_output=True,text=True,timeout=10)
    other_processes=sorted({int(line.strip()) for line in process_query.stdout.splitlines() if line.strip()}-{os.getpid()})
    if args.require_exclusive_gpu and other_processes:
        raise RuntimeError(f'Latency measurement requires an idle GPU; other compute PIDs: {other_processes}')
    initial=[];following=[]
    extra={'supports':supports,'support_weight':args.support_weight,'agreement':'pointwise',
           'cycle_mode':'none','cycle_weight':0.,'code_bound':3.} if args.env=='tworoom' else {}
    for repeat in range(args.repeats+2):
        policy=GuardedPolicy(Policy(routes,controllers,executor=args.executor,steps=120,starts=8,seed=901+repeat,**extra))
        torch.cuda.synchronize();start=time.perf_counter();value=policy.sample(state,contexts,bz)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        assert torch.isfinite(value).all()
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(20):value=policy.sample(state,contexts,bz)
        torch.cuda.synchronize();steady=(time.perf_counter()-start)/20
        if repeat>=2:initial.append(elapsed);following.append(steady)
    result={'environment':args.env,'scalar_batch_size':1,'repeats':args.repeats,'discarded_warmup_repeats':2,
        'iterations':120,'starts':8,'initial_planning_seconds':{'median':float(np.median(initial)),'p95':float(np.quantile(initial,.95))},
        'executor':args.executor,'support_weight':args.support_weight,'cycle_mode':'none',
        'following_controller_seconds':{'median':float(np.median(following)),'p95':float(np.quantile(following,.95))},
        'cuda_device':torch.cuda.get_device_name(),'torch':torch.__version__,
        'other_gpu_compute_pids_at_start':other_processes,'exclusive_gpu_required':args.require_exclusive_gpu,
        'simulator_queries_per_policy_call':0,'inference_tripwires_enabled':True,
        'weight_sha256':{path:digest_file(path) for path in weights},
        'protocol_sha256':digest_file(args.protocol) if args.protocol else None,
        'measurement_source_sha256':digest_file(__file__),
        'scope':'CUDA-synchronized scalar measurements, including frozen neural optimization and tripwire overhead; excludes model loading and environment stepping. Record concurrent jobs separately when interpreting timing.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
