"""Interactive single-scene action-agreement latency on real observed states."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from .actor_policy import Bank
from .action_model import reference_context,path_context,world_path_tensor
from .action_agreement import refine
from .manifolds import geometry_cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',default='cube_agreement_artifacts/actor_pilot_v3/val_supported_conflict/local_action_e1');p.add_argument('--out',required=True);p.add_argument('--queries',type=int,default=64);p.add_argument('--action-starts',type=int,default=12);args=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;device='cuda';r=Path('cube_agreement_artifacts');bank=Bank(str(r/'decoders/keepout_s42/best.pt'),str(r/'decoders/waypoint_s42/best.pt'),str(r/'action_adapter_v3/best.pt'));root=Path(args.evaluation);plans=json.loads((root/'planning.json').read_text());samples=[];initial=[]
    for i,ep in enumerate(plans):
        with np.load(root/f'rollout_{i:04d}.npz') as f:d={k:f[k] for k in f.files}
        tmp={**d,'contexts':d['contexts'][:,1],'path_done':np.array([q['path_done'] for q in ep])};ref=reference_context(tmp)
        initial.append((i,0,d,ref,ep))
        samples.extend((i,t,d,ref,ep) for t,q in enumerate(ep) if t>0 and q['iterations']==20)
    def query(sample,first=False):
        i,t,d,ref,ep=sample;g=geometry_cases([d],device);s=torch.as_tensor(d['states'][t:t+1],device=device);reference=torch.as_tensor(ref[t:t+1],device=device);cube=torch.as_tensor(d['poses'][t:t+1,3:],device=device);idx=0 if first else ep[t]['path_index'];previous=None if first else torch.as_tensor(d['codes'][t-1:t],device=device);seed=42+9973*i+1009*sum(q['iterations']==20 for q in ep[:t]);start=torch.as_tensor(d['initial_cube'],device=device,dtype=torch.float32)[None];goal=torch.as_tensor(d['goal'],device=device,dtype=torch.float32)[None];bz=torch.as_tensor(d['base_z'],device=device)[None]
        torch.cuda.synchronize();beg=time.perf_counter();z,k,w,metrics=refine(bank,g,d,s,reference,cube,idx,previous,starts=12 if first else args.action_starts,steps=80 if first else 20,seed=seed,continuity_weight=.001)
        with torch.inference_mode():action=bank.actor(s,bz,torch.tensor([[.8]],device=device),path_context(reference,cube,world_path_tensor(w,start,goal),idx))
        torch.cuda.synchronize();return time.perf_counter()-beg
    cold=query(initial[0],True)
    for sample in samples[:3]:query(sample)
    measured=[query(samples[int(i)]) for i in np.linspace(0,len(samples)-1,args.queries,dtype=int)];initial_times=[query(s,True) for s in initial]
    bank.assert_frozen();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);report={'args':vars(args),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'first_cold_initial_query_seconds':cold,'warm_initial_query_median_seconds':float(np.median(initial_times)),'warm_update_query_median_seconds':float(np.median(measured)),'warm_update_query_p95_seconds':float(np.quantile(measured,.95)),'warm_update_query_max_seconds':max(measured),'queries':len(measured),'native_control_period_seconds':.05,'fits_50ms_period_at_p95':bool(np.quantile(measured,.95)<=.05),'fits_working_400ms_median_target':bool(np.median(measured)<=.4),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True),'scope':'Unbatched one-scene latent-only action optimization plus one StyleGAN executor call, with CUDA synchronization. Inputs are real held transport observations. Does not measure physical success. Planning time is not hidden by averaging over a batch.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'DONE.json').write_text(json.dumps({'complete':True}));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
