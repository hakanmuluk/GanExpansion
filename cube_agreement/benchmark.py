"""Warm single-scene solver latency after other experiments have finished."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from cube_composition.evaluate import load_cases
from .actor_policy import Bank
from .manifolds import geometry_cases,segment_gap
from .agreement import solve


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=32);args=p.parse_args();torch.set_num_threads(2);device='cuda';bank=Bank('cube_agreement_artifacts/decoders/keepout_s42/best.pt','cube_agreement_artifacts/decoders/waypoint_s42/best.pt','cube_agreement_artifacts/action_adapter_v3/best.pt');cases=load_cases(args.tasks);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for steps,starts in [(120,12),(60,12),(40,12),(60,6)]:
        times=[];success=[];cold=None
        for i in range(args.repeats+3):
            g=geometry_cases([cases[i%len(cases)]],device);torch.cuda.synchronize();t=time.perf_counter();z,k,w,loss=solve(bank.models,g,starts,steps,42+i);torch.cuda.synchronize();seconds=time.perf_counter()-t
            with torch.inference_mode():valid=bool((segment_gap(w,g,'keepout')>=0).all() and (segment_gap(w,g,'waypoint')<=0).all())
            if cold is None:cold=seconds
            if i>=3:times.append(seconds);success.append(valid)
        row={'steps':steps,'starts':starts,'first_call_seconds':cold,'warm_queries':len(times),'warm_median_seconds':float(np.median(times)),'warm_p95_seconds':float(np.quantile(times,.95)),'warm_max_seconds':max(times),'executed_W_path_joint_geometry_fraction':float(np.mean(success)),'scope':'Geometry diagnostic only, not physical success; unbatched one-scene queries with CUDA synchronization and normal weight-freeze checks.'};rows.append(row);print(row,flush=True)
    bank.assert_frozen();report={'args':vars(args),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'runs':rows,'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'DONE.json').write_text(json.dumps({'complete':True}))


if __name__=='__main__':main()
