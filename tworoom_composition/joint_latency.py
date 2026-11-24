"""Batch-one minimal inference latency for the separately trained reference."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .joint_reference import JointPolicy
from .geometry import entry_fraction


def main():
    torch.set_num_threads(2);root=Path('composition_artifacts');model=JointPolicy().cuda();model.load_state_dict(torch.load(root/'joint_reference/best.pt',weights_only=False)['model']);model.eval().requires_grad_(False)
    data=np.load(root/'main_seed42/test_supported_easy/structured_task_h64/trajectories.npz');samples=[];warmup=[]
    with torch.inference_mode():
        for i in range(4):
            case={k:torch.as_tensor(data[k][i:i+1],device='cuda') for k in ['start','goal','base_z','keepout','waypoint']};p=torch.as_tensor(data['positions'][i],device='cuda');length=int(data['lengths'][i])
            for t in [0,length//4,length//2,3*length//4]:
                visited=torch.zeros(1,dtype=torch.bool,device='cuda') if t==0 else torch.isfinite(entry_fraction(p[:t],p[1:t+1],case['waypoint'][0,:2],case['waypoint'][0,2])).any()[None]
                for repetition in range(8):
                    torch.cuda.synchronize();started=time.perf_counter();model(p[t:t+1],case,visited,torch.full((1,),float(t),device='cuda'));torch.cuda.synchronize();elapsed=time.perf_counter()-started
                    (warmup if repetition<3 else samples).append(elapsed)
    record={'batch_size':1,'gpu':torch.cuda.get_device_name(),'seconds_p50':float(np.median(samples)),'seconds_p95':float(np.quantile(samples,.95)),
        'samples':samples,'excluded_warmup_seconds':warmup,'includes_observation_and_context':True,'training':'Joint-demonstration trained reference; not zero-shot.'}
    (root/'joint_reference/latency.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['samples','excluded_warmup_seconds']},indent=2))


if __name__=='__main__':main()
