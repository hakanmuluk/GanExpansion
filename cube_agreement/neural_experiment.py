"""Frozen learned manifold agreement on development scenes; geometry-only stage."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from .neural import Decoder
from .manifolds import geometry_cases,world_path,segment_gap,FREE
from .agreement import solve


def load(path,device='cuda'):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    model=Decoder(checkpoint['skill'],checkpoint['hidden']).to(device);model.load_state_dict(checkpoint['state']);model.eval().requires_grad_(False)
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--keepout',required=True);p.add_argument('--waypoint',required=True);p.add_argument('--banks',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=16);p.add_argument('--starts',type=int,default=12);p.add_argument('--steps',type=int,default=120);p.add_argument('--objective',choices=['path','delta'],default='path');args=p.parse_args();torch.set_num_threads(2)
    models={skill:load(getattr(args,skill)) for skill in ['keepout','waypoint']};out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    config={'args':vars(args),'frozen_files':{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [args.keepout,args.waypoint]},'mechanism':'Only frozen neural decoder agreement plus generic modifier prior. No joint constraint cost, analytic projection, native candidate forecasts or expert path inputs to optimizer.','scope':'Development geometry stage, not physical neural-control success.','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for bank in args.banks:
        with np.load(bank) as f:cases=[{k:f[k][i] for k in f.files} for i in range(min(args.count,len(f['seed'])))]
        geometry=geometry_cases(cases,'cuda');torch.cuda.synchronize();started=time.perf_counter();codes,k,w,loss=solve(models,geometry,args.starts,args.steps,objective=args.objective);torch.cuda.synchronize();seconds=time.perf_counter()-started
        kk,kw=segment_gap(k,geometry,'keepout'),segment_gap(k,geometry,'waypoint');wk,ww=segment_gap(w,geometry,'keepout'),segment_gap(w,geometry,'waypoint')
        for i,case in enumerate(cases):
            row={'kind':str(case['kind']),'seed':int(case['seed']),'index':i,'execute_K_geometry_success':bool(kk[i]>=-1e-7 and kw[i]<=0),'execute_W_geometry_success':bool(wk[i]>=-1e-7 and ww[i]<=0),'K_keepout_gap_m':float(kk[i]),'K_waypoint_gap_m':float(kw[i]),'W_keepout_gap_m':float(wk[i]),'W_waypoint_gap_m':float(ww[i]),'path_disagreement':float(((k[i]-w[i])/.1).square().mean()),'loss':float(loss[i]),'seconds_per_case_amortized':seconds/len(cases)};rows.append(row)
            np.savez_compressed(out/f"{case['kind']}_{i:04d}.npz",**case,geometry=geometry[i].cpu().numpy(),codes=codes[i].cpu().numpy(),path_K=world_path(k[i].cpu().numpy(),case),path_W=world_path(w[i].cpu().numpy(),case))
        selected=[r for r in rows if r['kind']==str(cases[0]['kind'])];print(bank,'K',sum(r['execute_K_geometry_success'] for r in selected),'W',sum(r['execute_W_geometry_success'] for r in selected),'of',len(selected),'seconds',seconds,flush=True)
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'DONE.json').write_text(json.dumps({'complete':True,'contexts':len(rows),'weights_frozen':True}))


if __name__=='__main__':main()
