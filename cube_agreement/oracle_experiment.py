"""Representation diagnostic: agreement of independently constrained teachers."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from .manifolds import oracle,segment_gap,geometry_cases,world_path,FREE


def solve(geometry,starts=12,steps=120,seed=42,prior_weight=1e-5):
    device=geometry.device;n=len(geometry);rng=torch.Generator(device=device).manual_seed(seed)
    g=geometry[:,None].expand(-1,starts,-1).reshape(-1,10)
    common=torch.randn((n,starts,FREE),device=device,generator=rng)*.5
    k=torch.nn.Parameter(torch.cat([common,torch.linspace(-torch.pi,torch.pi,starts+1,device=device)[:-1][None,:,None].expand(n,-1,-1)],-1).reshape(-1,FREE+1))
    w=torch.nn.Parameter(common.clone().reshape(-1,FREE))
    optimizer=torch.optim.Adam([k,w],lr=.12);best=None;best_loss=None
    for step in range(steps+1):
        pk,pw=oracle('keepout',k,g),oracle('waypoint',w,g)
        disagreement=((pk-pw)/.1).square().mean((1,2));loss=disagreement+prior_weight*(k[:,:FREE].square().mean(-1)+w.square().mean(-1))
        with torch.no_grad():
            values,indices=loss.reshape(n,starts).min(-1);ids=torch.arange(n,device=device)*starts+indices
            candidate=torch.cat([k[ids],w[ids]],-1).clone()
            if best is None:best=candidate;best_loss=values.clone()
            else:
                update=values<best_loss;best=torch.where(update[:,None],candidate,best);best_loss=torch.minimum(values,best_loss)
        if step==steps:break
        optimizer.zero_grad();loss.sum().backward();optimizer.step()
        with torch.no_grad():k[:,:FREE].clamp_(-3,3);w.clamp_(-3,3)
    with torch.no_grad():
        pk=oracle('keepout',best[:,:FREE+1],geometry);pw=oracle('waypoint',best[:,FREE+1:],geometry)
        return best,pk,pw,best_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--banks',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=16);p.add_argument('--steps',type=int,default=120);p.add_argument('--starts',type=int,default=12);args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=[];started=time.perf_counter()
    for bank in args.banks:
        with np.load(bank) as f:cases=[{k:f[k][i] for k in f.files} for i in range(min(args.count,len(f['seed'])))]
        geometry=geometry_cases(cases,'cuda');before=oracle('waypoint',torch.zeros((len(cases),FREE),device='cuda'),geometry)
        torch.cuda.synchronize();t=time.perf_counter();codes,k,w,loss=solve(geometry,args.starts,args.steps);torch.cuda.synchronize();seconds=time.perf_counter()-t
        kw=segment_gap(k,geometry,'waypoint');kk=segment_gap(k,geometry,'keepout');wk=segment_gap(w,geometry,'keepout');ww=segment_gap(w,geometry,'waypoint');bk=segment_gap(before,geometry,'keepout')
        for i,case in enumerate(cases):
            row={'kind':str(case['kind']),'seed':int(case['seed']),'index':i,'execute_K_geometry_success':bool(kk[i]>=-1e-7 and kw[i]<=0),'execute_W_geometry_success':bool(wk[i]>=-1e-7 and ww[i]<=0),'K_keepout_gap_m':float(kk[i]),'K_waypoint_gap_m':float(kw[i]),'W_keepout_gap_m':float(wk[i]),'W_waypoint_gap_m':float(ww[i]),'initial_W_keepout_gap_m':float(bk[i]),'path_disagreement':float((((k[i]-w[i])/.1)**2).mean()),'loss':float(loss[i]),'seconds_per_case_amortized':seconds/len(cases)}
            rows.append(row);np.savez_compressed(out/f"{case['kind']}_{i:04d}.npz",**case,geometry=geometry[i].cpu().numpy(),codes=codes[i].cpu().numpy(),path_K=world_path(k[i].cpu().numpy(),case),path_W=world_path(w[i].cpu().numpy(),case))
        print(bank,'K',sum(r['execute_K_geometry_success'] for r in rows if r['kind']==str(cases[0]['kind'])),'W',sum(r['execute_W_geometry_success'] for r in rows if r['kind']==str(cases[0]['kind'])), 'of',len(cases),'seconds',seconds,flush=True)
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'config.json').write_text(json.dumps({'args':vars(args),'scope':'Analytic own-skill teacher diagnostic only. Not neural evidence, not physical success. Sole composition loss is trajectory agreement plus generic latent prior. No joint witness or native task cost used.','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2)+'\n');(out/'DONE.json').write_text(json.dumps({'complete':True,'cases':len(rows),'seconds':time.perf_counter()-started}))


if __name__=='__main__':main()
