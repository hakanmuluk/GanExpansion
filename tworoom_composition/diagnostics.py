"""Plausibility and encoder diagnostics on executed composed trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tworoom_gan.env import observation
from .models import FrozenBank
from .geometry import contexts,entry_fraction,wall_clearance
from tworoom_addons.geometry import segment_distance


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--keepout',default='addon_artifacts/final/keepout.pt')
    p.add_argument('--waypoint',default='addon_artifacts/final/waypoint.pt');p.add_argument('--priors',default='composition_artifacts/priors/seed42.json')
    args=p.parse_args();torch.set_num_threads(2);bank=FrozenBank(args.keepout,args.waypoint);priors=json.loads(Path(args.priors).read_text())['skills'];reports=[]
    for run in sorted(Path(args.root).glob('*/*')):
        if not (run/'DONE.json').exists():continue
        cfg=json.loads((run/'config.json').read_text());assert cfg['frozen_files']==bank.files
        data=np.load(run/'trajectories.npz');n=len(data['start']);rows=[]
        with torch.inference_mode():
            for lo in range(0,n,32):
                hi=min(n,lo+32);case={k:torch.as_tensor(data[k][lo:hi],device='cuda') for k in ['start','goal','base_z','keepout','waypoint']}
                positions=torch.as_tensor(data['positions'][lo:hi],device='cuda');actions=torch.as_tensor(data['actions'][lo:hi],device='cuda');lengths=torch.as_tensor(data['lengths'][lo:hi],device='cuda')
                z=torch.as_tensor(data['latents'][lo:hi],device='cuda');b=len(positions)
                # Match the original encoder's 240-control storage exactly.
                pp=positions[:,-1:].expand(-1,241,-1).clone();pp[:,:positions.shape[1]]=positions
                aa=actions.new_zeros((b,240,2));aa[:,:actions.shape[1]]=actions
                valid=torch.arange(240,device='cuda')[None]<lengths[:,None]
                kc=segment_distance(pp[:,:-1],pp[:,1:],case['keepout'][:,None,:2])-case['keepout'][:,None,2]-7
                wc=wall_clearance(pp[:,:-1],pp[:,1:])
                min_keep=kc.masked_fill(~valid,float('inf')).amin(1);min_wall=wc.masked_fill(~valid,float('inf')).amin(1)
                visits=torch.isfinite(entry_fraction(pp[:,:-1],pp[:,1:],case['waypoint'][:,None,:2],case['waypoint'][:,None,2]))
                before=torch.cat([visits.new_zeros((b,1)),visits[:,:-1].long().cumsum(1)>0],1)
                expanded={k:v[:,None].expand(-1,240,*v.shape[1:]) for k,v in case.items()}
                previous=torch.cat([aa.new_zeros((b,1,2)),aa[:,:-1]],1);clock=torch.arange(240,device='cuda')[None].expand(b,-1)
                ctx=contexts(pp[:,:-1],expanded,before,clock,previous,bank.dimensions);states=observation(pp[:,:-1],expanded['goal'])
                inferred=[]
                for encoder,c in zip([bank.ek,bank.ew],ctx):
                    mu,_=encoder(states,aa,c,case['base_z'],lengths);inferred.append(mu[:,0])
                for i in range(b):
                    row={'context':lo+i,'minimum_keepout_clearance_pixels':float(min_keep[i]),'minimum_body_wall_clearance_pixels':float(min_wall[i])}
                    for j,skill in enumerate(['keepout','waypoint']):
                        code=z[i,:int(lengths[i]):8,j];stat=priors[skill];low,high=stat['encoded_quantiles'][1],stat['encoded_quantiles'][-2]
                        initial_bank=code.new_tensor(stat['initial_codes']);mu=code.new_tensor(stat['means']);sd=code.new_tensor(stat['stds'])
                        nll=-torch.logsumexp(-.5*((code[:,None]-mu)/sd).square()-sd.log()-1.612085713764618,-1)
                        row[skill]={'encoded_full_composed_trajectory':float(inferred[j][i]),'code_mean':float(code.mean()),
                            'code_empirical_nll_mean':float(nll.mean()),'code_outside_single_skill_encoded_1_99_fraction':float(((code<low)|(code>high)).float().mean()),
                            'distance_to_single_skill_code_bank_mean':float((code[:,None]-initial_bank).abs().amin(1).mean())}
                    rows.append(row)
        report={'run':str(run),'encoder_interpretation':'Encoder was trained on single-skill trajectories; composed trajectory codes are an out-of-distribution diagnostic, not a validity certificate.',
            'per_context':rows,'summary':{skill:{k:float(np.mean([r[skill][k] for r in rows])) for k in rows[0][skill]} for skill in ['keepout','waypoint']}}
        report['summary']['geometry']={k:{'minimum':min(r[k] for r in rows),'mean':float(np.mean([r[k] for r in rows]))} for k in ['minimum_keepout_clearance_pixels','minimum_body_wall_clearance_pixels']}
        (run/'diagnostics.json').write_text(json.dumps(report,indent=2));reports.append({k:v for k,v in report.items() if k!='per_context'});print(run,flush=True)
    bank.assert_frozen();(Path(args.root)/'diagnostics_summary.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
