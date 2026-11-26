"""Training-only own-task filtering; no joint inputs or policy inference calls.

The historical waypoint demonstrations allowed goal entry before visitation.
Keep only trajectories satisfying the current FIRST-goal-entry task and body
clearance. This changes supervision, never the evaluation predicate or bank.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();source=Path(args.source);out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh filtered-data directory required')
    out.mkdir(parents=True,exist_ok=True)
    from tworoom_composition.geometry import entry_fraction,wall_clearance
    from tworoom_addons.geometry import segment_distance
    report={}
    for skill in ['keepout','waypoint']:
        report[skill]={};(out/skill).mkdir()
        for split in ['train','val']:
            path=source/skill/f'{split}.npz'
            with np.load(path) as f:d={k:f[k] for k in f.files}
            position=torch.from_numpy((d['states'][...,:2]+1)*112)
            a,b=position[:,:-1],position[:,1:]
            length=torch.from_numpy(d['lengths']);n,t=a.shape[:2]
            times=torch.arange(t)[None].expand(n,-1)
            valid=times<length[:,None]
            goal=torch.from_numpy(d['goal'])[:,None]
            native_entry=entry_fraction(a,b,goal,16.)
            goal_time=(times+native_entry).masked_fill(~valid,float('inf')).amin(1)
            wall_ok=~((wall_clearance(a,b)<-1e-4)&valid).any(1)
            own=torch.from_numpy(d['spec'])
            if skill=='waypoint':
                enter=entry_fraction(a,b,own[:,None,:2],own[:,None,2])
                own_time=(times+enter).masked_fill(~valid,float('inf')).amin(1)
                own_ok=torch.isfinite(own_time)&(own_time<=goal_time+1e-6)
            else:
                clearance=segment_distance(a,b,own[:,None,:2])-own[:,None,2]-7
                own_ok=~((clearance < -1e-5)&valid).any(1)
            selected=(torch.isfinite(goal_time)&wall_ok&own_ok).numpy()
            if not selected.any():raise RuntimeError(f'No valid own-task demonstrations: {skill}/{split}')
            filtered={k:v[selected] for k,v in d.items()}
            np.savez_compressed(out/skill/f'{split}.npz',**filtered)
            report[skill][split]={'source_sha256':digest_file(path),'source_episodes':n,
                'retained_episodes':int(selected.sum()),'rejected_episodes':int((~selected).sum()),
                'retained_contexts':len(np.unique(filtered['pair_id'])),
                'definition':'Original goal + own skill + swept body clearance; waypoint precedes first goal entry. No other skill region provided.'}
    (out/'DONE.json').write_text(json.dumps({'source':str(source),'scope':'Single-skill training/encoder-validation demonstrations only. Final environment validation/test context banks remain unfiltered.','skills':report},indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
