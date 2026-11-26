"""Offline conversion of existing own-skill expert paths to coded actions.

Paths are teacher data only. Runtime models receive observed state, their own
region context, base code and skill code; never path points or a route network.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .collect_broad_tworoom import collect_plans
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);report={}
    for j,skill in enumerate(['keepout','waypoint']):
        (out/skill).mkdir();report[skill]={}
        for split in ['train','val']:
            source=Path(args.source)/skill/(split+'.npz')
            with np.load(source) as data:
                c=data['condition'];code=data['code'];paths=(data['path']+1)*112
            cases={'start':(c[:,:2]+1)*112,'goal':(c[:,2:4]+1)*112,'base_z':c[:,22:24],
                   'spec':np.concatenate([(c[:,10:12]+1)*112,c[:,12:13]*40],-1)}
            pieces=[]
            for a in range(0,len(c),1024):
                b=min(a+1024,len(c))
                result=collect_plans({k:v[a:b] for k,v in cases.items()},list(paths[a:b]),skill,code[a:b],'cuda')
                result['new_z']=result['new_z'].reshape(b-a,-1)
                keep=result['success'];pieces.append({k:v[keep] for k,v in result.items()})
            merged={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]}
            assert len(merged['lengths'])>0
            merged['seed']=np.arange(len(merged['lengths']),dtype=np.int64)+j*1000000+(10000000 if split=='val' else 0)
            np.savez_compressed(out/skill/(split+'.npz'),**merged)
            report[skill][split]={'attempted':len(c),'retained_own_successes':len(merged['lengths']),
                'code_dim':merged['new_z'].shape[1],'source_sha256':digest_file(source),
                'policy_inputs':'state, own context, base code, skill code; no path inputs'}
            (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'skill':skill,'split':split,**report[skill][split]}),flush=True)
    (out/'DONE.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
