"""Permutation-independent summaries from completed fixed-code diagnostics.

No probe network, fitting, policy update or test-outcome selection is involved.
Input code indices are matched across all scenes and both training schemes.
Only aggregate distributions/entropies are exported; detailed records stay local
to the execution host.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def entropy(x):
    _,n=np.unique(x,return_counts=True)
    if not len(n):return 0.
    p=n/n.sum();return float(-(p*np.log2(p)).sum())


def summarize(mode,success,z):
    # mode/ok [scene,code], binary physical routes0/1; -1 means no route.
    scene_entropy_all=[];scene_entropy_success=[];per_code=[]
    for row,ok in zip(mode,success):
        scene_entropy_all.append(entropy(row[row>=0]));scene_entropy_success.append(entropy(row[(row>=0)&ok]))
    for j,code in enumerate(z):
        successful=mode[:,j][success[:,j]&(mode[:,j]>=0)]
        crossed=mode[:,j][mode[:,j]>=0]
        counts=np.bincount(successful,minlength=2)
        per_code.append({'code':np.asarray(code).tolist(),'successful_route_counts':counts.tolist(),
            'successful_route1_fraction':float(counts[1]/counts.sum()) if counts.sum() else None,
            'successful_mode_consistency_across_scenes':float(counts.max()/counts.sum()) if counts.sum() else None,
            'crossed_route1_fraction':float(np.mean(crossed==1)) if len(crossed) else None,
            'success':float(success[:,j].mean()),'no_route_fraction':float(np.mean(mode[:,j]<0))})
    # This permits arbitrary relabeling of physical modes: code may pick either
    # route, but should pick it consistently. A collapsed one-route policy can
    # also score high here, so entropy and coverage must accompany this metric.
    values=[x['successful_mode_consistency_across_scenes'] for x in per_code if x['successful_mode_consistency_across_scenes'] is not None]
    return {'scenes':len(mode),'codes_per_scene':mode.shape[1],
        'success':float(success.mean()),'successful_both_routes_per_scene':float(np.mean([len(set(r[o&(r>=0)]))==2 for r,o in zip(mode,success)])),
        'mean_crossed_route_entropy_bits':float(np.mean(scene_entropy_all)),
        'mean_successful_route_entropy_bits_including_no_success_as_zero':float(np.mean(scene_entropy_success)),
        'scenes_with_any_success':float(np.mean(success.any(1))),
        'mean_per_code_successful_mode_consistency':float(np.mean(values)) if values else None,
        'per_code':per_code,
        'interpretation':'Route labels can be swapped without changing entropy/consistency. High consistency alone also permits collapse; assess together with success and within-scene successful route coverage.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--tworoom-seed',type=int);p.add_argument('--out',required=True)
    args=p.parse_args();results={}
    for environment in ['tworoom','cube']:
        results[environment]={}
        for method in ['full','adversarial']:
            seed=args.tworoom_seed if environment=='tworoom' and args.tworoom_seed is not None else args.seed
            name=f'rq1_{environment}_'+('latest_' if environment=='cube' else '')+f'{method}_s{seed}'
            root=Path('ablation_results/evaluation')/name
            if environment=='tworoom':
                data=torch.load(root/'private_rollouts.pt',map_location='cpu',weights_only=False)
                n=len(data['success'])//8;mode=data['mode'].numpy().reshape(n,8)
                success=data['success'].numpy().reshape(n,8);z=data['z'][:8].numpy()
            else:
                rows=json.loads((root/'private_episode_records.json').read_text())
                contexts=sorted({x['context'] for x in rows});n=len(contexts)
                mode=np.full((n,8),-1,int);success=np.zeros((n,8),bool);z=np.zeros((8,2))
                position={c:i for i,c in enumerate(contexts)}
                for row in rows:
                    i,j=position[row['context']],row['latent_index'];actual=int(row['route'])
                    mode[i,j]=0 if actual==-1 else 1 if actual==1 else -1
                    success[i,j]=row['released_success'];z[j]=row['z']
            results[environment][method]=summarize(mode,success,z)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({e:{m:{k:v for k,v in r.items() if k!='per_code'} for m,r in methods.items()} for e,methods in results.items()}),flush=True)


if __name__=='__main__':main()
