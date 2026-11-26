"""Isolate speed-code sign/magnitude in unchanged existing base checkpoints.

This supplementary diagnostic leaves the original benchmark and eight-code
results intact. All original scenes, dynamics and completion rules are reused.
Only the second latent coordinate varies; detailed rollouts stay on the host.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch


def read(path):return json.loads(Path(path).read_text())


def comparison(steps,success):
    both=success[:,0]&success[:,-1]
    delta=steps[both,-1]-steps[both,0]
    return {'paired_successful_contexts':int(both.sum()),
        'faster_at_z2_plus2_count':int((delta<0).sum()),
        'faster_at_z2_plus2_fraction':float((delta<0).mean()) if len(delta) else None,
        'mean_plus2_minus_minus2_steps':float(delta.mean()) if len(delta) else None,
        'scope':'Paired duration contrast restricted to scenes where both endpoint codes succeed.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--codes-json',default='ablation_study/configs/z2_magnitude_codes.json')
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    codes=np.asarray(read(args.codes_json),np.float32)
    assert codes.shape==(7,2) and np.all(codes[:,0]==1)
    assert np.array_equal(codes[:,1],[-2,-1,-.25,0,.25,1,2])
    specifications=[
        ('tworoom','artifacts/final/base_policy.pt','artifacts/data_v2',
         '1ae1be1011701f4502e617e8e7d6e8d64a90ac404c4af35ef41660d22d65f8fd'),
        ('cube','ablation_inputs/cube_latest_base.pt','cube/artifacts/data',
         '34864b592863c56faf75d55cd88fdd33178e2ef29cdd3f890d21c48ea035b46d')]
    results={};table=[];sources={}
    for env,checkpoint,data,expected in specifications:
        assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()==expected
        target=Path('ablation_results/evaluation')/f'z2_magnitude_{env}_existing'
        subprocess.run([sys.executable,'-m','ablation_study.latent_analysis',
            '--environment',env,'--checkpoint',checkpoint,'--data',data,'--out',str(target),
            '--codes-json',args.codes_json,'--workers',str(args.workers)],check=True)
        summary=read(target/'summary.json');sources[env]={'checkpoint':checkpoint,'sha256':expected,'results':str(target)}
        k=len(codes)
        if env=='tworoom':
            raw=torch.load(target/'private_rollouts.pt',map_location='cpu',weights_only=False)
            count=len(raw['success'])//k
            steps=raw['lengths'].numpy().reshape(count,k);ok=raw['success'].numpy().reshape(count,k)
            initial=raw['states'][:,0].numpy().reshape(count,k,-1)
            assert np.array_equal(initial,np.repeat(initial[:,:1],k,1))
            assert np.array_equal(raw['z'].numpy(),np.tile(codes,(count,1)))
        else:
            raw=read(target/'private_episode_records.json')
            contexts=sorted({r['context'] for r in raw});index={c:i for i,c in enumerate(contexts)};count=len(contexts)
            assert len(raw)==count*k and summary['initial_observations_exactly_matched_across_codes']
            steps=np.empty((count,k));ok=np.empty((count,k),bool);seen=set()
            for r in raw:
                i=index[r['context']];j=r['latent_index'];assert (i,j) not in seen;seen.add((i,j))
                assert np.array_equal(r['z'],codes[j])
                steps[i,j]=r['steps'];ok[i,j]=r['released_success']
        per_code=[]
        for j,code in enumerate(codes):
            successful=steps[:,j][ok[:,j]]
            physical=summary['per_latent_aggregate'][j]
            row={'environment':env,'z1':1.,'z2':float(code[1]),'contexts':count,
                'successful_episodes':int(ok[:,j].sum()),'success':float(ok[:,j].mean()),
                'mean_successful_steps':float(successful.mean()) if len(successful) else None,
                'mean_successful_action_magnitude':physical.get('mean_successful_speed'),
                'mean_successful_door_crossing_offset_pixels':physical.get('mean_successful_crossing_offset'),
                'mean_successful_lateral_amplitude_m':physical.get('mean_successful_abs_carry_amplitude_m'),
                'mean_successful_peak_height_m':physical.get('mean_successful_peak_height_m')}
            per_code.append(row);table.append(row)
        results[env]={'contexts':count,'episodes':count*k,'per_code':per_code,
            'paired_speed_effect':comparison(steps,ok),'initial_observations_exactly_matched_across_codes':True}
    scope=('Supplementary one-factor latent intervention, not a changed benchmark. Existing checkpoints only; '
        'z1=1 and z2 varies across seven fixed values. Same original scenes, 16-predict/8-execute controller '
        'and original success criteria. All codes share the same initial observation within each scene. '
        'One route is expected by design; route coverage is not interpreted for this sweep.')
    (out/'summary.json').write_text(json.dumps({'scope':scope,'codes':codes.tolist(),'results':results},indent=2)+'\n')
    with (out/'z2_magnitude_aggregate.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'savefig.dpi':200,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(9,5.5),constrained_layout=True)
    for i,env in enumerate(['tworoom','cube']):
        rows=results[env]['per_code'];x=[r['z2'] for r in rows]
        axes[i,0].plot(x,[r['mean_successful_steps'] for r in rows],'-o',c='#3975ad')
        axes[i,1].plot(x,[r['success']*100 for r in rows],'-o',c='#3975ad')
        for j,ax in enumerate(axes[i]):
            ax.set(xlabel='Speed code z2 (route code z1 = 1)',title='TwoRoom' if env=='tworoom' else 'Cube')
            ax.grid(alpha=.15)
        axes[i,0].set_ylabel('Mean successful episode steps')
        axes[i,1].set(ylabel='Task success (%)',ylim=(0,105))
    fig.suptitle('Unchanged existing base policies: isolated speed-code magnitude')
    for suffix in ['pdf','png']:fig.savefig(out/f'latent_speed_magnitude.{suffix}')
    plt.close(fig)
    (out/'figure_provenance.json').write_text(json.dumps({'scope':scope,'sources':sources,
        'code_grid_sha256':hashlib.sha256(Path(args.codes_json).read_bytes()).hexdigest(),
        'selection':'All original test scenes; no outcome filtering. Duration means condition on success; paired contrasts are supplied separately.'},indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'complete':True,'models_retrained':False,'raw_episode_data_exported':False})+'\n')
    print(json.dumps({'completed':True,'contexts':{e:r['contexts'] for e,r in results.items()},
        'paired_speed_effect':{e:r['paired_speed_effect'] for e,r in results.items()}}),flush=True)


if __name__=='__main__':main()
