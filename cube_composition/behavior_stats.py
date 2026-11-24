"""Causal phase, empirical-prior, retention and release diagnostics from logs.

No neural or physical queries are made. Agreement is the logged raw generator
comparison, before any common-action projection; priors use one common density
per frozen model, regardless of the regularizer selected by each experiment.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,csv,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np


def empirical_nll(codes,priors):
    values=[]
    for j,skill in enumerate(['keepout','waypoint']):
        p=priors[skill];mean=np.array(p['means']);std=np.array(p['stds']);weight=np.array(p['weights'])
        logp=-.5*((codes[:,j,None]-mean)/std)**2-np.log(std)-.5*np.log(2*np.pi)+np.log(weight)
        values.append(-np.logaddexp.reduce(logp,axis=1))
    return np.stack(values,-1)


def phase(context,cube,goal):
    k,w=context
    if k[14]<.5:return 'reach_grasp_lift'
    if k[13]<.5:return 'unsupported_transport' if np.linalg.norm(cube[:2]-goal[:2])>.06 else 'placement_release'
    return 'carry_to_goal' if w[15]>.5 else 'carry_to_waypoint'


def mean(values):return float(np.mean(values)) if len(values) else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',nargs='+',required=True);p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    folders=sorted({p.parent for base in map(Path,args.root) for p in base.rglob('DONE.json') if (p.parent/'config.json').exists() and (p.parent/'records.json').exists()})
    priors={};episodes=[];groups=defaultdict(list)
    for folder in folders:
        config=json.loads((folder/'config.json').read_text());seed=config['args']['model_seed'];priors.setdefault(seed,json.loads(Path(f'cube_composition_artifacts/priors/seed{seed}.json').read_text())['skills'])
        rows=json.loads((folder/'records.json').read_text());planning=json.loads((folder/'planning.json').read_text())
        for row,plans in zip(rows,planning):
            source=folder/f"rollout_{row['context']:04d}.npz"
            with np.load(source) as f:contexts=f['contexts'];poses=f['poses'];held=f['held'];forces=f['forces'];goal=f['goal'];codes=f['codes']
            assert np.isfinite(poses).all() and np.isfinite(codes).all() and np.isfinite(forces).all()
            plan_codes=np.array([v['codes'] for v in plans]);nll=empirical_nll(plan_codes,priors[seed]);names=[];by_phase=defaultdict(list)
            for index,plan in enumerate(plans):
                t=plan['step'];name=phase(contexts[t],poses[t,3:],goal);names.append(name);by_phase[name].append(index)
            base={'method':folder.name,'kind':row['kind'],'model_seed':seed,'search_seed':config['args']['seed'],'source':str(source),'context_seed':row['seed'],'joint_success':row['success'],
                'released_success':row['released_success'],'dropped':row['dropped'],'grasp_losses':row['grasp_losses'],
                'empirical_nll_keepout':float(nll[:,0].mean()),'empirical_nll_waypoint':float(nll[:,1].mean()),'empirical_nll_pair':float(nll.mean())}
            for j,skill in enumerate(['keepout','waypoint']):
                quantiles=priors[seed][skill]['encoded_quantiles'];base[skill+'_code_outside_training_1_99_quantiles']=float(np.mean((plan_codes[:,j]<quantiles[1])|(plan_codes[:,j]>quantiles[-2])))
            release=np.flatnonzero(held[:-1]&~held[1:]&(np.linalg.norm(poses[1:-1,3:5]-goal[:2],axis=1)<.06))
            base['last_release_cube_height_m']=float(poses[release[-1]+1,5]) if row['released_success'] and len(release) else None
            base['sampled_pad_normal_force_p95_N']=np.quantile(forces,.95,axis=0).tolist()
            for name,indices in by_phase.items():
                changes=[float(np.linalg.norm(plan_codes[i]-plan_codes[i-1])) for i in indices if i and names[i-1]==name]
                flips=[(np.sign(plan_codes[i])!=np.sign(plan_codes[i-1])).astype(int).tolist() for i in indices if i and names[i-1]==name]
                disagreements=[plans[i]['disagreement'] for i in indices if plans[i].get('disagreement') is not None]
                executable=[plans[i]['executable_grip_disagreement'] for i in indices if plans[i].get('executable_grip_disagreement') is not None]
                hints=[float(np.linalg.norm(np.array(plans[i]['encoder_hint'])-plan_codes[i])) for i in indices if plans[i].get('encoder_hint') is not None]
                record={**base,'phase':name,'plans':len(indices),'raw_action_disagreement_mean':mean(disagreements),'executable_gripper_disagreement_mean':mean(executable),
                    'same_phase_code_jump_mean':mean(changes),'same_phase_code_sign_flips':np.sum(flips,axis=0).tolist() if flips else [0,0],
                    'encoder_hint_to_selected_code_l2_mean':mean(hints),'empirical_nll_pair':float(nll[indices].mean()),'empirical_nll_waypoint':float(nll[indices,1].mean())}
                groups[(row['kind'],folder.name,name)].append(record)
            base['carry_raw_action_disagreement_mean']=mean([plans[i]['disagreement'] for i,name in enumerate(names) if name=='carry_to_waypoint' and plans[i].get('disagreement') is not None]);episodes.append(base)
    aggregates=[]
    for (kind,method,name),records in sorted(groups.items()):
        a={'kind':kind,'method':method,'phase':name,'episode_count':len(records),'model_seeds':sorted({v['model_seed'] for v in records})}
        for key in ['raw_action_disagreement_mean','executable_gripper_disagreement_mean','same_phase_code_jump_mean','encoder_hint_to_selected_code_l2_mean','empirical_nll_pair','empirical_nll_waypoint']:
            a[key]=mean([v[key] for v in records if v[key] is not None])
        a['same_phase_code_sign_flips_mean']=np.mean([v['same_phase_code_sign_flips'] for v in records],axis=0).tolist();aggregates.append(a)
    (out/'episodes.json').write_text(json.dumps(episodes,indent=2)+'\n');(out/'phases.json').write_text(json.dumps({'args':vars(args),'groups':aggregates,
        'scope':'Offline log analysis only. Phase results condition on episodes that reached the phase. Raw agreement precedes any action projection. Prior density is the same frozen single-skill encoder fit for all regularizers; it is not a calibrated probability of joint-task validity.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    with (out/'phases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
    print('Analyzed episodes:',len(episodes),'phase groups:',len(aggregates),flush=True)


if __name__=='__main__':main()
