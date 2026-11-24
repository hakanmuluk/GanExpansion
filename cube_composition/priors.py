"""Fit latent statistics from successful original single-skill physical rollouts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from cube_addons.calibrate import physical_data
from .models import FrozenBank


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(2)
    bank=FrozenBank(args.seed,args.device);skills={}
    for name,encoder in [('keepout',bank.ek),('waypoint',bank.ew)]:
        folder=Path(f'cube_addon_artifacts/physical/{name}_s{args.seed}');source=json.loads((folder/'summary.json').read_text());selection=json.loads(Path(f'cube_addon_artifacts/final/{name}_s{args.seed}/selection.json').read_text())
        assert source['generator_checkpoint_sha256']==selection['generator_source_sha256'] and selection['generator_identical_before_after_calibration']
        assert Path(source['args']['contexts']).name=='train_contexts.npz'
        data,data_sha=physical_data(folder,args.device);pred=[]
        with torch.inference_mode():
            for ids in torch.arange(len(data['lengths']),device=args.device).split(64):
                mu,_=encoder(data['states'][ids],data['actions'][ids],data['contexts'][ids],data['base_z'][ids],data['lengths'][ids]);pred.extend(mu[:,0].cpu().tolist())
        with np.load(f'cube_addon_artifacts/data_force/{name}/train_contexts.npz') as contexts:
            seeds={v.astype(np.float32).tobytes():int(s) for v,s in zip(contexts['base_z'],contexts['seed'])}
        successes={(int(r['context_seed']),np.float32(r['new_z'][0]).tobytes()):r['success'] for r in source['records']}
        bz=data['base_z'].cpu().numpy();codes=data['new_z'][:,0].cpu().numpy();mask=np.array([successes[(seeds[b.tobytes()],np.float32(z).tobytes())] for b,z in zip(bz,codes)])
        x=np.asarray(pred)[mask];labels=codes[mask];means=[];stds=[];weights=[]
        for sign in [-1,1]:
            values=x[np.sign(labels)==sign];assert len(values)>20;means.append(float(values.mean()));stds.append(max(.08,float(values.std())));weights.append(len(values)/len(x))
        skills[name]={'means':means,'stds':stds,'weights':weights,'initial_codes':np.quantile(x,np.linspace(.005,.995,128)).tolist(),
            'encoded_quantiles':np.quantile(x,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),'known_code_quantiles':np.quantile(labels,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),
            'successful_episodes':len(x),'total_physical_training_episodes':len(mask),'encoder_label_rmse':float(np.sqrt(np.mean((x-labels)**2))),
            'encoder_sign_accuracy':float(np.mean(np.sign(x)==np.sign(labels))),'source_physical_trajectory_sha256':data_sha,
            'source_summary_sha256':hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest()}
    bank.assert_frozen();out=Path(f'cube_composition_artifacts/priors/seed{args.seed}.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'skills':skills,'frozen_files':bank.files,'joint_training_episodes_used':0,'args':vars(args)},indent=2)+'\n')
    print(json.dumps({k:{a:v[a] for a in ['successful_episodes','means','stds','encoder_label_rmse','encoder_sign_accuracy']} for k,v in skills.items()},indent=2))


if __name__=='__main__':main()
