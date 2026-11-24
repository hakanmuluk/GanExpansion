"""Fit only latent statistics from existing successful single-skill training data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .models import FrozenBank


def main():
    p=argparse.ArgumentParser();p.add_argument('--keepout',default='addon_artifacts/final/keepout.pt');p.add_argument('--waypoint',default='addon_artifacts/final/waypoint.pt')
    p.add_argument('--out',default='composition_artifacts/priors/seed42.json');p.add_argument('--device',default='cuda');p.add_argument('--episodes',type=int,default=2048)
    args=p.parse_args();torch.set_num_threads(2);bank=FrozenBank(args.keepout,args.waypoint,device=args.device);result={}
    for skill,encoder in [('keepout',bank.ek),('waypoint',bank.ew)]:
        path=Path('addon_artifacts/data')/skill/'train.npz'
        with np.load(path) as d:data={k:d[k][:args.episodes] for k in ['states','actions','contexts','base_z','lengths','new_z','success']}
        assert data['success'].all();pred=[]
        with torch.inference_mode():
            for lo in range(0,len(data['lengths']),64):
                b={k:torch.as_tensor(v[lo:lo+64],device=args.device) for k,v in data.items()}
                mu,_=encoder(b['states'][:,:-1],b['actions'],b['contexts'][:,:-1],b['base_z'],b['lengths']);pred.extend(mu[:,0].cpu().tolist())
        x=np.array(pred);label=data['new_z'][:,0];means=[];stds=[]
        for sign in [-1,1]:
            values=x[np.sign(label)==sign];means.append(float(values.mean()));stds.append(max(.08,float(values.std())))
        result[skill]={'means':means,'stds':stds,'initial_codes':np.quantile(x,np.linspace(.005,.995,128)).tolist(),
                       'encoded_quantiles':np.quantile(x,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),
                       'known_code_quantiles':np.quantile(label,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),
                       'encoder_label_rmse':float(np.mean((x-label)**2)**.5),'encoder_sign_accuracy':float(np.mean(np.sign(x)==np.sign(label))),
                       'episodes':len(x),'source_data_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    bank.assert_frozen();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'skills':result,'args':vars(args),'frozen_files':bank.files,'joint_demonstrations_used':0},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
