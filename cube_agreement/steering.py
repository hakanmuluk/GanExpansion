"""Shared learned bounded manipulation steering, fitted to single-skill physics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn


class Steering(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(3,5)
    def forward(self,x):
        a=self.linear(x)
        return torch.cat([a[:,:3].clamp(-.65,.65),a[:,3:].clamp(-1,1)],-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);args=p.parse_args();root=Path(args.data);assert(root/'DONE.json').exists();x=[];y=[];sources=[]
    for path in sorted(root.glob('*/episode_*.npz')):
        with np.load(path) as f:
            assert set(np.unique(f['skill'])).issubset({'keepout','waypoint'})
            mask=f['teacher_phase']==4
            if mask.any():x.append(f['steering_inputs'][mask]);y.append(f['actions'][mask]);sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'steps':int(mask.sum())})
    x=np.concatenate(x).astype(np.float64);y=np.concatenate(y).astype(np.float64);design=np.c_[x,np.ones(len(x))];coefficients=[]
    # Saturated targets are censored; fit the corresponding unsaturated linear map.
    for j in range(5):
        valid=np.abs(y[:,j])<.64 if j<3 else np.ones(len(y),bool)
        coef=np.linalg.lstsq(design[valid],y[valid,j],rcond=None)[0];coefficients.append(coef)
    coefficients=np.stack(coefficients);model=Steering()
    with torch.no_grad():model.linear.weight.copy_(torch.tensor(coefficients[:,:3],dtype=torch.float32));model.linear.bias.copy_(torch.tensor(coefficients[:,3],dtype=torch.float32))
    model.eval().requires_grad_(False)
    with torch.no_grad():prediction=model(torch.tensor(x,dtype=torch.float32)).numpy()
    error=prediction-y;out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    record={'scope':'Shared neural bounded-linear steering fitted from actual independently collected single-skill transport states/actions. No joint examples, joint costs or task regions are training inputs. This does not yet train reaching/grasping/release.','args':vars(args),'transport_samples':len(x),'sources':sources,'RMSE':float(np.sqrt(np.mean(error**2))),'maximum_action_error':float(np.abs(error).max()),'coefficients':coefficients.tolist()}
    torch.save({'state':model.state_dict(),'metadata':record},out);out.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
