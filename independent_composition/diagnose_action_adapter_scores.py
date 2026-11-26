"""Aggregate denoising/context-sensitivity diagnostics; never export action arrays."""
import argparse
import json
from pathlib import Path
import torch
from .action_adapter_diffusion import load
from .windows_data import Windows
from .legacy import digest_file
from .inference_guard import tripwires


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();torch.set_num_threads(2);out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    results=[]
    for path in args.checkpoints:
        model,saved=load(path);data=Windows(saved['config']['args']['validation'])
        rng=torch.Generator(device='cuda').manual_seed(229714)
        b=data.sample(1024,rng,alignment=8 if model.environment=='cube' else 1,start_fraction=.2)
        base=model.actor.base(b['state'],b['base_z']);target=b['actions'];mask=b['mask']
        def rmse(x):return float(((x.square()*mask[...,None]).sum()/(target.shape[-1]*mask.sum().clamp_min(1))).sqrt())
        truth_distance=(((target-base).square()*mask[...,None]).sum((1,2))/(target.shape[-1]*mask.sum(-1).clamp_min(1))).sqrt()
        rows=[]
        for time in [10,50,70,90]:
            t=torch.full((len(target),),time,device='cuda',dtype=torch.long);a=model.alpha_bar[t,None,None]
            noise=torch.randn(target.shape,device='cuda',generator=rng);noisy=a.sqrt()*target+(1-a).sqrt()*noise
            with tripwires():
                own=model.predict_clean(noisy,t,b['state'],b['context'],b['base_z'])
                null=model.predict_clean(noisy,t,b['state'],b['context'],b['base_z'],target.new_zeros(len(target),1))
                shuffled=model.predict_clean(noisy,t,b['state'],b['context'].roll(1,0),b['base_z'])
            x=(own-base)*mask[...,None];y=(null-base)*mask[...,None]
            cosine=(x.flatten(1)*y.flatten(1)).sum(1)/(x.flatten(1).norm(dim=1)*y.flatten(1).norm(dim=1)).clamp_min(1e-8)
            rows.append({'time':time,'clean_rmse':rmse(own-target),'own_null_rmse':rmse(own-null),
                'shuffled_context_prediction_change_rmse':rmse(own-shuffled),'own_base_rmse':rmse(own-base),
                'null_base_rmse':rmse(null-base),'mean_own_null_residual_cosine':float(cosine.mean())})
        results.append({'path':path,'sha256':digest_file(path),'environment':model.environment,'skill':saved['skill'],
            'training_reference':'base' if model.config.score_residual else 'context_drop',
            'teacher_base_rmse':rmse(target-base),'own_intervention_fraction_above_0.1':float((truth_distance>=.1).float().mean()),'rows':rows})
        del model,data,b;torch.cuda.empty_cache()
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
