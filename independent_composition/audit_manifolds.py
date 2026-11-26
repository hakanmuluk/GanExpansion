"""Audit code information, action-channel dependence and private frozen models."""
import argparse
import ast
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .cvae import load_manifold
from .train_diffusion import load_base
from .windows_data import Windows
from .legacy import digest_file


@torch.no_grad()
def diagnose(model,base,windows,seed):
    rng=torch.Generator(device=windows.device).manual_seed(seed)
    b=windows.sample(256,rng,horizon=model.config.horizon,alignment=8 if model.config.action_dim==5 else 1)
    ba=base(b['state'],b['base_z']);condition=model.features(b['state'],b['context'],b['base_z'],ba)
    copies=8;condition=condition.repeat_interleave(copies,0);ba=ba.repeat_interleave(copies,0)
    code=torch.randn(len(condition),model.config.code_dim,device=condition.device,generator=rng)
    action=model.decode(condition,ba,code);reference=model.extended_base(ba)
    recovered,lv=model.encode(condition,action)
    def rmse(a):return float((model.encode(condition,a)[0]-code).square().mean().sqrt())
    shuffled=action.reshape(256,copies,*action.shape[1:]).roll(1,1).reshape_as(action)
    smooth=F.avg_pool1d(F.pad(action.transpose(1,2),(2,2),mode='replicate'),5,stride=1).transpose(1,2)
    channel_rmse=[]
    for channel in range(action.shape[-1]):
        ablated=action.clone();ablated[:,:,channel]=reference[:,:,channel]
        channel_rmse.append(rmse(ablated))
    groups=action.reshape(256,copies,*action.shape[1:])
    variation=groups.var(1).mean(0).sqrt()
    centered=code-code.mean(0);encoded_centered=recovered-recovered.mean(0)
    correlation=(centered*encoded_centered).mean(0)/(centered.square().mean(0)*encoded_centered.square().mean(0)).sqrt().clamp_min(1e-9)
    mask=b['mask'].repeat_interleave(copies,0)
    masked_recovered,_=model.encode(condition,action,mask)
    return {'code_recovery_rmse':float((recovered-code).square().mean().sqrt()),
            'constant_zero_rmse':float(code.square().mean().sqrt()),'code_correlation':correlation.tolist(),
            'train_length_mask_recovery_rmse':float((masked_recovered-code).square().mean().sqrt()),
            'shuffled_actions_same_state_rmse':rmse(shuffled),'smoothed_actions_rmse':rmse(smooth),
            'quantized_actions_0_05_rmse':rmse((action/.05).round()*.05),
            'restored_to_base_channel_recovery_rmse':channel_rmse,
            'action_std_across_codes_by_channel':variation.mean(0).tolist(),
            'first_four_controls_std_by_channel':variation[:4].mean(0).tolist(),
            'mean_encoder_posterior_std':float((lv*.5).exp().mean()),
            'generated_vs_base_rms':float((action-reference).square().mean().sqrt())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--checkpoints',nargs=2,required=True);p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(2);base,source=load_base(args.env,args.device)
    models=[];results={};provenance={}
    for path,skill in zip(args.checkpoints,['keepout','waypoint']):
        model,saved=load_manifold(path,args.device)
        assert saved['config']['base']==source and saved['config']['args']['skill']==skill
        models.append(model);data=Path(args.data)/skill/'val.npz'
        windows=Windows(data,args.device)
        results[skill]=diagnose(model,base,windows,7521901)
        provenance[skill]={'checkpoint':path,'sha256':digest_file(path),'validation_sha256':digest_file(data)}
    ids=[{p.data_ptr() for p in m.parameters()} for m in [base,*models]]
    assert all(not ids[i]&ids[j] for i in range(3) for j in range(i))
    assert all(p.grad is None and not p.requires_grad for m in [base,*models] for p in m.parameters())
    tree=ast.parse(Path(__file__).with_name('cvae.py').read_text());imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
    assert not any(x.startswith(('mujoco','cube','tworoom','stable_worldmodel')) for x in imports)
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'environment':args.env,'models':provenance,'base':source,'diagnostics':results,
            'private_parameter_storage':True,'models_frozen':True,'sampler_imports':imports,
            'scope':'Own-skill validation windows only. Code recovery is a diagnostic, not a substitute for physical own-skill or joint task success.'}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
