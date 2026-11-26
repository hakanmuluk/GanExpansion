"""Original held-out skill protocols, with loading support for branch removal."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .evaluation import load_adapter,install_cube_variant_loader


def functional_checks(model,data,environment):
    if environment=='tworoom':
        state=data['states'][:128,0];context=data['contexts'][:128,0];bz=data['base_z'][:128]
    else:
        ids=np.arange(min(128,len(data['lengths'])));t=np.maximum(0,data['lengths'][ids]//2)
        state=data['states'][ids,t];context=data['contexts'][ids,t];bz=data['base_z'][ids]
    device=next(model.parameters()).device
    state,context,bz=[torch.as_tensor(x,device=device) for x in [state,context,bz]]
    u=torch.ones((len(state),1),device=device)
    with torch.inference_mode():
        disabled=model(state,bz,u,context,gate=0);original=model.base(state,bz)
        plus=model(state,bz,u,context);minus=model(state,bz,-u,context)
    assert torch.equal(disabled,original),'Frozen base path must be exact'
    return {'disabled_prediction_bitwise_equal':True,
            'private_sign_action_rms':float((plus-minus).square().mean().sqrt()),
            'private_sign_action_max_abs':float((plus-minus).abs().max()),'observed_states':len(state)}


def benchmark(model,state,context,bz):
    """Same observed inputs, fixed repeats. Latency excludes loading and encoders."""
    device=next(model.parameters()).device
    inputs=[torch.as_tensor(x[:1],device=device) for x in [state,context,bz]]
    s,c,b=inputs;u=torch.ones((1,1),device=device)
    with torch.inference_mode():
        for _ in range(20):model(s,b,u,c)
        values=[]
        for _ in range(100):
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();model(s,b,u,c)
            if device.type=='cuda':torch.cuda.synchronize()
            values.append((time.perf_counter()-start)*1000)
    return {'batch_size':1,'warmup_calls':20,'measured_calls':100,
            'median_ms':float(np.median(values)),'p90_ms':float(np.quantile(values,.9)),
            'device':str(device),'scope':'Action-generator call only; no encoder or simulator, synchronized on CUDA.'}


def cube_disabled_native_check(checkpoint,base,cases,skill):
    """Two fixed native trajectories against the common original-base control.

    This engineering preservation check is separate from the unmodified skill
    benchmark above. The original base control uses the same test scene/codes.
    """
    from cube_addons.env import make_env,close_env
    from cube_addons.rollout import rollout
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    actor,_,_=load_adapter(checkpoint,base,'cube','cpu');env=make_env();errors={};exact=True
    try:
        for j in [0,4]:
            path=Path(f'ablation_results/baseline/disabled_base_cube_{skill}/native_rollouts/rollout_0000_{j:02d}.npz')
            with np.load(path) as reference:
                row,traj=rollout(env,actor,cases[0],cases[0]['spec'],skill,reference['new_z'],gate=0,
                    rng=np.random.default_rng(123+j))
                for field in ['states','actions','poses']:
                    same=traj[field].shape==reference[field].shape and np.array_equal(traj[field],reference[field])
                    exact=exact and same
                    error=float(np.max(np.abs(traj[field]-reference[field]))) if traj[field].shape==reference[field].shape else None
                    errors[f'code{j}_{field}_max_abs']=error
        assert exact,'Disabled native path differs from the original base control'
    finally:close_env(env);torch.set_num_threads(previous_threads)
    return {'native_trajectories':2,'first_test_scene':True,'states_actions_poses_bitwise_equal':exact,
        'errors':errors,'scope':'Fixed engineering spot-check plus exact action-level bypass checks; main skill scores use the original complete test bank.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=['tworoom','cube'],required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--base',required=True);p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    model,encoder,ck=load_adapter(args.checkpoint,args.base,args.environment,args.device)
    metadata={'args':vars(args),'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'base_sha256':hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),'adapter_config':asdict(model.config),
        'ablation_config':ck.get('ablation_config',{}),'parameters':{
            'frozen_base_generator':sum(p.numel() for p in model.base.parameters()),
            'action_adapter':sum(p.numel() for p in model.trainable()),
            'private_encoder':sum(p.numel() for p in encoder.parameters())},
        'test_contexts_sha256':hashlib.sha256((Path(args.data)/args.skill/'test_contexts.npz').read_bytes()).hexdigest()}
    (out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.environment=='tworoom':
        from tworoom_addons.evaluate import contexts,evaluate,preservation,offline
        d=contexts(Path(args.data)/args.skill/'test_contexts.npz',args.device)
        report={}
        for label,options in [('quantiles',{'intervention':True}),('gaussian',{'gaussian':True,'intervention':True}),
                              ('resampled',{'resample':True}),('base',{'gate':0})]:
            metrics,result=evaluate(model,encoder if label!='base' else None,d,args.skill,8,**options)
            report[label]=metrics
            if label=='quantiles':torch.save(result,out/'private_quantile_rollouts.pt')
        report['preservation']=preservation(model,args.base,d)
        report['offline']=offline(model,encoder,Path(args.data)/args.skill/'test.npz')
    else:
        from cube_addons.evaluate import load_contexts
        from cube_addons.parallel_eval import evaluate_parallel
        install_cube_variant_loader();cases=load_contexts(Path(args.data)/args.skill/'test_contexts.npz')
        # Original native paired-Gaussian eight-code test, seed123.
        metric,_=evaluate_parallel(args.checkpoint,args.base,cases,args.skill,8,seed=123,
            workers=args.workers,out=out/'native_rollouts')
        report={'gaussian':metric}
    # Functional/inference measurements use existing recorded own-skill inputs.
    with np.load(Path(args.data)/args.skill/'train.npz') as d:
        data={k:d[k][:128] for k in ['states','contexts','base_z','lengths']}
    report['functional_checks']=functional_checks(model,data,args.environment)
    report['inference']=benchmark(model,data['states'][:,0],data['contexts'][:,0],data['base_z'])
    report['parameters']=metadata['parameters']
    if args.environment=='cube':report['disabled_native_preservation']=cube_disabled_native_check(args.checkpoint,args.base,cases,args.skill)
    if ck.get('ablation_config',{}).get('lora_only'):
        assert report['functional_checks']['private_sign_action_max_abs']==0.
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'complete':True,'checkpoint_frozen':True})+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
