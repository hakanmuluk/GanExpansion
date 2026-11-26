"""Code recovery from generated chunks at held-out single-skill observations.

Repeated tokens contain one measured state/context, never future states. Codes
are varied at each fixed condition; shuffled-action controls cannot use context
to infer the target code. This audit does not claim alternate physics rollouts.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from .regularization_models import load_bank,load_cycle_encoders,digest_file,state_digest
from .regularized_latent_composition import window_encode,issued_controls
from .windows_data import Windows


@torch.no_grad()
def diagnostics(actor,encoder,windows,horizon,groups=128,seed=1391001):
    device=next(actor.parameters()).device
    rng=torch.Generator(device=device).manual_seed(seed)
    observed=windows.sample(groups,rng,alignment=8)
    # Symmetric fixed quantiles, both signs, at each identical own condition.
    values=observed['state'].new_tensor([-1.5,-.8,-.25,.25,.8,1.5])
    m=len(values);s=observed['state'].repeat_interleave(m,0)
    c=observed['context'].repeat_interleave(m,0)
    b=observed['base_z'].repeat_interleave(m,0)
    z=values.repeat(groups)[:,None]
    actions=issued_controls(actor(s,b,z,c)[:,:horizon])
    mu,lv=window_encode(encoder,s,c,b,actions)
    shuffled=actions.reshape(groups,m,horizon,-1).flip(1).reshape_as(actions)
    wrong,_=window_encode(encoder,s,c,b,shuffled)
    zero,_=window_encode(encoder,s,c,b,torch.zeros_like(actions))
    # Flip each target's sign by using the opposite-code action chunk.
    result={'groups':groups,'conditions':len(s),'horizon':horizon,
            'rmse':float((mu-z).square().mean().sqrt()),
            'prior_zero_rmse':float(z.square().mean().sqrt()),
            'shuffled_within_condition_rmse':float((wrong-z).square().mean().sqrt()),
            'zero_action_rmse':float((zero-z).square().mean().sqrt()),
            'sign_accuracy':float((mu.sign()==z.sign()).float().mean()),
            'mean_posterior_std':float((.5*lv).exp().mean()),
            'code_flip_action_mse':float((actions-shuffled).square().mean()),
            'negative_branch_rmse':float((mu[z[:,0]<0]-z[z[:,0]<0]).square().mean().sqrt())}
    informative=(actions-shuffled).square().mean((1,2))>1e-5
    result['action_informative_fraction']=float(informative.float().mean())
    if informative.any():
        result['informative_rmse']=float((mu[informative]-z[informative]).square().mean().sqrt())
        result['informative_shuffled_rmse']=float((wrong[informative]-z[informative]).square().mean().sqrt())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['env','base','keepout','waypoint','data','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--runtime');p.add_argument('--device',default='cuda');p.add_argument('--groups',type=int,default=128)
    p.add_argument('--encoders',nargs=2);p.add_argument('--horizon',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2);started=time.time()
    bank=load_bank(a.env,a.base,a.keepout,a.waypoint,a.device,a.runtime)
    if a.encoders:load_cycle_encoders(bank,a.env,a.encoders,a.horizon)
    metrics={}
    for skill,actor,encoder in [('keepout',bank.k,bank.ek),('waypoint',bank.w,bank.ew)]:
        data=Path(a.data)/skill/'train.npz'
        windows=Windows(data,a.device,split='validation')
        metrics[skill]={str(h):diagnostics(actor,encoder,windows,h,a.groups) for h in [8,16]}
    bank.assert_frozen()
    result={'scope':__doc__,'environment':a.env,'metrics':metrics,'seconds':time.time()-started,
            'models':bank.files,'all_weights_unchanged':True,'new_neural_modules':0,
            'calibrated_encoders_used':bool(a.encoders),'raw_arrays_exported':False}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
