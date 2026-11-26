"""Audit the exact prior/cycle values and action-to-encoder gradient path.

Only recorded current observations are used. This is an algebra/gradient audit,
not an additional task-performance claim. Raw observations stay on the host.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .regularization_models import load_bank,load_cycle_encoders
from .regularized_latent_composition import RegularizedConfig,RegularizedComposer,window_encode,issued_controls
from .inference_guard import tripwires


def observed_inputs(environment,run,device):
    if environment=='cube':
        with np.load(Path(run)/'rollout_0000.npz') as x:
            indices=[0,len(x['states'])//2]
            s=torch.tensor(x['states'][indices],device=device)
            contexts=tuple(torch.tensor(x['contexts'][indices,j],device=device) for j in [0,1])
            b=torch.tensor(x['base_z'],device=device)[None].expand(2,-1)
        return s,contexts,b
    from tworoom_gan.env import observation
    from tworoom_composition.geometry import contexts
    from tworoom_addons.geometry import segment_distance
    with np.load(Path(run)/'trajectories.npz') as x:
        case={k:torch.tensor(x[k][:1],device=device) for k in ['start','goal','base_z','keepout','waypoint']}
        positions=torch.tensor(x['positions'][:1],device=device);actions=torch.tensor(x['actions'][:1],device=device)
        indices=[0,int(x['lengths'][0])//2]
    states=[];cs=[]
    for t in indices:
        visited=torch.zeros(1,dtype=torch.bool,device=device)
        if t:
            visited=(segment_distance(positions[:,:t],positions[:,1:t+1],case['waypoint'][:,None,:2])<=case['waypoint'][:,2:3]).any(1)
        previous=actions[:,t-1] if t else torch.zeros(1,2,device=device)
        states.append(observation(positions[:,t],case['goal']))
        cs.append(contexts(positions[:,t],case,visited,torch.full((1,),float(t),device=device),previous))
    return torch.cat(states),tuple(torch.cat([c[j] for c in cs]) for j in [0,1]),case['base_z'].expand(2,-1)


def main():
    p=argparse.ArgumentParser()
    for name in ['env','base','keepout','waypoint','config','observations','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--encoders',nargs=2,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(2)
    def bank():return load_bank(a.env,a.base,a.keepout,a.waypoint,a.device)
    template=json.loads(Path(a.config).read_text());template.update(private_prior_weight=0.,cycle_weight=0.,cycle_horizon=16,cycle_target='self')
    base=RegularizedComposer(bank(),RegularizedConfig(**template))
    calibrated=bank();load_cycle_encoders(calibrated,a.env,a.encoders,16)
    config=dict(template,private_prior_weight=.02,cycle_weight=.03)
    regularized=RegularizedComposer(calibrated,RegularizedConfig(**config))
    state,ctx,bz=observed_inputs(a.env,a.observations,a.device)
    z=torch.cat([state.new_full((len(state),2),-.8),bz[:,:1].abs().expand(-1,2)],-1).clone().requires_grad_(True)
    with tripwires():
        old=base.objective(state,bz,ctx,z)[0]
        new,k,w,details=regularized.objective(state,bz,ctx,z)
        bk=torch.stack([bz[:,0].sign()*z[:,2],bz[:,1]],-1)
        bw=torch.stack([bz[:,0].sign()*z[:,3],bz[:,1]],-1)
        expected_cycle=torch.zeros_like(old);detached_cycle=torch.zeros_like(old)
        for j,(encoder,b,actions) in enumerate(zip([calibrated.ek,calibrated.ew],[bk,bw],[k,w])):
            controls=issued_controls(actions[:,:16])
            mu,_=window_encode(encoder,state,ctx[j],b,controls)
            dm,_=window_encode(encoder,state,ctx[j],b,controls.detach())
            expected_cycle+=.5*(mu-z[:,j:j+1]).abs().mean(1)
            detached_cycle+=.5*(dm-z[:,j:j+1]).abs().mean(1)
        expected_prior=.5*z[:,:2].square().sum(1)
        expected=.02*expected_prior+.03*expected_cycle
        error=float(((new-old)-expected).abs().max().detach())
        assert error<1e-5,error
        full_gradient,=torch.autograd.grad((new-old-.02*expected_prior).sum(),z,retain_graph=True)
        detached_gradient,=torch.autograd.grad((.03*detached_cycle).sum(),z)
        action_path=float((full_gradient-detached_gradient).abs().max())
        assert torch.isfinite(full_gradient).all() and action_path>1e-10,action_path
    base.bank.assert_frozen();calibrated.assert_frozen()
    proof={'scope':__doc__,'environment':a.env,'observed_conditions':len(state),
        'exact_weighted_private_prior_and_cycle_value_error':error,
        'nonzero_gradient_through_generated_actions_and_existing_encoder':action_path,
        'all_generator_base_adapter_encoder_weights_frozen_unchanged':True,
        'inference_tripwires_enabled':True,'new_neural_modules':0,
        'base_original_encoders_remain_forbidden':True,'cycle_horizon':16}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
