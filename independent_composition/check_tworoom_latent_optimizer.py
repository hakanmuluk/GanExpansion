"""Check actual code optimization and simulator-free inference at measured states.

This is a computational audit, not a task-performance benchmark. Reference
observations remain on the evaluation host and only aggregate errors are saved.
"""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    for key in ['config','reference','out']:p.add_argument('--'+key,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    blocked={'stable_worldmodel','mujoco','gymnasium','dm_control','dm_robotics'}
    attempts=[]
    assert not {n.split('.')[0] for n in sys.modules}&blocked
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in blocked:
                attempts.append(fullname);raise AssertionError('Forbidden import: '+fullname)
    sys.meta_path.insert(0,Block())
    from tworoom_composition.models import FrozenBank
    from .latent_action_composition import AmplitudeConfig,AmplitudeComposer
    from .inference_guard import GuardedPolicy,tripwires
    import tworoom_gan.env as env
    bank=FrozenBank(device='cpu')
    config=AmplitudeConfig(**json.loads(Path(a.config).read_text()))
    policy=GuardedPolicy(AmplitudeComposer(bank,config,241))
    with np.load(a.reference) as x:
        states=torch.tensor(x['states']);base_z=torch.tensor(x['base_z'])
        contexts=tuple(torch.tensor(x['contexts'][:,j]) for j in [0,1])
    errors=[];improvements=[];changed=[];max_input_grad=0.
    for i in range(len(states)):
        s=states[i:i+1];b=base_z[i:i+1];ctx=tuple(c[i:i+1] for c in contexts)
        result=policy.sample(s,ctx,b)
        z=s.new_tensor(policy.last_plan['selected_existing_inputs'])
        initial=policy.noise.reshape(-1,4)
        with torch.no_grad():
            start_cost=policy.objective(s.expand(len(initial),-1),b.expand(len(initial),-1),
                                        tuple(c.expand(len(initial),-1) for c in ctx),initial)[0].min()
            value,k,w,_=policy.objective(s,b,ctx,z)
            expected=(w+(k-bank.base(s,b))).clamp(-1,1) if config.merge=='residual' else w
        errors.append(float((expected-result).abs().max()))
        assert torch.equal(result,expected)
        improvement=float(start_cost-value[0]);improvements.append(improvement)
        assert improvement>=-2e-5, (i,improvement)
        reference=torch.cat([s.new_full((1,2),-.8),b[:,:1].abs().expand(-1,2)],-1)
        changed.append(float((z-reference).norm()))
        variable=z.clone().requires_grad_(True)
        gradient,=torch.autograd.grad(policy.objective(s,b,ctx,variable)[0].sum(),variable)
        assert torch.isfinite(gradient).all();max_input_grad=max(max_input_grad,float(gradient.abs().max()))
        assert result.shape==(1,16,2) and torch.isfinite(result).all()
        assert policy.last_plan['optimizer_iterations']==20 and policy.last_plan['simulator_queries']==0
    assert max_input_grad>0 and max(changed)>1e-5
    caught=False
    try:
        with tripwires():env.transition(torch.zeros(1,2),torch.zeros(1,2))
    except AssertionError:caught=True
    assert caught and not attempts and not {n.split('.')[0] for n in sys.modules}&blocked
    bank.assert_frozen()
    proof={'observations_checked':len(states),'actual_adam_steps':20,'existing_scalar_inputs_optimized':4,
           'no_new_neural_modules':True,'external_simulator_imports_blocked':sorted(blocked),
           'forbidden_import_attempts':attempts,'transition_tripwire_tested':caught,
           'maximum_output_formula_error':max(errors),'objective_improvement_from_best_initial_start':improvements,
           'minimum_code_distance_from_fixed_reference':min(changed),'maximum_code_distance_from_fixed_reference':max(changed),
           'maximum_finite_input_gradient':max_input_grad,'all_network_weights_frozen_unchanged':True,
           'encoders_used_for_planning':False,'scope':__doc__}
    Path(a.out).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
