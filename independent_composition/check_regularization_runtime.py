"""Original-versus-exported prior/cycle parity at recorded observed states.

Raw reference arrays remain on the evaluation host. Only aggregate numerical
errors are written to the proof JSON. This does not itself prove task success.
"""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np
import torch


def predict(policy,arrays):
    states=torch.tensor(arrays['states']);base=torch.tensor(arrays['base_z'])
    contexts=tuple(torch.tensor(arrays['contexts'][:,j]) for j in [0,1])
    plans=[];codes=[]
    for i in range(len(states)):
        plan=policy.sample(states[i:i+1],tuple(c[i:i+1] for c in contexts),base[i:i+1])
        plans.append(plan.numpy()[0]);codes.append(policy.last_plan['selected_existing_inputs'][0])
    policy.bank.assert_frozen()
    return {'actions':np.asarray(plans),'codes':np.asarray(codes)}


def main():
    p=argparse.ArgumentParser()
    for name in ['env','base','keepout','waypoint','reference','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=['baseline','prior','prior_cycle'],default='prior')
    p.add_argument('--encoders',nargs=2);p.add_argument('--write-reference');p.add_argument('--config');p.add_argument('--bundle')
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if a.write_reference:
        from .regularization_models import load_bank,load_cycle_encoders
        from .regularized_latent_composition import RegularizedConfig,RegularizedComposer
        from .inference_guard import GuardedPolicy
        from .check_regularization_loss_paths import observed_inputs
        s,c,b=observed_inputs(a.env,a.write_reference,'cpu')
        arrays={'states':s.numpy(),'contexts':torch.stack(c,1).numpy(),'base_z':b.numpy()}
        config=RegularizedConfig(**json.loads(Path(a.config).read_text()))
        bank=load_bank(a.env,a.base,a.keepout,a.waypoint,'cpu')
        if config.cycle_weight:load_cycle_encoders(bank,a.env,a.encoders,config.cycle_horizon)
        policy=GuardedPolicy(RegularizedComposer(bank,config))
        values=predict(policy,arrays)
        np.savez_compressed(a.reference,**arrays,**values,frozen_fingerprint=np.array(bank.fingerprint))
        Path(a.out).write_text(json.dumps({'reference_created_remote_only':True,'observations':len(s)})+'\n');return
    blocked={'mujoco','gymnasium','dm_control','dm_robotics','cube','cube_addons','cube_composition',
             'tworoom_gan','tworoom_addons','tworoom_composition','stable_worldmodel','independent_composition'}
    assert not {name.split('.')[0] for name in sys.modules}&blocked
    attempts=[]
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in blocked:
                attempts.append(fullname);raise AssertionError('Forbidden import: '+fullname)
    sys.meta_path.insert(0,Block())
    if a.bundle:sys.path.insert(0,a.bundle)
    from skill_latent_regularization import load_composer
    policy=load_composer(a.env,a.base,a.keepout,a.waypoint,regularizer=a.mode,encoders=a.encoders)
    with np.load(a.reference) as x:
        arrays={key:x[key] for key in ['states','contexts','base_z']}
        actual=predict(policy,arrays);errors={key:float(np.max(np.abs(value-x[key]))) for key,value in actual.items()}
        assert all(np.array_equal(value,x[key]) for key,value in actual.items()),errors
        assert policy.bank.fingerprint==str(x['frozen_fingerprint'])
    assert not attempts and not {name.split('.')[0] for name in sys.modules}&blocked
    result={'environment':a.env,'mode':a.mode,'observations':len(arrays['states']),
        'original_actions_and_optimized_codes_bitwise_equal':True,'maximum_errors':errors,
        'all_weights_frozen_unchanged':True,'simulator_and_original_project_imports_blocked':True,
        'forbidden_import_attempts':attempts,'new_neural_modules':0,
        'private_prior_weight':policy.config.private_prior_weight,'cycle_weight':policy.config.cycle_weight,
        'cycle_uses_existing_calibrated_encoders':bool(policy.config.cycle_weight),'scope':__doc__}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
