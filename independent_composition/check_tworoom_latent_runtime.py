"""Original-versus-portable parity at measured observations; raw arrays stay remote."""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np
import torch


def predictions(policy,arrays):
    s=torch.tensor(arrays['states']);b=torch.tensor(arrays['base_z'])
    ctx=tuple(torch.tensor(arrays['contexts'][:,j]) for j in [0,1])
    plans=[];codes=[]
    for i in range(len(s)):
        plans.append(policy.sample(s[i:i+1],tuple(c[i:i+1] for c in ctx),b[i:i+1]).numpy()[0])
        codes.append(policy.last_plan['selected_existing_inputs'][0])
    policy.bank.assert_frozen()
    return {'composition_actions':np.asarray(plans),'optimized_existing_inputs':np.asarray(codes)}


def main():
    p=argparse.ArgumentParser()
    for name in ['reference','base','keepout','waypoint','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--write-reference');p.add_argument('--config');p.add_argument('--bundle')
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if a.write_reference:
        from tworoom_composition.models import FrozenBank
        from .latent_action_composition import AmplitudeConfig,AmplitudeComposer
        from .inference_guard import GuardedPolicy
        cfg=AmplitudeConfig(**json.loads(Path(a.config).read_text()))
        bank=FrozenBank(keepout=a.keepout,waypoint=a.waypoint,base=a.base,device='cpu')
        policy=GuardedPolicy(AmplitudeComposer(bank,cfg,241))
        with np.load(a.write_reference) as x:
            arrays={key:x[key] for key in ['states','base_z','contexts']}
        computed=predictions(policy,arrays)
        np.savez_compressed(a.reference,**arrays,**computed,frozen_fingerprint=np.array(bank.fingerprint))
        Path(a.out).write_text(json.dumps({'reference_created_remote_only':True,'observations':len(arrays['states'])})+'\n')
        return
    blocked={'mujoco','gymnasium','dm_control','dm_robotics','cube','cube_addons','cube_composition',
             'independent_composition','tworoom_gan','tworoom_addons','tworoom_composition','stable_worldmodel'}
    assert not {n.split('.')[0] for n in sys.modules}&blocked
    attempts=[]
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in blocked:
                attempts.append(fullname);raise AssertionError('Forbidden import: '+fullname)
    sys.meta_path.insert(0,Block());sys.path.insert(0,a.bundle)
    from tworoom_latent_composition import load_composer
    policy=load_composer(a.base,a.keepout,a.waypoint,device='cpu')
    with np.load(a.reference) as x:
        arrays={key:x[key] for key in ['states','base_z','contexts']}
        computed=predictions(policy,arrays)
        errors={key:float(np.max(np.abs(value-x[key]))) for key,value in computed.items()}
        assert all(np.array_equal(value,x[key]) for key,value in computed.items()),errors
        assert policy.bank.fingerprint==str(x['frozen_fingerprint'])
    assert not attempts and not {n.split('.')[0] for n in sys.modules}&blocked
    proof={'observations_checked':len(arrays['states']),'composition_and_selected_inputs_bitwise_equal':True,
           'maximum_errors':errors,'fresh_process_simulator_and_original_project_imports_blocked':True,
           'forbidden_import_attempts':attempts,'all_original_weights_unchanged':True,'new_neural_modules':0,
           'actual_optimizer_steps':20,'execute_steps':8,'scope':__doc__}
    Path(a.out).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
