"""Original-checkpoint parity and mode-restriction checks, Cube only."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import importlib.util
import json
from pathlib import Path
import sys
import torch
from .unchanged_composition import CompositionConfig,OriginalComposer
from .inference_guard import GuardedPolicy
from .legacy import digest_file


def main():
    torch.set_num_threads(2)
    from cube_composition.models import FrozenBank
    from cube_composition.evaluate import load_cases
    from cube_composition.scene import prepare
    from cube_composition.geometry import JointObserver
    from cube_addons.env import make_env,close_env,features
    from cube_addons.contacts import contact_info,force_holding
    bank=FrozenBank();env=make_env()
    try:
        case=load_cases('cube_composition_artifacts/tasks_v2/val_supported_easy.npz',1)[0]
        ob,info=prepare(env,case);contact_info(env,info)
        observer=JointObserver(case);observer.previous_held=force_holding(info)
        observer.k.previous_holding=observer.w.previous_holding=observer.previous_held
        state=torch.as_tensor(features(env,ob,info,case['initial_cube'],None),device='cuda',dtype=torch.float32)[None]
        ctx=tuple(torch.as_tensor(x,device='cuda',dtype=torch.float32)[None] for x in observer.features(info,0))
        bz=torch.as_tensor(case['base_z'],device='cuda',dtype=torch.float32)[None]
        reference=Path('independent_composition_artifacts/original_agreement_window_v27_cube_e8/e8_h2_supported_easy/source/unchanged_composition.py')
        spec=importlib.util.spec_from_file_location('independent_composition._mode_reference',reference)
        old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        settings=dict(alpha=.25,optimize=True,history=0.,agreement_horizon=2,steps=2)
        previous=GuardedPolicy(old.OriginalComposer(bank,old.CompositionConfig(**settings),241))
        current=GuardedPolicy(OriginalComposer(bank,CompositionConfig(**settings),241))
        for repeat in range(2):
            assert torch.equal(previous.sample(state,ctx,bz),current.sample(state,ctx,bz))
            assert torch.equal(previous.previous,current.previous)
        for sign in [-1,1]:
            policy=GuardedPolicy(OriginalComposer(bank,CompositionConfig(**settings,keepout_sign=sign),241))
            for repeat in range(2):
                assert torch.isfinite(policy.sample(state,ctx,bz)).all()
                assert (policy.noise[:,:,0].sign()==sign).all()
                assert (policy.previous[:,0].sign()==sign).all()
        for sign in [-1,1]:
            policy=GuardedPolicy(OriginalComposer(bank,CompositionConfig(**settings,preserve_mode=True),241))
            policy.policy.previous=state.new_tensor([[sign,1.]])
            for repeat in range(2):
                policy.sample(state,ctx,bz)
                assert (policy.noise[:,:,0].sign()==sign).all()
                assert (policy.previous[:,0].sign()==sign).all()
        bank.assert_frozen()
        assert all(digest_file(p)==sha for p,sha in bank.files.items())
        print(json.dumps({'cube_only':True,'default_behavior_exactly_matches_frozen_v27_source':True,
            'positive_and_negative_mode_constraints_checked':True,'preserve_selected_mode_checked':True,
            'original_model_inputs_weights_and_dimensions_unchanged':True,'original_model_parameter_gradients':0,
            'inference_tripwires_enabled':True}))
    finally:close_env(env)


if __name__=='__main__':main()
