"""Check W injection parity, gradient isolation and hook cleanup on Cube."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import torch
from .style_composition import StyleComposer,StyleCompositionConfig,mapped_style,style_pair,mapping_override
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
        s=torch.as_tensor(features(env,ob,info,case['initial_cube'],None),device='cuda',dtype=torch.float32)[None]
        ctx=tuple(torch.as_tensor(x,device='cuda',dtype=torch.float32)[None] for x in observer.features(info,0))
        bz=torch.as_tensor(case['base_z'],device='cuda',dtype=torch.float32)[None]
        with torch.no_grad():base=bank.base(s,bz).clone()
        for sign in [-1,1]:
            z=s.new_tensor([[sign*.8,.8]])
            with torch.no_grad():
                styles=tuple(mapped_style(actor,z[:,j:j+1]) for j,actor in enumerate([bank.k,bank.w]))
                direct=bank.pair(s,bz,z,ctx);injected=style_pair(bank,s,bz,z,ctx,styles)
                assert all(torch.equal(a,b) for a,b in zip(direct,injected))
        leaf=tuple(w.detach().clone().requires_grad_(True) for w in styles)
        result=style_pair(bank,s,bz,z,ctx,leaf)
        gradients=torch.autograd.grad(sum(x.square().mean() for x in result),leaf)
        assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients)
        assert all(not actor.mapping._forward_hooks for actor in [bank.k,bank.w])
        try:
            with mapping_override(bank.k,styles[0]):raise RuntimeError('cleanup probe')
        except RuntimeError:pass
        assert not bank.k.mapping._forward_hooks
        policy=GuardedPolicy(StyleComposer(bank,StyleCompositionConfig(alpha=.25,steps=2,style_radius=.5),241))
        for repeat in range(2):
            actions=policy.sample(s,ctx,bz)
            assert actions.shape==(1,16,5) and torch.isfinite(actions).all() and actions.abs().max()<=1
            assert all(not actor.mapping._forward_hooks for actor in [bank.k,bank.w])
        broad=GuardedPolicy(StyleComposer(bank,StyleCompositionConfig(alpha=.25,steps=2,style_radius=1.,style_condition_on_sign=False),241))
        assert torch.isfinite(broad.sample(s,ctx,bz)).all()
        assert all(not actor.mapping._forward_hooks for actor in [bank.k,bank.w])
        assert torch.equal(bank.base(s,bz),base)
        bank.assert_frozen();assert all(digest_file(p)==sha for p,sha in bank.files.items())
        print(json.dumps({'cube_only':True,'mapped_W_injection_exact_original_action_parity':True,
            'private_W_gradients_finite_and_nonzero':True,'temporary_hooks_removed_even_on_exception':True,
            'base_output_exactly_unchanged':True,'original_model_weights_unchanged':True,
            'original_network_parameter_gradients':0,'bounded_W_optimizer_checked':True,
            'private_style_dimensions':policy.style_dimensions,'base_W_optimized':False,'inference_tripwires_enabled':True}))
    finally:close_env(env)


if __name__=='__main__':main()
