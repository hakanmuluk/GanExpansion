"""Real-checkpoint checks for composition algebra and code-only gradients."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import torch
from .unchanged_composition import CompositionConfig,OriginalComposer,action_merge
from .inference_guard import GuardedPolicy


def check(bank,state,contexts,bz,waypoint_sign):
    with torch.no_grad():
        z=state.new_tensor([.8,.8*waypoint_sign])[None]
        k,w=bank.pair(state,bz,z,contexts);base=bank.base(state,bz)
        assert torch.equal(action_merge(k,w,base,CompositionConfig(alpha=0)),w)
        reference=action_merge(k,w,base,CompositionConfig(w_code=.8*waypoint_sign))
    fixed=GuardedPolicy(OriginalComposer(bank,CompositionConfig(w_code=.8*waypoint_sign)))
    assert torch.equal(fixed.sample(state,contexts,bz),reference)
    if k.shape[-1]==5:
        assert torch.equal(reference[...,3:],w[...,3:])
        only_z=action_merge(k,w,base,CompositionConfig(axes='z'))
        assert torch.equal(only_z[...,:2],w[...,:2])
    policy=GuardedPolicy(OriginalComposer(bank,CompositionConfig(optimize=True,steps=2,history=.25,waypoint_sign=waypoint_sign)))
    output=policy.sample(state,contexts,bz)
    policy.observe_action(output[:,0]);assert len(policy.records)==1
    policy.observe_action(output[:,1]);assert len(policy.records)==1
    output=policy.sample(state,contexts,bz)
    assert torch.isfinite(output).all() and output.abs().max()<=1
    assert policy.last_plan['history_observations']==1
    assert policy.previous.abs().min()>=.25 and policy.previous.abs().max()<=1.5
    assert (policy.previous[:,1].sign()==waypoint_sign).all()
    # Perturbing unexecuted outputs must leave prefix loss and code gradients
    # unchanged, while changing the full-horizon control loss.
    prefix=OriginalComposer(bank,CompositionConfig(agreement_horizon=2))
    code=state.new_tensor([[.8,.8*waypoint_sign]]).requires_grad_(True)
    before=prefix.objective(state,bz,contexts,code)[3]['agreement']
    before_grad,=torch.autograd.grad(before.sum(),code)
    full=OriginalComposer(bank,CompositionConfig(agreement_horizon=16))
    full_before=full.objective(state,bz,contexts,code)[3]['agreement'].detach()
    def perturb_future(module,args,result):
        altered=result.clone();altered[:,2:]+=3.;return altered
    handle=bank.k.register_forward_hook(perturb_future)
    try:
        after=prefix.objective(state,bz,contexts,code)[3]['agreement']
        after_grad,=torch.autograd.grad(after.sum(),code)
        assert torch.equal(before.detach(),after.detach())
        assert torch.equal(before_grad,after_grad)
        assert not torch.allclose(full_before,full.objective(state,bz,contexts,code)[3]['agreement'])
    finally:handle.remove()
    assert not any(p.grad is not None or p.requires_grad for p in bank.parameters())
    bank.assert_frozen()
    from .legacy import digest_file
    assert all(digest_file(path)==sha for path,sha in bank.files.items())
    return {'exact_fixed_forward_parity':True,'original_model_file_hashes_unchanged':True,
            'original_model_gradient_count':0,'finite_bounded_original_codes':True,
            'actual_action_history_checked':True,'simulator_tripwires_enabled':True,
            'agreement_prefix_loss_and_code_gradient_ignore_unexecuted_tail':True}


def main():
    torch.set_num_threads(2)
    from tworoom_composition.models import FrozenBank as TwoRoomBank
    from tworoom_composition.evaluate import load_cases
    from tworoom_composition.geometry import contexts
    from tworoom_gan.env import observation
    case=load_cases('composition_artifacts/tasks/val_supported_easy.npz','cuda',1)
    s=observation(case['start'],case['goal'])
    c=contexts(case['start'],case,torch.zeros(1,dtype=torch.bool,device='cuda'),torch.zeros(1,device='cuda'),torch.zeros(1,2,device='cuda'))
    result={'tworoom':check(TwoRoomBank(),s,c,case['base_z'],-1)}
    from cube_composition.models import FrozenBank as CubeBank
    from cube_composition.evaluate import load_cases as cube_cases
    from cube_composition.scene import prepare
    from cube_composition.geometry import JointObserver
    from cube_addons.env import make_env,close_env,features
    from cube_addons.contacts import contact_info,force_holding
    env=make_env()
    try:
        case=cube_cases('cube_composition_artifacts/tasks_v2/val_supported_easy.npz',1)[0]
        ob,info=prepare(env,case);contact_info(env,info)
        observer=JointObserver(case);observer.previous_held=force_holding(info)
        observer.k.previous_holding=observer.w.previous_holding=observer.previous_held
        s=torch.as_tensor(features(env,ob,info,case['initial_cube'],None),device='cuda',dtype=torch.float32)[None]
        c=tuple(torch.as_tensor(x,device='cuda',dtype=torch.float32)[None] for x in observer.features(info,0))
        bz=torch.as_tensor(case['base_z'],device='cuda',dtype=torch.float32)[None]
        result['cube']=check(CubeBank(),s,c,bz,1)
    finally:close_env(env)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
