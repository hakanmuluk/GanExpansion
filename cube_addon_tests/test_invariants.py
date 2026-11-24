"""Scientific invariants: immutable backbone, real physics, masked episodes."""
import numpy as np
import pytest
import torch
from cube_addons.models import GatedGenerator,AdapterConfig,SkillEncoder,SkillDiscriminator
from cube_addons.geometry import segment_distance,Observer


@pytest.fixture
def model():
    torch.set_num_threads(1);torch.manual_seed(5)
    return GatedGenerator('cube/artifacts/final/base_policy.pt')


def inputs(n=7):
    return torch.randn(n,47),torch.randn(n,2),torch.randn(n,1),torch.randn(n,20)


def test_zero_init_and_frozen_training_switch(model):
    s,z,n,c=inputs();frozen=model.frozen_digest()
    reference=model.base(s,z).detach()
    assert torch.equal(model(s,z,n,c),reference)
    optimizer=torch.optim.Adam(model.trainable(),lr=.01)
    for _ in range(3):
        optimizer.zero_grad();model(s,z,n,c).square().mean().backward();optimizer.step()
    assert model.frozen_digest()==frozen
    assert all(p.grad is None for p in model.base.parameters())
    assert all(p.grad is None for p in model.original_encoder.parameters())
    assert torch.equal(model(s,z,gate=0),reference)
    gate=torch.tensor([0,1,0,1,1,0,1])
    mixed=model(s,z,n,c,gate)
    assert torch.equal(mixed[gate==0],reference[gate==0])
    assert not torch.equal(model(s,z,n,c),reference)
    model.train()
    assert not model.base.training and not model.original_encoder.training


def test_no_new_latent_cannot_observe_code():
    model=GatedGenerator('cube/artifacts/final/base_policy.pt',AdapterConfig(no_new_latent=True))
    for mod in model.modulations:torch.nn.init.normal_(mod[-1].weight,std=.01)
    s,z,n,c=inputs()
    assert torch.equal(model(s,z,n,c),model(s,z,-n*10,c))


def test_copied_discriminator_starts_with_original_conditional_function(model):
    from cube.cube_gan.models import Discriminator
    source=torch.load('cube/artifacts/final/discriminator.pt',map_location='cpu',weights_only=False)['discriminator']
    old=Discriminator(model.base_config);old.load_state_dict(source);old.eval()
    new=SkillDiscriminator(model.base,source).eval()
    s,z,n,c=inputs();a=torch.randn(7,16,5);mask=torch.ones(7,16)
    torch.testing.assert_close(new(s,a,c,z,mask),old(s,a,mask),atol=1e-5,rtol=1e-5)


def test_full_finetuning_comparator_keeps_immutable_disabled_reference():
    model=GatedGenerator('cube/artifacts/final/base_policy.pt',AdapterConfig(finetune_base=True))
    s,z,n,c=inputs();reference=model(s,z,gate=0).detach();frozen=model.frozen_digest()
    opt=torch.optim.SGD(model.trainable(),lr=.1)
    opt.zero_grad();model(s,z,n,c).square().mean().backward();opt.step()
    assert torch.equal(model(s,z,gate=0),reference)
    assert model.frozen_digest()==frozen
    assert not torch.equal(model(s,z,n,c),reference)


def test_observed_transport_gate_protects_original_grasp_actions():
    model=GatedGenerator('cube/artifacts/final/base_policy.pt',AdapterConfig(transport_gate=True))
    for mod in model.modulations:torch.nn.init.normal_(mod[-1].weight,std=.02)
    s,z,n,c=inputs();c[:,14]=0;c[:,10]=0;c[:,13]=0
    assert torch.equal(model(s,z,n,c),model.base(s,z))
    c[:,14]=1
    assert not torch.equal(model(s,z,n,c),model.base(s,z))
    c[:,14]=0;c[:,10]=.2;c[:,13]=1
    assert not torch.equal(model(s,z,n,c),model.base(s,z))


@pytest.mark.parametrize('geometry_only',[False,True])
def test_encoder_padding_does_not_encode_length_of_storage(model,geometry_only):
    encoder=SkillEncoder(model.base,hidden=64 if geometry_only else 128,geometry_only=geometry_only).eval()
    states=torch.randn(2,51,47);actions=torch.randn(2,51,5);ctx=torch.randn(2,51,20)
    bz=torch.randn(2,2);lengths=torch.tensor([19,37])
    mu,lv=encoder(states,actions,ctx,bz,lengths)
    for i,length in enumerate(lengths):
        states[i,length:]=1e4;actions[i,length:]=-1e4;ctx[i,length:]=1e4
    actual_mu,actual_lv=encoder(states,actions,ctx,bz,lengths)
    torch.testing.assert_close(mu,actual_mu,rtol=0,atol=0)
    torch.testing.assert_close(lv,actual_lv,rtol=0,atol=0)
    single,_=encoder(states[0:1,:19],actions[0:1,:19],ctx[0:1,:19],bz[:1],lengths[:1])
    torch.testing.assert_close(mu[:1],single,rtol=1e-5,atol=1e-6)


def test_native_physics_trace_is_observational():
    from cube_addons.env import make_env,reset_case,snapshot,restore,step_traced,close_env
    env=make_env()
    try:
        reset_case(env,9191);initial=snapshot(env)
        actions=np.random.default_rng(5).uniform(-.5,.5,(12,5))
        for a in actions:env.step(a)
        expected=snapshot(env)
        restore(env,initial)
        for a in actions:
            _,path=step_traced(env,a)
            assert len(path)==26
        np.testing.assert_array_equal(snapshot(env),expected)
    finally:close_env(env)


def test_swept_sphere_detects_crossing_between_endpoints():
    # Both endpoints are outside; the segment crosses the forbidden center.
    assert segment_distance(np.array([-1.,0,0]),np.array([1.,0,0]),np.zeros(3))==0.
    assert segment_distance(np.array([-1.,2,0]),np.array([1.,2,0]),np.zeros(3))==2.


def test_native_bilateral_support_accepts_a_wide_but_secure_grasp(model):
    from cube_addons.env import make_env,close_env
    from cube_addons.probe import case,constraint
    from cube_addons.rollout import rollout
    env=make_env()
    try:
        c,style=case(env,610004)
        row,trajectory=rollout(env,model.base,c,constraint(c,style,'keepout',610004),'keepout',[1.],force_contacts=True)
        # This exact original-policy case was falsely rejected by the old
        # >0.5 gripper-opening proxy despite a normal bilateral carry.
        assert row['released_success'] and row['has_lifted']
        assert row['held_samples']>=3
        assert trajectory['poses'][:,5].max()>.15
    finally:close_env(env)


@pytest.mark.parametrize('transport_gate',[False,True])
@pytest.mark.parametrize('output_rank',[8,64])
def test_capacity_variant_logits_keep_exact_initial_actions(transport_gate,output_rank):
    model=GatedGenerator('cube/artifacts/final/base_policy.pt',AdapterConfig(rank=8,output_modulation=True,output_rank=output_rank,transport_gate=transport_gate))
    s,z,n,c=inputs();a,raw=model(s,z,n,c,return_logits=True)
    assert torch.equal(a,model.base(s,z))
    assert torch.equal(a,raw.tanh())
    off,raw_off=model(s,z,gate=0,return_logits=True)
    assert torch.equal(off,model.base(s,z)) and torch.equal(off,raw_off.tanh())


def test_stable_grip_loss_corrects_confident_wrong_commands():
    from cube_addons.losses import reconstruction
    raw=torch.tensor([[[0.,0.,0.,0.,12.]]],requires_grad=True)
    target=torch.tensor([[[0.,0.,0.,0.,-1.]]]);mask=torch.ones(1,1)
    loss=reconstruction(raw.tanh(),target,mask,raw);loss.backward()
    assert torch.isfinite(loss) and raw.grad[0,0,4]>.29


def test_stationary_contacts_cannot_mask_unsupported_transport():
    start=np.array([0.,0.,.2]);goal=np.array([1.,0.,.2])
    observer=Observer(start,goal,np.array([5.,5.,.2,.02]),'keepout',holding_fn=lambda info:info['held'])
    previous=start
    for x,held in [(.25,True),(.251,True),(.252,True),(.7,False),(.75,True),(.76,True)]:
        after=np.array([x,0.,.2]);observer.update(previous,after,{'held':held});previous=after
    assert observer.held_transport_samples>=3 and not observer.table_violation and not observer.violated
    assert observer.supported_fraction()<.8 and not observer.satisfied()


def test_disabled_deployment_preserves_sampling_and_actions(model,tmp_path):
    from dataclasses import asdict
    from pathlib import Path
    import hashlib
    from cube.cube_gan.policy import CubePolicy
    from cube_addons.policy import CubeSkillPolicy
    base='cube/artifacts/final/base_policy.pt';checkpoint=tmp_path/'addon.pt'
    encoder=SkillEncoder(model.base)
    torch.save({'adapter_config':asdict(model.config),'addon':model.addon_state(),'encoder':encoder.state_dict(),
                'base_sha256':hashlib.sha256(Path(base).read_bytes()).hexdigest(),'execute_steps':4},checkpoint)
    original=CubePolicy(base);extended=CubeSkillPolicy(checkpoint,base)
    rng_a=np.random.default_rng(31);rng_b=np.random.default_rng(31);states=np.random.default_rng(12)
    for _ in range(4):
        z=original.reset(rng=rng_a);other,_=extended.reset(gate=0,rng=rng_b)
        np.testing.assert_array_equal(z,other)
        for _ in range(10):
            state=states.standard_normal(47).astype(np.float32)
            np.testing.assert_array_equal(original.act(state),extended.act(state))
    assert rng_a.bit_generator.state==rng_b.bit_generator.state


def test_geometry_conditioning_ignores_elapsed_time():
    model=GatedGenerator('cube/artifacts/final/base_policy.pt',AdapterConfig(rank=8,output_modulation=True,output_rank=64,ignore_time=True))
    torch.nn.init.normal_(model.output_modulation[-1].weight,std=.02)
    s,z,n,c=inputs();other=c.clone();other[:,16]+=4
    assert torch.equal(model(s,z,n,c),model(s,z,n,other))
    assert torch.equal(model(s,z,gate=0),model.base(s,z))


def test_aligned_demonstration_chunks_start_at_real_replanning_boundaries(tmp_path):
    from cube_addons.data import EpisodeDataset
    from cube_addons.geometry import MAX_STEPS
    path=tmp_path/'train.npz';lengths=np.array([19,37])
    states=np.zeros((2,MAX_STEPS,47),np.float32)
    states[:,:,0]=np.arange(MAX_STEPS)
    actions=np.zeros((2,MAX_STEPS,5),np.float32)
    actions[:,:,0]=np.arange(MAX_STEPS)
    np.savez(path,states=states,actions=actions,contexts=np.zeros((2,MAX_STEPS,20),np.float32),
             lengths=lengths,new_z=np.ones((2,1),np.float32),base_z=np.ones((2,2),np.float32))
    data=EpisodeDataset(path,'cpu',alignment=8);batch=data.sample(64,4)
    starts=batch['states'][...,0]
    assert torch.all(starts.remainder(8)==0)
    assert torch.all(starts<batch['lengths'][:,None])
    assert torch.equal(batch['actions'][:,:,0,0],starts)
    assert torch.equal(batch['mask'].bool(),starts[...,None]+torch.arange(16)<batch['lengths'][:,None,None])


def test_compact_calibration_uses_all_episodes_with_partial_raw_examples(tmp_path):
    from cube_addons.calibrate import physical_data
    arrays={'states':np.zeros((3,240,47),np.float32),'actions':np.zeros((3,240,5),np.float32),
            'contexts':np.zeros((3,240,20),np.float32),'lengths':np.array([10,20,30]),
            'base_z':np.ones((3,2),np.float32),'new_z':np.array([[-1.],[.3],[1.7]],np.float32)}
    np.savez_compressed(tmp_path/'episodes.npz',**arrays,original_trajectory_sha256=np.array('a'*64))
    np.savez_compressed(tmp_path/'learner_0000.npz',states=arrays['states'][0,:10],actions=arrays['actions'][0,:10],
                        contexts=arrays['contexts'][0,:10],base_z=arrays['base_z'][0],new_z=arrays['new_z'][0])
    restored,digest=physical_data(tmp_path,'cpu')
    assert digest=='a'*64
    assert len(restored['states'])==3
    for key,value in arrays.items():np.testing.assert_array_equal(restored[key].numpy(),value)
