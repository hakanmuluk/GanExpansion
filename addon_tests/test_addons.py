import torch
import numpy as np
import pytest
from tworoom_gan.env import observation
from tworoom_addons.models import GatedGenerator,AdapterConfig,SkillEncoder
from tworoom_addons.geometry import segment_distance,update_constraint,skill_features,constraint_loss


def test_swept_body_and_waypoint_checks():
    a=torch.tensor([[-20.,0.],[-20.,10.],[-3.,1.9]])
    b=torch.tensor([[20.,0.],[20.,10.],[3.,1.9]])
    centers=torch.zeros_like(a)
    torch.testing.assert_close(segment_distance(a,b,centers),torch.tensor([0.,10.,1.9]))
    spec=torch.tensor([[0.,0.,2.]]).repeat(3,1)
    visited,violated=update_constraint(a,b,spec,torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool),"keepout")
    assert violated.tolist()==[True,False,True]
    visited,_=update_constraint(a,b,spec,visited,violated,"waypoint")
    assert visited.tolist()==[True,False,True]


@pytest.mark.parametrize("config",[AdapterConfig(),AdapterConfig(rank=8,context_hidden=128,modulation_rank=16,output_modulation=True)])
def test_exact_initialization_switch_off_and_frozen_updates(config):
    torch.set_num_threads(1);torch.manual_seed(7)
    m=GatedGenerator("artifacts/final/base_policy.pt",config,"cpu")
    start=torch.rand(48,2)*160+30;goal=torch.rand(48,2)*160+30
    state=observation(start,goal);z=torch.randn(48,2);zk=torch.randn(48,1);ctx=torch.randn(48,12)
    before=m.frozen_digest()
    expected=m.base(state,z)
    torch.testing.assert_close(m(state,z,zk,ctx),expected,atol=1e-7,rtol=1e-7)
    assert torch.equal(m(state,z,gate=0),expected)
    assert not any(p.requires_grad for p in m.base.parameters())
    assert not any(p.requires_grad for p in m.original_encoder.parameters())
    opt=torch.optim.Adam(m.trainable(),lr=.001)
    for _ in range(3):
        m.train();opt.zero_grad(set_to_none=True)
        loss=m(state,z,zk,ctx).square().mean();loss.backward();opt.step()
    assert before==m.frozen_digest()
    assert all(p.grad is None for p in m.base.parameters())
    assert not m.base.training and not m.original_encoder.training
    assert torch.equal(m(state,z,gate=0),expected)
    assert not torch.equal(m(state,z,zk,ctx),expected)
    gates=(torch.arange(48)%2).float()
    mixed=m(state,z,zk,ctx,gate=gates)
    assert torch.equal(mixed[gates==0],expected[gates==0])
    assert m.mapping[0].weight.grad.abs().sum()>0


def test_encoder_excludes_all_padding():
    e=SkillEncoder(hidden=32)
    states=torch.randn(3,241,10);actions=torch.randn(3,240,2);context=torch.randn(3,241,12)
    lengths=torch.tensor([50,100,150]);z=torch.randn(3,2)
    a,b=e(states,actions,context,z,lengths)
    for j,n in enumerate(lengths):
        states[j,n:]=900;actions[j,n:]=-300;context[j,n:]=600
    aa,bb=e(states,actions,context,z,lengths)
    torch.testing.assert_close(a,aa);torch.testing.assert_close(b,bb)


def test_no_new_latent_comparator_ignores_new_codes():
    m=GatedGenerator("artifacts/final/base_policy.pt",AdapterConfig(no_new_latent=True),"cpu")
    with torch.no_grad():
        for mod in m.modulations:mod[-1].weight.normal_(0,.05)
    state=observation(torch.tensor([[40.,110.]]),torch.tensor([[180.,110.]]))
    z=torch.randn(1,2);ctx=torch.randn(1,12)
    assert torch.equal(m(state,z,torch.tensor([[-2.]]),ctx),m(state,z,torch.tensor([[2.]]),ctx))


def test_constraint_objectives_ignore_padding_and_repeated_waiting():
    spec=torch.tensor([[100.,100.,8.]])
    positions=torch.tensor([[[0.,0.],[1.,1.],[100.,100.],[100.,100.]]])
    goal=torch.zeros(1,2)
    short=constraint_loss(positions,goal,spec,"waypoint",torch.tensor([1]))
    long=constraint_loss(positions,goal,spec,"waypoint",torch.tensor([2]))
    assert short>100 and long==0
    stationary=torch.full((1,241,2),110.)
    one=constraint_loss(stationary[:,:2],goal,spec,"waypoint",torch.tensor([1]))
    many=constraint_loss(stationary,goal,spec,"waypoint",torch.tensor([240]))
    torch.testing.assert_close(one,many)


def test_history_is_previous_executed_action_without_planned_targets():
    from tworoom_addons.rollout import rollout
    m=GatedGenerator("artifacts/final/base_policy.pt",AdapterConfig(context_dim=14),"cpu")
    start=torch.tensor([[40.,110.],[184.,110.]])
    goal=torch.tensor([[180.,100.],[44.,100.]])
    z=torch.tensor([[-1.,0.],[-1.,0.]])
    spec=torch.tensor([[165.,170.,8.],[59.,170.,8.]])
    with torch.no_grad():r=rollout(m,start,goal,z,torch.ones(2,1),spec,"waypoint",steps=16)
    assert torch.equal(r["contexts"][:,0,-2:],torch.zeros(2,2))
    expected=r["actions"][:,:-1].clone();expected[1,:,0]*=-1
    torch.testing.assert_close(r["contexts"][:,1:,-2:],expected)
    assert r["contexts"].shape[-1]==14


def test_finite_feedback_does_not_assume_demonstration_waypoint_progress(monkeypatch):
    import tworoom_addons.feedback as fb
    from tworoom_gan.env import transition
    start=torch.tensor([[165.,150.]]);goal=torch.tensor([[165.,130.]])
    spec=torch.tensor([[165.,170.,8.]])
    points=torch.tensor([[[165.,150.],[165.,158.],[165.,164.],[165.,160.],[165.,156.],[165.,152.],[165.,148.],[165.,144.]]])
    points=torch.cat([points,points[:,-1:].expand(-1,232,-1)],1)
    ectx=torch.zeros(1,240,12);ectx[:,2:,8]=1
    b={"start":start,"goal":goal,"spec":spec,"base_z":torch.zeros(1,2),"new_z":torch.ones(1,1),
       "lengths":torch.tensor([8]),"episode_states":observation(points,goal[:,None].expand(-1,240,-1)),
       "episode_contexts":ectx,"episode_actions":torch.tensor([[[0.,.8]]]).expand(1,240,2)}
    r={"states":observation(start,goal)[:,None].expand(-1,8,-1),"contexts":torch.zeros(1,8,12),
       "positions":start[:,None].expand(-1,9,-1),"lengths":torch.tensor([8])}
    monkeypatch.setattr(fb,"rollout",lambda *a,**kw:r)
    *_,targets=fb.finite_path_feedback(None,b,"waypoint")
    pos=start.clone();minimum=1000.
    for a in targets[0]:
        nxt,_=transition(pos,a[None]);minimum=min(minimum,float(segment_distance(pos,nxt,spec[:,:2])));pos=nxt
    assert minimum<=8,"Finite feedback must visit before tracking the stored post-visit segment"
