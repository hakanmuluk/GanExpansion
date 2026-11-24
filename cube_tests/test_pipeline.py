import numpy as np
import torch
from cube.cube_gan.env import TASK,make_env,reset_context,snapshot,restore,completion,features,close_env
from cube.cube_gan.expert import Style,select_action
from cube.cube_gan.models import ModelConfig,Generator,TrajectoryEncoder,Discriminator,reconstruction
from cube.cube_gan.style import encode_style,decode_style
from cube.cube_gan.data import context_specs
from cube.cube_gan.evaluate import select_contexts
from cube.cube_gan.metrics import carry_geometry
from cube.cube_gan.policy import CubePolicy


def test_seed_banks_are_disjoint():
    banks=[{s["seed"] for s in context_specs(1000,k)} for k in [1729,2741,3967]]
    assert all(len(b)==1000 for b in banks)
    assert not (banks[0]&banks[1] or banks[0]&banks[2] or banks[1]&banks[2])


def test_gaussian_style_roundtrip():
    z=torch.tensor([[-2.,-1.],[-.1,.5],[.8,2.]])
    mode,amp,speed=decode_style(z)
    torch.testing.assert_close(encode_style(mode,amp,speed),z,atol=2e-5,rtol=2e-5)


def test_stratified_validation_and_route_measure():
    tasks=np.array([0]*50+sum(([i]*10 for i in range(1,6)),[]))
    ids=select_contexts(tasks,32)
    assert len(ids)==32 and set(tasks[ids])==set(range(6))
    t=np.linspace(0,1,51)
    poses=np.zeros((51,6));poses[:,3]=.3+.2*t;poses[:,4]=.05*np.sin(np.pi*t);poses[:,5]=.2
    m=carry_geometry(poses,50,[.3,0,.02],[.5,0,.02])
    assert m["route"]==1 and abs(m["carry_amplitude"]-.05)<1e-6


def test_models_and_padding():
    torch.set_num_threads(1)
    c=ModelConfig(hidden=32,style_dim=16,encoder_hidden=16)
    g,e,d=Generator(c),TrajectoryEncoder(c),Discriminator(c)
    x=torch.randn(3,201,47);a=torch.randn(3,200,5).tanh();lengths=torch.tensor([60,80,100])
    mu,lv=e(x,a,lengths)
    pred=g(x[:,0],mu)
    assert pred.shape==(3,16,5) and pred.abs().max()<=1
    mask=torch.ones(3,16);mask[:,8:]=0
    altered=pred.detach().clone();altered[:,8:]=1
    torch.testing.assert_close(reconstruction(pred,pred.detach(),mask),reconstruction(pred,altered,mask))
    xx=x.clone();aa=a.clone()
    for j,n in enumerate(lengths):xx[j,n:]=1000;aa[j,n:]=-1000
    mm,ll=e(xx,aa,lengths)
    torch.testing.assert_close(mu,mm);torch.testing.assert_close(lv,ll)
    assert d(x[:,0],pred,mask).shape==(3,)
    (pred.mean()+mu.mean()).backward()
    assert g.blocks[0].style.weight.grad.abs().sum()>0


def test_deployment_replans_and_fixed_latent(tmp_path):
    c=ModelConfig(hidden=32,style_dim=16,encoder_hidden=16)
    g=Generator(c).eval()
    from dataclasses import asdict
    p=tmp_path/"policy.pt"
    torch.save({"model":asdict(c),"generator":g.state_dict()},p)
    policy=CubePolicy(p)
    z=np.array([-.7,.8],np.float32)
    policy.reset(z)
    x=np.random.default_rng(1).normal(size=(17,47)).astype(np.float32)
    with torch.inference_mode():
        for t in range(17):
            if t%8==0:chunk=g(torch.from_numpy(x[t])[None],torch.from_numpy(z)[None])[0].numpy()
            expected=chunk[t%8].copy();expected[4]=1 if expected[4]>=0 else -1
            np.testing.assert_allclose(policy.act(x[t]),expected,atol=1e-6)
            np.testing.assert_array_equal(policy.z,z)


def test_exact_real_physics_replay_and_expert():
    env=make_env()
    try:
        for task in range(6):
            ob,info=reset_context(env,900+task,task)
            start=info["privileged/block_0_pos"].copy();s=snapshot(env)
            reset_context(env,1900+task,(task+1)%6)
            ob,info=reset_context(env,900+task,task)
            np.testing.assert_array_equal(snapshot(env),s)
            assert features(env,ob,info,start).shape==(47,)
            rng=np.random.default_rng(task)
            sequence=rng.uniform(-.3,.3,(12,5))
            outputs=[]
            for _ in range(2):
                restore(env,s)
                for action in sequence:env.step(action)
                outputs.append(snapshot(env))
            np.testing.assert_allclose(outputs[0],outputs[1],atol=1e-10,rtol=0)
            for mode in range(2):
                ob,info=restore(env,s);stable=0
                for t in range(200):
                    action,_=select_action(env,info,start,Style(mode,.055,.85))
                    ob,_,_,_,info=env.step(action)
                    _,strict=completion(env,info)
                    stable=stable+1 if strict else 0
                    if stable>=10:break
                assert stable>=10,(task,mode)
    finally:close_env(env)
