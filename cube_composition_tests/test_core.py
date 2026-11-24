import copy
import numpy as np
import torch
from cube_composition.models import FrozenBank
from cube_composition.search import LatentSearch,SearchConfig
from cube_addons.env import make_env,reset_case,snapshot,restore,close_env,features,step_traced
from cube_addons.contacts import contact_info,force_holding
from cube_addons.probe import case as old_case,constraint
from cube_composition.geometry import JointObserver
from cube_composition.planner import joint_path,distance
from cube_composition.physics import capture,restore_branch


def make_case(env):
    c,style=old_case(env,2100021);c.update(kind=np.array('supported_easy'),keepout=constraint(c,style,'keepout',2100021),waypoint=constraint(c,style,'waypoint',2100021))
    return c


def test_frozen_bank_gradient_and_exact_single_gate():
    torch.set_num_threads(1);bank=FrozenBank(device='cpu');rng=np.random.default_rng(42)
    s=torch.from_numpy(rng.normal(size=(4,47)).astype('float32'));bz=torch.from_numpy(rng.normal(size=(4,2)).astype('float32'))
    ctx=(torch.zeros(4,20),torch.zeros(4,20));z=torch.tensor([[-.8,1.2]]*4,requires_grad=True)
    a,b=bank.pair(s,bz,z,ctx)
    torch.testing.assert_close(a,bank.summed(s,bz,z,ctx,(1,0)),atol=0,rtol=0)
    torch.testing.assert_close(b,bank.summed(s,bz,z,ctx,(0,1)),atol=0,rtol=0)
    torch.testing.assert_close(bank.base(s,bz),bank.summed(s,bz,z,ctx,(0,0)),atol=0,rtol=0)
    loss=(a-b).square().mean()+bank.summed(s,bz,z,ctx).square().mean();grad,=torch.autograd.grad(loss,z)
    assert torch.isfinite(grad).all() and grad.abs().sum()>0;bank.assert_frozen()


def test_native_midtrajectory_snapshot_branch_parity():
    env=make_env();clone=make_env()
    try:
        c=make_case(env);reset_case(clone,int(c['seed']));rng=np.random.default_rng(611)
        for _ in range(12):step_traced(env,rng.uniform(-.5,.5,5).astype('float32'))
        state=capture(env);restore_branch(clone,state)
        for _ in range(12):
            action=rng.uniform(-.5,.5,5).astype('float32');a,p=step_traced(env,action);b,q=step_traced(clone,action)
            np.testing.assert_allclose(p,q,atol=1e-9,rtol=0);np.testing.assert_allclose(snapshot(env),snapshot(clone),atol=1e-9,rtol=0)
    finally:close_env(env);close_env(clone)


def test_native_held_cube_branch_preserves_forces_and_robot_future():
    from cube_addons.geometry import Observer
    from cube_addons.expert import SkillExpert
    from cube_addons.probe import case as sample_case
    env=make_env();clone=make_env()
    try:
        c,style=sample_case(env,100001);spec=constraint(c,style,'keepout',100001);ob,info=restore(env,c['initial_state']);contact_info(env,info)
        observer=Observer(c['initial_cube'],c['goal'],spec,'keepout',force_holding);teacher=SkillExpert(c['initial_cube'],c['goal'],style,spec,[1.],'keepout')
        for _ in range(150):
            action,_=teacher.action(info,observer);before=info['privileged/block_0_pos'].copy();result,path=step_traced(env,action);ob,_,_,_,info=result;contact_info(env,info);observer.update(before,info['privileged/block_0_pos'],info,path)
            if force_holding(info) and info['privileged/block_0_pos'][2]>.16:break
        assert force_holding(info)
        reset_case(clone,int(c['seed']));restore_branch(clone,capture(env,ob,info))
        for _ in range(12):
            action,_=teacher.action(info,observer);before=info['privileged/block_0_pos'].copy();a,p=step_traced(env,action);b,q=step_traced(clone,action)
            ob,_,_,_,info=a;contact_info(env,info);contact_info(clone,b[4]);observer.update(before,info['privileged/block_0_pos'],info,p)
            np.testing.assert_allclose(snapshot(env),snapshot(clone),atol=1e-9,rtol=0)
            np.testing.assert_allclose(p,q,atol=1e-9,rtol=0)
            for key in ['addon/left_cube_normal_force','addon/right_cube_normal_force']:np.testing.assert_allclose(info[key],b[4][key],atol=1e-8,rtol=0)
    finally:close_env(env);close_env(clone)


def test_joint_geometric_paths_have_complete_cube_clearance():
    c={'initial_cube':np.array([.42,.17,.02]),'goal':np.array([.42,-.17,.02]),'keepout':np.array([.47,0,.2,.025]),'waypoint':np.array([.523,0,.2,.022])}
    p=joint_path(c);assert p is not None
    assert np.min(distance(p[:-1],p[1:],c['keepout'][:3]))>=.025+np.sqrt(3)*.02+.0049


def test_factorized_grid_matches_explicit_agreement_enumeration():
    torch.set_num_threads(1);bank=FrozenBank(device='cpu');s=np.zeros(47,np.float32);ctx=(np.zeros(20,np.float32),np.zeros(20,np.float32));bz=np.array([.7,-.3],np.float32)
    cfg=SearchConfig(method='grid',grid_size=8,regularizer='gaussian');planner=LatentSearch(bank,cfg);_,z,_=planner.plan(s,ctx,bz)
    vals=torch.linspace(-3,3,8);pairs=torch.cartesian_prod(vals,vals)
    with torch.no_grad():score,_=planner.objective(pairs,torch.tensor(s)[None],tuple(torch.tensor(v)[None] for v in ctx),torch.tensor(bz)[None])
    np.testing.assert_allclose(z,pairs[score.argmin()].numpy(),atol=0,rtol=0)


def test_parallel_native_forecast_matches_actual_neural_prefix_and_does_not_mutate_live_env():
    from cube_composition.simulator import forecast,close_pool
    from cube_composition.scene import prepare
    env=make_env();bank=FrozenBank(device='cpu')
    try:
        c=make_case(env);ob,info=prepare(env,c);contact_info(env,info);observer=JointObserver(c)
        state=features(env,ob,info,c['initial_cube']);ctx=observer.features(info,0);before=snapshot(env)
        runtime={'env':env,'ob':ob,'info':info,'observer':observer,'case':c,'previous_ee':None,'stable':0,'time':0}
        predicted=forecast(bank,state,ctx,c['base_z'],np.array([[.8,.8]],np.float32),['keepout'],runtime,horizon=8,workers=2)
        np.testing.assert_array_equal(snapshot(env),before)
        actual=[np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']]]
        for continuous in predicted['first_chunks'][0,:8]:
            action=continuous.copy();action[4]=1 if action[4]>=0 else -1;result,_=step_traced(env,action);info=result[4];actual.append(np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']])
        np.testing.assert_allclose(np.array(actual),predicted['paths'][0],atol=1e-9,rtol=0);bank.assert_frozen()
    finally:close_pool();close_env(env)


def test_native_task_search_executes_the_verified_prefix_and_keeps_models_frozen():
    from cube_composition.simulator import close_pool
    from cube_composition.scene import prepare
    env=make_env();bank=FrozenBank(device='cpu')
    try:
        c=make_case(env);ob,info=prepare(env,c);contact_info(env,info);observer=JointObserver(c)
        state=features(env,ob,info,c['initial_cube']);ctx=observer.features(info,0)
        runtime={'env':env,'ob':ob,'info':info,'observer':observer,'case':c,'previous_ee':None,'stable':0,'time':0}
        cfg=SearchConfig(method='random',objective='task',family='best',candidates=4,lookahead=16,workers=2,initialization='structured',temporal='fixed')
        planner=LatentSearch(bank,cfg);chunk,z,meta=planner.plan(state,ctx,c['base_z'],runtime)
        assert meta['native_controls']==4*3*16 and meta['family'] in ['keepout','waypoint','sum']
        assert np.isfinite(chunk).all() and chunk.shape==(16,5)
        first_z=z.copy();_,z2,meta2=planner.plan(state,ctx,c['base_z'],runtime)
        np.testing.assert_array_equal(first_z,z2);assert meta2['native_controls']==0
        bank.assert_frozen()
    finally:close_pool();close_env(env)


def test_common_action_extension_is_inactive_before_grasp_and_projects_held_motion():
    from cube_composition.common import project
    b=np.zeros((7,6));b[:3,:3]=b[:3,3:]=np.eye(3)*.025
    k=torch.zeros(1,20);w=k.clone();k[:,3]=.25;k[:,4]=.07/.15;k[:,13:15]=1
    state=torch.zeros(1,47);chunk=torch.zeros(1,16,5);chunk[:,:,0]=-.5;chunk[:,:,4]=1
    inactive=k.clone();inactive[:,13]=0
    torch.testing.assert_close(project(chunk,state,(inactive,w),b),chunk,atol=0,rtol=0)
    result=project(chunk,state,(k,w),b);r=.07+.025*result[0,:,0].cumsum(0)
    assert bool((r>=.025+np.sqrt(3)*.02+.005-1e-6).all())
    torch.testing.assert_close(result[:,:,3:],chunk[:,:,3:],atol=0,rtol=0)


def test_direct_action_native_cem_can_warm_start_after_a_plan():
    from cube_composition.simulator import close_pool
    from cube_composition.scene import prepare
    env=make_env();bank=FrozenBank(device='cpu')
    try:
        c=make_case(env);ob,info=prepare(env,c);contact_info(env,info);observer=JointObserver(c)
        state=features(env,ob,info,c['initial_cube']);ctx=observer.features(info,0)
        runtime={'env':env,'ob':ob,'info':info,'observer':observer,'case':c,'previous_ee':None,'stable':0,'time':0}
        planner=LatentSearch(bank,SearchConfig(method='action',objective='task',family='waypoint',candidates=4,steps=1,lookahead=8,workers=2))
        for _ in range(2):
            chunk,z,meta=planner.plan(state,ctx,c['base_z'],runtime)
            assert chunk.shape==(8,5) and np.isfinite(chunk).all() and meta['native_controls']==40
        bank.assert_frozen()
    finally:close_pool();close_env(env)


def test_frozen_encoders_accept_only_the_executed_causal_prefix():
    from cube_composition.scene import prepare
    env=make_env();bank=FrozenBank(device='cpu')
    try:
        c=make_case(env);ob,info=prepare(env,c);contact_info(env,info);observer=JointObserver(c);previous=None
        states=[];contexts=[];actions=[]
        for t in range(8):
            states.append(features(env,ob,info,c['initial_cube'],previous));contexts.append(np.stack(observer.features(info,t)))
            action=np.zeros(5,np.float32);actions.append(action);before=info['privileged/block_0_pos'].copy();previous=info['proprio/effector_pos'].copy()
            result,path=step_traced(env,action);ob,_,_,_,info=result;contact_info(env,info);observer.update(before,info['privileged/block_0_pos'],info,path,env)
        runtime={'history':{'states':np.asarray(states),'contexts':np.asarray(contexts),'actions':np.asarray(actions)}}
        cfg=SearchConfig(method='random',candidates=4,initialization='encoder_prefix',regularizer='none')
        priors={name:{'initial_codes':[-.8,.8]} for name in ['keepout','waypoint']};planner=LatentSearch(bank,cfg,priors)
        chunk,z,meta=planner.plan(features(env,ob,info,c['initial_cube'],previous),observer.features(info,8),c['base_z'],runtime)
        assert meta['encoder_calls']==2 and np.isfinite(meta['encoder_hint']).all() and np.isfinite(chunk).all()
        bank.assert_frozen()
    finally:close_env(env)
