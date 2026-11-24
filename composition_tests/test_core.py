import numpy as np
import torch
from tworoom_composition.geometry import entry_fraction,wall_clearance,update,contexts
from tworoom_composition.planner import witness,native_step
from tworoom_gan.env import transition,observation


def case():
    return {'start':torch.tensor([[35.,112.]]),'goal':torch.tensor([[185.,110.]]),
            'base_z':torch.tensor([[-1.,.2]]),'keepout':torch.tensor([[65.,88.,12.]]),'waypoint':torch.tensor([[165.,170.,8.]])}


def test_swept_wall_check_detects_overlap_between_clear_endpoints():
    a=torch.tensor([[99.,72.]]);b=torch.tensor([[125.,72.]])
    assert wall_clearance(a,a)>0 and wall_clearance(b,b)>0
    assert wall_clearance(a,b)<-1.9


def test_within_control_step_event_order_is_not_lost():
    a=torch.tensor([[30.,50.]]);b=torch.tensor([[90.,50.]])
    c=case();c['goal']=torch.tensor([[76.,50.]]);c['waypoint']=torch.tensor([[84.,50.,4.]])
    flags=torch.zeros(1,dtype=torch.bool)
    assert update(a,b,c,flags,flags,flags,flags)[3].item()
    c['waypoint']=torch.tensor([[50.,50.,4.]])
    assert not update(a,b,c,flags,flags,flags,flags)[3].item()
    assert update(a,b,c,flags,flags,flags,flags)[0].item()


def test_independent_numpy_dynamics_matches_tensor_reference():
    rng=np.random.default_rng(41);positions=rng.uniform(21,203,(1000,2)).astype(np.float32);actions=rng.uniform(-1.5,1.5,(1000,2)).astype(np.float32)
    expected,hit=transition(torch.from_numpy(positions),torch.from_numpy(actions))
    actual=[native_step(p,a) for p,a in zip(positions,actions)]
    np.testing.assert_array_equal(np.stack([x[0] for x in actual]),expected.numpy())
    np.testing.assert_array_equal(np.array([x[1] for x in actual]),hit.numpy())


def test_cached_tensor_formulas_are_bitwise_equal_to_original_reference():
    from tworoom_composition.runtime import transition as cached_transition,observation as cached_observation
    rng=np.random.default_rng(193);p=torch.from_numpy(rng.uniform(21,203,(3000,2)).astype('float32'));a=torch.from_numpy(rng.uniform(-1.5,1.5,(3000,2)).astype('float32'))
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        pp=p.to(device);aa=a.to(device)
        x,h=transition(pp,aa);y,j=cached_transition(pp,aa)
        torch.testing.assert_close(x,y,atol=0,rtol=0);torch.testing.assert_close(h,j,atol=0,rtol=0)
        torch.testing.assert_close(observation(pp,pp.flip(0)),cached_observation(pp,pp.flip(0)),atol=0,rtol=0)


def test_geometric_witness_satisfies_strict_joint_task():
    c={k:v[0].numpy() for k,v in case().items()};proof=witness(c)
    assert proof is not None and proof['steps']<=240
    p=torch.from_numpy(proof['positions']);assert wall_clearance(p[:-1],p[1:]).min()>0
    flags=torch.zeros(1,dtype=torch.bool);visited=violated=wall=early=flags
    cc={k:torch.tensor(v)[None] for k,v in c.items()}
    for a,b in zip(p[:-1],p[1:]):visited,violated,wall,early,reached=update(a[None],b[None],cc,visited,violated,wall,early)
    assert visited and reached and not violated and not wall and not early


def test_vectorized_visibility_edges_match_independent_scalar_checker():
    from tworoom_composition.planner import clear_segment,clear_segments
    rng=np.random.default_rng(833);a=rng.uniform(21,203,(700,2));b=rng.uniform(21,203,(700,2))
    keep=np.array([65.,88.,12.]);goal=np.array([185.,110.])
    expected=np.array([clear_segment(x,y,keep,goal,0) for x,y in zip(a,b)])
    np.testing.assert_array_equal(clear_segments(a,b,keep,goal,0),expected)


def test_frozen_bank_gradients_and_exact_single_gate_reduction():
    from tworoom_composition.models import FrozenBank
    torch.set_num_threads(1);bank=FrozenBank(device='cpu');c=case();z=torch.tensor([[-.8,1.1]],requires_grad=True)
    ctx=contexts(c['start'],c,torch.zeros(1,dtype=torch.bool),0,torch.zeros(1,2),bank.dimensions)
    state=observation(c['start'],c['goal']);a,b=bank.pair(state,c['base_z'],z,ctx)
    torch.testing.assert_close(bank.additive(state,c['base_z'],z,ctx,(1,0)),a,atol=0,rtol=0)
    torch.testing.assert_close(bank.additive(state,c['base_z'],z,ctx,(0,1)),b,atol=0,rtol=0)
    torch.testing.assert_close(bank.additive(state,c['base_z'],z,ctx,(0,0)),bank.k.base(state,c['base_z']),atol=0,rtol=0)
    loss=(a-b).square().mean()+bank.additive(state,c['base_z'],z,ctx).square().mean()
    grad,=torch.autograd.grad(loss,z);assert torch.isfinite(grad).all() and grad.abs().sum()>0
    bank.assert_frozen()


def test_factorized_grid_equals_exhaustive_pair_search():
    from tworoom_composition.models import FrozenBank
    from tworoom_composition.search import SearchConfig,LatentSearch,repeat_case
    torch.set_num_threads(1);bank=FrozenBank(device='cpu');c=case();flags=torch.zeros(1,dtype=torch.bool)
    cfg=SearchConfig(method='grid',grid_size=8,regularizer='gaussian',prior_weight=.01,temporal='cold')
    planner=LatentSearch(bank,cfg);_,z,_=planner.plan(c['start'],c,flags,torch.zeros(1),torch.zeros(1,2))
    values=torch.linspace(-3,3,8);pairs=torch.cartesian_prod(values,values);cc=repeat_case(c,len(pairs))
    with torch.no_grad():score,_=planner.objective(pairs,cc['start'],cc,flags.repeat(len(pairs)),torch.zeros(len(pairs)),torch.zeros(len(pairs),2))
    torch.testing.assert_close(z[0],pairs[score.argmin()])


def test_direct_action_objective_has_finite_action_gradients():
    from tworoom_composition.action_search import ActionSearch
    planner=object.__new__(ActionSearch);c=case();actions=torch.full((1,16,2),.2,requires_grad=True)
    loss=planner.cost(actions,c['start'],c,torch.zeros(1,dtype=torch.bool));grad,=torch.autograd.grad(loss.sum(),actions)
    assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_cuda_graph_matches_eager_and_does_not_alias_previous_reports():
    if not torch.cuda.is_available():return
    from tworoom_composition.models import FrozenBank
    from tworoom_composition.search import SearchConfig,LatentSearch
    bank=FrozenBank();c={k:v.cuda() for k,v in case().items()};flags=torch.zeros(1,dtype=torch.bool,device='cuda')
    planner=LatentSearch(bank,SearchConfig(method='random',objective='task',family='sum',lookahead=16))
    z=torch.tensor([[-.8,1.1]],device='cuda');clock=torch.zeros(1,device='cuda');prev=torch.zeros(1,2,device='cuda')
    with torch.no_grad():
        expected,details=planner.objective(z,c['start'],c,flags,clock,prev,return_plan=True)
        planner.config.cuda_graph=True
        actual,captured=planner.objective(z,c['start'],c,flags,clock,prev,return_plan=True)
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(captured['_chunk'],details['_chunk'],atol=1e-6,rtol=1e-6)
        old=actual.clone();old_chunk=captured.pop('_chunk').clone()
        planner.objective(-z,c['start']+1,c,flags,clock+8,prev,return_plan=True)
        torch.testing.assert_close(actual,old,atol=0,rtol=0)
        assert old_chunk.shape==(1,16,2)
    bank.assert_frozen()
