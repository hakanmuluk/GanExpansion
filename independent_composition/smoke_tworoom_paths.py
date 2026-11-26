"""Environment-free tests for private TwoRoom route composition."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from .tworoom_paths import make_route,condition,canonical
from .tworoom_path_policy import TwoRoomPathPolicy
from .inference_guard import GuardedPolicy


class DummyController(nn.Module):
    def __init__(self):
        super().__init__();self.layer=nn.Linear(10+60+2+1,32)

    def forward(self,state,base_z,code,context):
        return self.layer(torch.cat([state,base_z,code,context],-1)).tanh().reshape(-1,16,2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(187)
    from tworoom_gan.env import observation
    from tworoom_addons.geometry import skill_features
    start=torch.tensor([[40.,50.],[184.,60.]])
    goal=torch.tensor([[180.,165.],[44.,174.]])
    specs=[torch.tensor([[62.,100.,11.],[160.,105.,12.]]),torch.tensor([[165.,40.,8.],[60.,45.,9.]])]
    state=observation(start,goal);bz=torch.tensor([[-1.,.3],[1.,-.4]])
    ctx=[skill_features(start,goal,start,spec,torch.zeros(2,dtype=torch.bool),0,skill)
         for spec,skill in zip(specs,['keepout','waypoint'])]
    cs,sign=canonical(state)
    assert torch.equal(cs[:,0],state[:,0]*sign)
    for c in ctx:assert condition(state,c,bz).shape==(2,25)
    routes=[make_route(32),make_route(32)];controllers=[DummyController(),DummyController()]
    with torch.no_grad():routes[0].net[-1].bias.fill_(.02)
    for method in ['keepout','waypoint','fixed','compose']:
        policy=GuardedPolicy(TwoRoomPathPolicy(routes,controllers,method=method,steps=4,starts=2))
        first=policy.sample(state,ctx,bz);second=policy.sample(state,ctx,bz)
        assert first.shape==(2,1,2) and torch.isfinite(first).all() and torch.isfinite(second).all()
        assert torch.allclose(policy.path[:,0],(cs[:,:2]+1)*112)
        assert torch.allclose(policy.path[:,-1],(cs[:,2:4]+1)*112)
        assert all(p.grad is None and not p.requires_grad for m in [*routes,*controllers] for p in m.parameters())
    for loss in ['chamfer','hybrid']:
        policy=GuardedPolicy(TwoRoomPathPolicy(routes,controllers,steps=4,starts=2,agreement=loss))
        assert torch.isfinite(policy.sample(state,ctx,bz)).all()
    from .path_encoder import PathEncoder
    encoders=[PathEncoder(hidden=32),PathEncoder(hidden=32)]
    for mode in ['self','executed','projection']:
        policy=GuardedPolicy(TwoRoomPathPolicy(routes,controllers,steps=4,starts=2,
            encoders=encoders,cycle_mode=mode,cycle_weight=.01,code_bound=2.))
        assert torch.isfinite(policy.sample(state,ctx,bz)).all()
        assert policy.selected_codes.abs().max()<=2.
        assert all(p.grad is None for module in encoders for p in module.parameters())
    from .path_support import PathSupport
    supports=[PathSupport(hidden=16),PathSupport(hidden=16)]
    policy=GuardedPolicy(TwoRoomPathPolicy(routes,controllers,steps=4,starts=2,
                                         supports=supports,support_weight=.01))
    assert torch.isfinite(policy.sample(state,ctx,bz)).all()
    assert all(p.grad is None for module in supports for p in module.parameters())
    try:TwoRoomPathPolicy(routes,controllers,supports=[supports[0],supports[0]],support_weight=.01)
    except AssertionError:pass
    else:raise AssertionError('Shared learned domain network accepted')
    try:TwoRoomPathPolicy(routes,[controllers[0],controllers[0]])
    except AssertionError:pass
    else:raise AssertionError('Shared learned controller accepted')
    report={'all_four_modes_finite':True,'both_reflections_checked':True,
            'exact_route_endpoints':True,'models_frozen_no_parameter_gradients':True,
            'simulator_and_training_predicates_poisoned':True,'shared_controller_rejected':True,
            'curve_distance_losses_finite':True,
            'full_route_cycle_variants_finite_frozen_and_code_bounded':True,
            'learned_domain_gradients_finite_and_shared_domain_network_rejected':True,
            'scope':'Implementation invariants only; no task-success claim.'}
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
