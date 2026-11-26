"""Check observation binding, private parameters and tensor-only optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .private_paths import RouteConfig,RouteDecoder
from .path_policy import PrivatePathPolicy,observed_task


class DummyController(nn.Module):
    def __init__(self):
        super().__init__();self.layer=nn.Linear(47+91+2+1,80)

    def forward(self,state,base_z,code,context):
        return self.layer(torch.cat([state,base_z,code,context],-1)).tanh().reshape(-1,16,5)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(17)
    records=json.loads((Path(args.data)/'records.json').read_text())
    observations=[]
    for j,skill in enumerate(['keepout','waypoint']):
        row=next(r for r in records if r['skill']==skill and r['success'])
        with np.load(row['source']) as d:trace={k:d[k] for k in d.files}
        s=torch.tensor(trace['states'][:1]);ctx=torch.tensor(trace['contexts'][:1]);bz=torch.tensor(trace['base_z'])[None]
        start,goal,own,position=observed_task(s,[ctx,ctx],bz,[j])
        assert np.allclose(start[0],trace['initial_cube'],atol=1e-6)
        assert np.allclose(goal[0],trace['goal'],atol=1e-6)
        assert np.allclose(position[0],trace['poses'][0,3:],atol=1e-6)
        expected=trace['geometry'][:6] if j==0 else np.r_[trace['geometry'][:2],trace['geometry'][6:10]]
        assert np.allclose(own[j][0],expected,atol=1e-6)
        observations.append((s,ctx,bz))
    s,ctx,bz=observations[0]
    routes=[RouteDecoder(RouteConfig(11,hidden=32)),RouteDecoder(RouteConfig(10,hidden=32))]
    controllers=[DummyController(),DummyController()]
    with torch.no_grad():routes[0].net[-1].bias.fill_(.03)
    for method in ['keepout','waypoint','fixed','compose']:
        policy=PrivatePathPolicy(routes,controllers,method=method,steps=3,starts=2)
        first=policy.sample(s,[ctx,ctx],bz);second=policy.sample(s,[ctx,ctx],bz)
        assert first.shape==(1,16,5) and torch.isfinite(first).all() and torch.isfinite(second).all()
        assert all(v.grad is None and not v.requires_grad for m in [*routes,*controllers] for v in m.parameters())
        assert all(np.isfinite(v).all() for v in policy.trajectory_fields().values())
        # Completing transport must remain recorded while descending to place.
        policy.index.fill_(64)
        at_end=ctx.clone();spec=ctx[:,:3]*ctx.new_tensor([.2,.3,.2])+ctx.new_tensor([.425,0,.2])
        at_end[:,4:7]=(policy.path[:,-1]-spec)/.15
        policy.sample(s,[at_end,at_end],bz)
        assert bool(policy.done.all())
        descending=at_end.clone();descending[:,6]-=.05/.15
        policy.sample(s,[descending,descending],bz)
        assert bool(policy.done.all()),'Route completion was cleared during placement descent'
    try:PrivatePathPolicy(routes,[controllers[0],controllers[0]])
    except AssertionError:pass
    else:raise AssertionError('A shared learned controller was accepted')
    report={'observed_state_binding_matches_both_datasets':True,'all_four_modes_finite':True,
                      'frozen_parameters_no_gradients':True,'shared_controller_rejected':True,
                      'completion_stays_latched_during_descent':True,
                      'scope':'Environment-free invariants only; not evidence of task success.'}
    if args.out:
        target=Path(args.out)
        if target.exists():raise FileExistsError(target)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
