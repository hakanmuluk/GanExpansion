from types import SimpleNamespace
import numpy as np
import torch
from cube_agreement.neural import Decoder,sample_training
from cube_agreement.action_agreement import refine
from cube_agreement.batch_search import refine_batch


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Linear(90,80)
    def forward(self,state,bz,z,context):
        return self.net(context).tanh().reshape(-1,16,5)


def test_batched_losses_and_adam_moments_remain_scene_independent():
    torch.manual_seed(71);rng=torch.Generator().manual_seed(83)
    _,g=sample_training('keepout',2,'cpu',rng);_,w=sample_training('waypoint',2,'cpu',rng);g[:,6:]=w[:,6:]
    bank=SimpleNamespace(models={s:Decoder(s).eval().requires_grad_(False) for s in ['keepout','waypoint']},actor=Actor().eval().requires_grad_(False))
    cases=[{'base_z':np.array([.2,.4],np.float32),'initial_cube':np.array([.42,.17,.02]),'goal':np.array([.4,-.17,.02])} for _ in range(2)]
    states=torch.randn(2,47);reference=torch.randn(2,20);cube=torch.tensor([[.41,.05,.2],[.4,-.01,.21]]);index=torch.tensor([20,34]);seeds=[31,89]
    for starts in [1,3]:
        for previous in [None,torch.randn(2,21)*.3]:
            combined,k,w,metrics=refine_batch(bank,g,cases,states,reference,cube,index,seeds,previous,starts=starts,steps=5)
            for i in range(2):
                one,pk,pw,metric=refine(bank,g[i:i+1],cases[i],states[i:i+1],reference[i:i+1],cube[i:i+1],index[i:i+1],None if previous is None else previous[i:i+1],starts=starts,steps=5,seed=seeds[i],continuity_weight=.001)
                torch.testing.assert_close(combined[i:i+1],one,rtol=1e-4,atol=1e-5)
                torch.testing.assert_close(k[i:i+1],pk,rtol=1e-4,atol=1e-6)
                torch.testing.assert_close(w[i:i+1],pw,rtol=1e-4,atol=1e-6)
                assert abs(metrics[i]['action_disagreement']-metric['action_disagreement'])<1e-6
