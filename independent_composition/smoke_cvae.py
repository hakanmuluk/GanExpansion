"""Environment-free checks of training and composed inference invariants."""
import json
import torch
from torch import nn
from .cvae import ActionManifold, CVAEConfig, ManifoldComposer
from .train_cvae import losses


class DummyBase(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__(); self.action_dim=action_dim
        self.fc=nn.Linear(state_dim+2,16*action_dim)

    def forward(self,state,code):
        return self.fc(torch.cat([state,code],-1)).tanh().reshape(-1,16,self.action_dim)


def main():
    torch.set_num_threads(2);torch.manual_seed(11);results=[]
    for state_dim,context_dim,action_dim in [(10,12,2),(47,20,5)]:
        base=DummyBase(state_dim,action_dim).eval().requires_grad_(False)
        config=CVAEConfig(state_dim,context_dim,action_dim,hidden=32)
        models=[ActionManifold(config),ActionManifold(config)]
        state=torch.randn(2,state_dim);ctx=(torch.randn(2,context_dim),torch.randn(2,context_dim));z=torch.randn(2,2)
        ba=base(state,z)
        for model,context in zip(models,ctx):
            condition=model.features(state,context,z,ba);code=torch.randn(2,4)
            assert torch.equal(model.decode(condition,ba,code,enabled=0),model.extended_base(ba))
            with torch.no_grad():model.output.weight.normal_(0,.01)
            mask=torch.ones(2,32);mask[1,20:]=0
            batch={'state':state,'context':context,'base_z':z,
                   'actions':torch.randn(2,32,action_dim).clamp(-1,1),'mask':mask}
            loss,*_=losses(model,batch,ba);loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            model.zero_grad(set_to_none=True)
        for method in ['keepout','waypoint','fixed','compose']:
            policy=ManifoldComposer(models,base,method=method,steps=3,starts=2)
            action=policy.sample(state,ctx,z)
            assert action.shape==(2,32,action_dim) and torch.isfinite(action).all() and action.abs().max()<=1
            second=policy.sample(state,ctx,z);assert torch.isfinite(second).all()
            assert all(p.grad is None and not p.requires_grad for m in [base,*models] for p in m.parameters())
        for slot,method in [(0,'keepout'),(1,'waypoint')]:
            private=[None,None];private[slot]=models[slot]
            result=ManifoldComposer(private,base,method=method).sample(state,ctx,z)
            assert torch.isfinite(result).all()
        results.append({'state_dim':state_dim,'context_dim':context_dim,'action_dim':action_dim,
                        'disabled_identity':True,'finite_training_gradients':True,
                        'all_inference_modes_and_warm_start_finite':True,'frozen_independent_models':True})
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
