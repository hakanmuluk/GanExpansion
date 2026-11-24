"""Latent-only agreement between frozen independently trained skill models."""
import hashlib
import math
import torch
from .manifolds import FREE


def fingerprint(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def solve(models,geometry,starts=12,steps=120,seed=42,prior_weight=1e-5,objective='path'):
    before={key:fingerprint(model) for key,model in models.items()}
    for model in models.values():model.eval().requires_grad_(False)
    n=len(geometry);device=geometry.device;rng=torch.Generator(device=device).manual_seed(seed)
    g=geometry[:,None].expand(-1,starts,-1).reshape(-1,10)
    common=.5*torch.randn((n,starts,FREE),device=device,generator=rng)
    angles=torch.linspace(-math.pi,math.pi,starts+1,device=device)[:-1][None,:,None].expand(n,-1,-1)
    k=torch.nn.Parameter(torch.cat([common,angles],-1).reshape(-1,FREE+1));w=torch.nn.Parameter(common.clone().reshape(-1,FREE))
    optimizer=torch.optim.Adam([k,w],lr=.12);best=None;best_loss=None
    for step in range(steps+1):
        pk,pw=models['keepout'](k,g),models['waypoint'](w,g)
        if objective=='path':error=(pk-pw)/.1
        elif objective=='delta':error=(pk[:,1:]-pk[:,:-1]-pw[:,1:]+pw[:,:-1])/.01
        else:raise ValueError(objective)
        disagreement=error.square().mean((1,2));prior=k[:,:FREE].square().mean(-1)+w.square().mean(-1)
        values=disagreement+prior_weight*prior
        with torch.no_grad():
            loss,index=values.reshape(n,starts).min(-1);ids=torch.arange(n,device=device)*starts+index
            candidate=torch.cat([k[ids],w[ids]],-1).clone()
            if best is None:best=candidate;best_loss=loss.clone()
            else:
                update=loss<best_loss;best=torch.where(update[:,None],candidate,best);best_loss=torch.minimum(best_loss,loss)
        if step==steps:break
        optimizer.zero_grad();values.sum().backward();optimizer.step()
        with torch.no_grad():k[:,:FREE].clamp_(-3,3);w.clamp_(-3,3)
    with torch.no_grad():pk=models['keepout'](best[:,:FREE+1],geometry);pw=models['waypoint'](best[:,FREE+1:],geometry)
    for name,model in models.items():
        assert fingerprint(model)==before[name]
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return best,pk,pw,best_loss
