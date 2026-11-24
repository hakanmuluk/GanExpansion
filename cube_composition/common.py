"""Explicit common-action extension: small held-regime barrier projection.

This uses known constraint geometry and a single-skill-only linear surrogate.
It is deliberately separate from pure latent composition. Physical rollout is
the authority; this local linear projection is not a formal safety guarantee.
"""
import torch


def project(chunk,state,ctx,coefficients,blend=None,margin=.005):
    k,w=ctx;center=k[:,:3]*k.new_tensor([.2,.3,.2])+k.new_tensor([.425,0,.2]);cube=center+.15*k[:,4:7]
    radius=.1*k[:,3]+3**.5*.02+margin;velocity=state[:,-3:]*.01
    active=(k[:,13]>.5)&(k[:,14]>.5)&(cube[:,2]>.10)
    b=torch.as_tensor(coefficients,device=chunk.device,dtype=chunk.dtype);jacobian=b[:3,3:]
    output=[]
    for t in range(chunk.shape[1]):
        action=chunk[:,t];xyz=action[:,:3]
        if blend is not None:xyz=.75*xyz+.25*blend[:,t,:3]
        offset=cube-center;distance=offset.norm(dim=-1).clamp_min(1e-6);normal=offset/distance[:,None]
        intercept=velocity@b[3:6,3:]+b[6,3:];direction=normal@jacobian.T
        desired=(xyz@jacobian+intercept)*normal;radial=desired.sum(-1)
        lower=-.5*(distance-radius);amount=(lower-radial).clamp_min(0)/direction.square().sum(-1).clamp_min(1e-8)
        corrected=(xyz+amount[:,None]*direction).clamp(-1,1)
        xyz=torch.where(active[:,None],corrected,action[:,:3]);a=torch.cat([xyz,action[:,3:]],-1)
        delta=torch.cat([xyz,velocity,torch.ones_like(velocity[:,:1])],-1)@b;cube=cube+delta[:,3:];velocity=delta[:,:3]
        output.append(a)
    return torch.stack(output,1)
