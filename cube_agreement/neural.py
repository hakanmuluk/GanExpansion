"""Learned independent single-skill manifold decoders.

The common spline layer gives stable global latent coordinates. Skill heads are
trained independently against their own constraints. No joint geometric cost or
post-decoding correction is used when the frozen neural models are composed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .manifolds import raw_path,NODES,FREE


class Decoder(nn.Module):
    def __init__(self,skill,hidden=128):
        super().__init__();self.skill=skill;self.hidden=hidden
        inputs=6 if skill=='keepout' else 3
        outputs=1 if skill=='keepout' else 2
        self.net=nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,outputs))
        if skill=='keepout':nn.init.constant_(self.net[-1].bias,.2)

    def forward(self,z,g):
        raw,t,weights=raw_path(z,g)
        if self.skill=='waypoint':
            v=z[:,:FREE].reshape(-1,5,2)[:,2]
            target=g[:,7:9]+.1*self.net(torch.cat([v,g[:,9,None]/.1],-1))
            return raw+weights[:,3][None,:,None]*(target-raw[:,NODES//2])[:,None]
        angle=z[:,FREE];direction=torch.stack([angle.cos(),angle.sin()],-1)
        signed=((raw-g[:,None,3:5])*direction[:,None]).sum(-1)/.1
        du=(t[None]*g[:,0,None]-g[:,2,None])/.1
        x=torch.stack([du,g[:,5,None].expand_as(du)/.1,g[:,0,None].expand_as(du)/(.1*(NODES-1)),signed,t[None].expand_as(du),g[:,0,None].expand_as(du)/.4],-1)
        amount=.1*F.relu(self.net(x).squeeze(-1))
        # Shared endpoint interpolation preserves the original pick/place columns.
        amount=amount*torch.sin(torch.pi*t)[None]
        out=raw+amount[:,:,None]*direction[:,None]
        return torch.cat([raw[:,:1],out[:,1:-1],raw[:,-1:]],1)


def sample_training(skill,n,device,rng):
    g=torch.zeros((n,10),device=device)
    u=lambda lo,hi,shape:(lo+(hi-lo)*torch.rand(shape,device=device,generator=rng))
    g[:,0]=u(.27,.42,(n,));g[:,1]=u(.035,.075,(n,))*torch.where(u(0,1,(n,))>.5,1.,-1.)
    if skill=='keepout':
        g[:,2]=g[:,0]*u(.30,.70,(n,));g[:,3]=g[:,1]+u(-.01,.01,(n,));g[:,4]=u(-.01,.01,(n,));g[:,5]=u(.018,.030,(n,))
    else:
        g[:,6]=.5*g[:,0];g[:,7]=g[:,1]+torch.sign(g[:,1])*u(.045,.075,(n,));g[:,8]=u(-.005,.005,(n,));g[:,9]=u(.020,.028,(n,))
    dimensions=FREE+int(skill=='keepout');z=torch.randn((n,dimensions),device=device,generator=rng)
    mix=u(0,1,(n,))>.5;z[mix,:FREE]=u(-3,3,(int(mix.sum()),FREE));z[:,:FREE].clamp_(-3,3)
    if skill=='keepout':z[:,FREE]=u(-torch.pi,torch.pi,(n,))
    return z,g
