"""Private learned trajectory-domain scores, with no geometric predicates."""
import math
import torch
from torch import nn


class PathSupport(nn.Module):
    """Same segment/sequence architecture for either independently trained skill."""
    def __init__(self,context_dim=25,coordinates=2,hidden=128,points=65):
        super().__init__();self.context_dim=context_dim;self.coordinates=coordinates
        self.hidden=hidden;self.points=points
        self.point=nn.Sequential(nn.Linear(context_dim+2*coordinates+8,hidden),nn.SiLU(),
                                 nn.Linear(hidden,hidden),nn.SiLU())
        self.temporal=nn.Sequential(nn.Conv1d(hidden,hidden,5,padding=2),nn.SiLU(),
                                    nn.Conv1d(hidden,hidden,5,padding=2),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(2*hidden+context_dim,hidden),nn.SiLU(),nn.Linear(hidden,2))
        self.register_buffer('context_mean',torch.zeros(context_dim))
        self.register_buffer('context_std',torch.ones(context_dim))
        t=torch.linspace(0,1,points-1)
        features=torch.stack([t,t.square(),*[(2*math.pi*f*t).sin() for f in [1,2,4]],
                              *[(2*math.pi*f*t).cos() for f in [1,2,4]]],-1)
        self.register_buffer('time_features',features)

    def forward(self,path,context):
        assert path.shape[1:]==(self.points,self.coordinates)
        c=((context-self.context_mean)/self.context_std).clamp(-10,10)
        x=torch.cat([path[:,:-1],path[:,1:],c[:,None].expand(-1,self.points-1,-1),
                     self.time_features[None].expand(len(path),-1,-1)],-1)
        h=self.point(x).transpose(1,2);h=h+self.temporal(h)
        return self.head(torch.cat([h.mean(-1),h.amax(-1),c],-1))


def load_path_support(path,device='cuda'):
    ck=torch.load(path,map_location=device,weights_only=False)
    model=PathSupport(**ck['model_config']).to(device);model.load_state_dict(ck['state'])
    return model.eval().requires_grad_(False),ck
