"""Private full-physical-route code encoder; no geometry or environment queries."""
import torch
from torch import nn


class PathEncoder(nn.Module):
    def __init__(self,context_dim=25,code_dim=10,hidden=256):
        super().__init__();self.context_dim=context_dim;self.code_dim=code_dim;self.hidden=hidden
        self.net=nn.Sequential(nn.Linear(context_dim+130,hidden),nn.SiLU(),nn.Linear(hidden,hidden),
                               nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,code_dim))
        self.register_buffer('context_mean',torch.zeros(context_dim))
        self.register_buffer('context_std',torch.ones(context_dim))

    def forward(self,path,context):
        c=((context-self.context_mean)/self.context_std).clamp(-10,10)
        return 3*torch.tanh(self.net(torch.cat([path.flatten(1),c],-1))/3)


def load_path_encoder(path,device='cuda'):
    saved=torch.load(path,map_location=device,weights_only=False)
    model=PathEncoder(**saved['model_config']).to(device);model.load_state_dict(saved['state'])
    return model.eval().requires_grad_(False),saved
