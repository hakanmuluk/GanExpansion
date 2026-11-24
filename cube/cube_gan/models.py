"""State-conditioned, style-modulated action generator, episode encoder, and D."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


@dataclass
class ModelConfig:
    obs_dim: int = 47
    action_dim: int = 5
    horizon: int = 16
    z_dim: int = 2
    hidden: int = 384
    style_dim: int = 128
    blocks: int = 4
    encoder_hidden: int = 128
    route_feature: bool = True


class Normalized(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def normalize(self, x):
        return ((x-self.obs_mean)/self.obs_std).clamp(-10, 10)


class ModulatedBlock(nn.Module):
    def __init__(self, hidden, style_dim):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.style = nn.Linear(style_dim, 2*hidden)
        nn.init.normal_(self.style.weight, std=0.02)
        nn.init.zeros_(self.style.bias)

    def forward(self, h, w):
        scale, shift = self.style(w).chunk(2, -1)
        return (h + F.leaky_relu(self.norm(self.linear(h))*(1+scale)+shift, .2))*2**-.5


class Generator(Normalized):
    def __init__(self, config):
        super().__init__(config.obs_dim)
        self.config = config
        self.mapping = nn.Sequential(nn.Linear(config.z_dim+int(config.route_feature), config.style_dim), nn.LeakyReLU(.2),
                                     nn.Linear(config.style_dim, config.style_dim), nn.LeakyReLU(.2),
                                     nn.Linear(config.style_dim, config.style_dim))
        self.state_input = nn.Linear(config.obs_dim, config.hidden)
        self.blocks = nn.ModuleList([ModulatedBlock(config.hidden, config.style_dim) for _ in range(config.blocks)])
        self.to_actions = nn.Linear(config.hidden, config.horizon*config.action_dim)
        nn.init.normal_(self.to_actions.weight, std=.01)
        nn.init.zeros_(self.to_actions.bias)

    def forward(self, state, z):
        leading = state.shape[:-1]
        h = F.leaky_relu(self.state_input(self.normalize(state).reshape(-1, self.config.obs_dim)), .2)
        z = z.reshape(-1, self.config.z_dim)
        if self.config.route_feature:
            z = torch.cat([z, torch.sign(z[:, :1])], -1)
        w = self.mapping(z)
        for block in self.blocks:
            h = block(h, w)
        return self.to_actions(h).tanh().reshape(*leading, self.config.horizon, self.config.action_dim)


class TrajectoryEncoder(Normalized):
    """Encode masked complete episodes; pooling excludes padded timesteps."""
    def __init__(self, config):
        super().__init__(config.obs_dim)
        h = config.encoder_hidden
        self.config = config
        self.temporal = nn.Sequential(nn.Conv1d(config.obs_dim+config.action_dim,h,5,stride=2,padding=2),nn.LeakyReLU(.2),
                                      nn.Conv1d(h,h,5,stride=2,padding=2),nn.LeakyReLU(.2),
                                      nn.Conv1d(h,h,3,padding=1),nn.LeakyReLU(.2))
        self.head = nn.Sequential(nn.Linear(2*h+config.obs_dim+2,h),nn.LeakyReLU(.2))
        self.mu = nn.Linear(h,config.z_dim)
        self.logvar = nn.Linear(h,config.z_dim)
        nn.init.constant_(self.logvar.bias,-3)

    def forward(self, states, actions, lengths):
        t = actions.shape[1]
        valid = torch.arange(t,device=states.device)[None] < lengths[:,None]
        tokens = torch.cat([self.normalize(states[:,:t]),actions],-1)*valid[...,None]
        h = self.temporal(tokens.transpose(1,2))
        n = (lengths+3)//4
        mask = torch.arange(h.shape[-1],device=h.device)[None] < n[:,None]
        mean = (h*mask[:,None]).sum(-1)/n.clamp_min(1)[:,None]
        maximum = h.masked_fill(~mask[:,None],-1e4).amax(-1)
        magnitude = actions[...,:3].norm(dim=-1)
        speed = (magnitude*valid).sum(-1)/lengths.clamp_min(1)
        moments = torch.stack([speed,lengths.to(states.dtype)/200],-1)
        pooled = self.head(torch.cat([mean,maximum,self.normalize(states[:,0]),moments],-1))
        return self.mu(pooled), self.logvar(pooled).clamp(-7,2)


class Discriminator(Normalized):
    def __init__(self, config):
        super().__init__(config.obs_dim)
        h = config.hidden
        d = config.obs_dim+config.horizon*(config.action_dim+1)
        self.net = nn.Sequential(spectral_norm(nn.Linear(d,h)),nn.LeakyReLU(.2),
                                 spectral_norm(nn.Linear(h,h)),nn.LeakyReLU(.2),
                                 spectral_norm(nn.Linear(h,h//2)),nn.LeakyReLU(.2),
                                 spectral_norm(nn.Linear(h//2,1)))

    def forward(self, states, actions, mask):
        return self.net(torch.cat([self.normalize(states),(actions*mask[...,None]).flatten(-2),mask],-1)).squeeze(-1)


def reconstruction(pred, target, mask):
    # Cartesian/yaw regression and binary gripper likelihood, with valid chunks.
    regression = ((pred[...,:4]-target[...,:4]).abs()*mask[...,None]).sum()/(4*mask.sum().clamp_min(1))
    p = ((pred[...,4]+1)/2).clamp(1e-5,1-1e-5)
    grip = (F.binary_cross_entropy(p,(target[...,4]+1)/2,reduction="none")*mask).sum()/mask.sum().clamp_min(1)
    return regression+.15*grip


def kl_loss(mu, logvar):
    return .5*(mu.square()+logvar.exp()-1-logvar).sum(-1).mean()
