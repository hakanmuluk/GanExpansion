"""Each skill owns a separate encoder of CURRENT observation + future actions.

No future states, environment transitions, geometry costs, or shared parameters
are required by forward(). Log variance records local code ambiguity.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn


@dataclass
class WindowConfig:
    state_dim: int
    context_dim: int
    action_dim: int
    horizon: int = 16
    hidden: int = 256
    base_z_dim: int = 2
    code_dim: int = 1


class WindowEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        condition_dim = c.state_dim + c.context_dim + c.base_z_dim
        self.register_buffer("condition_mean", torch.zeros(condition_dim))
        self.register_buffer("condition_std", torch.ones(condition_dim))
        inputs = condition_dim + c.horizon * (c.action_dim + 1)
        self.net = nn.Sequential(nn.Linear(inputs, c.hidden), nn.SiLU(),
                                 nn.Linear(c.hidden, c.hidden), nn.SiLU(),
                                 nn.Linear(c.hidden, c.hidden), nn.SiLU())
        self.mean = nn.Linear(c.hidden, c.code_dim)
        self.log_variance = nn.Linear(c.hidden, c.code_dim)

    def forward(self, states, contexts, base_z, actions, mask=None):
        if mask is None:
            mask = torch.ones(actions.shape[:-1], device=actions.device, dtype=actions.dtype)
        condition = torch.cat([states, contexts, base_z], -1)
        condition = ((condition-self.condition_mean)/self.condition_std).clamp(-10, 10)
        x = torch.cat([condition, (actions*mask[..., None]).flatten(-2), mask], -1)
        h = self.net(x)
        return self.mean(h), self.log_variance(h).clamp(-4, 4)

    @torch.no_grad()
    def fit_normalization(self, condition):
        self.condition_mean.copy_(condition.mean(0))
        self.condition_std.copy_(condition.std(0).clamp_min(.01))


def load_encoder(path, device="cuda"):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = WindowEncoder(WindowConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["encoder"])
    return model.eval().requires_grad_(False), saved


def gaussian_nll(mean, log_variance, code):
    return .5*((mean-code).square()*(-log_variance).exp()+log_variance).mean()
