"""Modular base generator, full-episode encoder, and conditional discriminator.

No adapter/composition logic and no BatchNorm. A future adapter can attach to
the explicit MappingNetwork/ModulatedBlock boundaries without changing state
normalization or the base model. Gaussian latent magnitudes are preserved.
"""

from dataclasses import asdict, dataclass
import torch
from torch import nn, Tensor
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F

from .env import OBS_DIM, HORIZON
from .style import canonical_state


@dataclass
class ModelConfig:
    z_dim: int = 2
    style_dim: int = 128
    hidden: int = 256
    blocks: int = 4
    encoder_hidden: int = 128
    horizon: int = HORIZON
    encoder_moments: bool = False
    canonical: bool = False
    route_feature: bool = False
    semantic_style: bool = False

    def to_dict(self):
        return asdict(self)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, style_dim, route_feature=False):
        super().__init__()
        self.route_feature = route_feature
        self.net = nn.Sequential(nn.Linear(z_dim + int(route_feature), style_dim), nn.LeakyReLU(0.2),
                                 nn.Linear(style_dim, style_dim), nn.LeakyReLU(0.2),
                                 nn.Linear(style_dim, style_dim), nn.LeakyReLU(0.2))

    def forward(self, z):
        if self.route_feature:
            # Explicit sign feature permits a discontinuity between two valid
            # routes. Continuous z is also retained for within-route variation.
            z = torch.cat([z, torch.where(z[..., :1] >= 0, 1.0, -1.0)], -1)
        return self.net(z)


class ModulatedBlock(nn.Module):
    def __init__(self, hidden, style_dim):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.style = nn.Linear(style_dim, 2 * hidden)
        nn.init.normal_(self.style.weight, std=0.02)
        nn.init.zeros_(self.style.bias)

    def forward(self, h, w):
        scale, shift = self.style(w).chunk(2, -1)
        update = F.leaky_relu(self.norm(self.linear(h)) * (1 + scale) + shift, 0.2)
        return (h + update) * (2 ** -0.5)


class Generator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.mapping = MappingNetwork(config.z_dim, config.style_dim, config.route_feature)
        self.state_input = nn.Linear(OBS_DIM, config.hidden)
        self.blocks = nn.ModuleList([ModulatedBlock(config.hidden, config.style_dim) for _ in range(config.blocks)])
        self.to_actions = nn.Linear(config.hidden, config.horizon * 2)
        nn.init.normal_(self.to_actions.weight, std=0.01)
        nn.init.zeros_(self.to_actions.bias)

    def forward(self, state: Tensor, z: Tensor) -> Tensor:
        leading = state.shape[:-1]
        if self.config.canonical:
            state, reflection = canonical_state(state)
        h = F.leaky_relu(self.state_input(state.reshape(-1, OBS_DIM)), 0.2)
        w = self.mapping(z.reshape(-1, self.config.z_dim))
        for block in self.blocks:
            h = block(h, w)
        actions = torch.tanh(self.to_actions(h)).reshape(*leading, self.config.horizon, 2)
        if self.config.canonical:
            actions = actions * torch.stack([reflection, torch.ones_like(reflection)], -1)[..., None, :]
        return actions


class TrajectoryEncoder(nn.Module):
    """Encode complete, masked episodes rather than independent action chunks.

    Temporal convolutions operate on state/action pairs sampled every 2 steps.
    Both real and latent-cycle inputs use the same representation and masks.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        h = config.encoder_hidden
        self.config = config
        self.temporal = nn.Sequential(
            nn.Conv1d(OBS_DIM + 2, h, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(h, h, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(h, h, 3, padding=1), nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(nn.Linear(h * 2 + OBS_DIM + (2 if config.encoder_moments else 0), h), nn.LeakyReLU(0.2))
        self.mu = nn.Linear(h, config.z_dim)
        self.logvar = nn.Linear(h, config.z_dim)
        nn.init.constant_(self.logvar.bias, -2.0)

    def forward(self, states: Tensor, actions: Tensor, lengths: Tensor):
        if self.config.canonical:
            states, reflection = canonical_state(states)
            actions = actions * torch.stack([reflection, torch.ones_like(reflection)], -1)
        t = min(states.shape[1], actions.shape[1])
        tokens = torch.cat([states[:, :t:2], actions[:, :t:2]], -1)
        token_lengths = (lengths + 1) // 2
        mask = torch.arange(tokens.shape[1], device=states.device)[None] < token_lengths[:, None]
        tokens = tokens * mask[..., None]
        h = self.temporal(tokens.transpose(1, 2))
        down_lengths = (token_lengths + 3) // 4
        valid = torch.arange(h.shape[-1], device=h.device)[None] < down_lengths[:, None]
        mean = (h * valid[:, None]).sum(-1) / down_lengths.clamp_min(1)[:, None]
        maximum = h.masked_fill(~valid[:, None], -1e4).amax(-1)
        summary = [mean, maximum, states[:, 0]]
        if self.config.encoder_moments:
            # Unsupervised episode-level velocity moments make speed variation
            # visible to E without route labels or hand-assigned latent targets.
            full_valid = torch.arange(t, device=states.device)[None] < lengths[:, None]
            magnitude = actions[:, :t].norm(dim=-1)
            speed_mean = (magnitude * full_valid).sum(-1) / lengths.clamp_min(1)
            speed_var = ((magnitude - speed_mean[:, None]).square() * full_valid).sum(-1) / lengths.clamp_min(1)
            summary.append(torch.stack([(speed_mean - 0.75) / 0.25, speed_var.add(1e-6).sqrt() / 0.1], -1))
        pooled = self.head(torch.cat(summary, -1))
        return self.mu(pooled), self.logvar(pooled).clamp(-6, 2)


class Discriminator(nn.Module):
    """D(state, action chunk, valid-step mask), shared by both GAN branches."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = OBS_DIM + config.horizon * 3
        h = config.hidden
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(d, h)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(h, h)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(h, h // 2)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(h // 2, 1)),
        )

    def forward(self, state, actions, mask=None):
        if mask is None:
            mask = torch.ones(actions.shape[:-1], device=actions.device, dtype=actions.dtype)
        masked_actions = actions * mask[..., None]
        return self.net(torch.cat([state, masked_actions.flatten(-2), mask.to(state.dtype)], -1)).squeeze(-1)


def reparameterize(mu, logvar, noise_scale=1.0):
    return mu + noise_scale * torch.randn_like(mu) * torch.exp(0.5 * logvar)


def kl_loss(mu, logvar):
    return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()


def masked_l1(pred, target, mask):
    return ((pred - target).abs() * mask[..., None]).sum() / (mask.sum().clamp_min(1) * pred.shape[-1])


def gaussian_prior_match(mu, projections=32):
    """Label-free sliced Wasserstein matching of aggregate posterior means.

    Conditional KL alone does not ensure that distinct expert styles populate
    the Gaussian prior used at inference. Match random 1D projections as well.
    """
    direction = torch.randn(mu.shape[-1], projections, device=mu.device)
    direction = direction / direction.norm(dim=0, keepdim=True).clamp_min(1e-6)
    projected = (mu @ direction).sort(dim=0).values
    q = (torch.arange(len(mu), device=mu.device, dtype=mu.dtype) + 0.5) / len(mu)
    target = (2 ** 0.5) * torch.erfinv(2 * q - 1)
    return (projected - target[:, None]).square().mean()
