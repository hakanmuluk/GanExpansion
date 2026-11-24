"""Optional data-informed Gaussian coordinates for known expert styles.

This is auxiliary supervision, not part of vanilla BicycleGAN. A balanced
route bit plus uniform crossing offset is transformed into a standard normal
coordinate; the exact speed sampling CDF gives the other normal coordinate.
These functions are used in training/diagnostics only, never policy inference.
"""

import math
import torch


def encode_style(mode, speed, offset):
    u_offset = ((offset + 12) / 24).clamp(1e-4, 1 - 1e-4)
    magnitude = math.sqrt(2) * torch.erfinv(u_offset)
    route_sign = mode.to(speed.dtype) * 2 - 1
    u_speed = torch.where(speed < 0.78, (speed - 0.55) / 0.46, 0.5 + (speed - 0.78) / 0.44).clamp(1e-4, 1 - 1e-4)
    z_speed = math.sqrt(2) * torch.erfinv(2 * u_speed - 1)
    return torch.stack([route_sign * magnitude, z_speed], -1)


def decode_style(z):
    mode = (z[..., 0] >= 0).long()
    u_offset = torch.erf(z[..., 0].abs() / math.sqrt(2))
    offset = 24 * u_offset - 12
    u_speed = 0.5 * (1 + torch.erf(z[..., 1] / math.sqrt(2)))
    speed = torch.where(u_speed < 0.5, 0.55 + 0.46 * u_speed, 0.78 + 0.44 * (u_speed - 0.5))
    return mode, speed, offset


def canonical_state(state):
    """Reflect left-going tasks into the equivalent right-going task.

    Only agent/goal x coordinates change: active doors are centered at x=0
    after normalization and inactive door coordinates remain padding constants.
    Goal side stays fixed throughout an episode, including after wall crossing.
    """
    sign = torch.where(state[..., 2] >= 0, 1.0, -1.0)
    canonical = torch.cat([state[..., 0:1] * sign[..., None], state[..., 1:2],
                           state[..., 2:3] * sign[..., None], state[..., 3:]], -1)
    return canonical, sign
