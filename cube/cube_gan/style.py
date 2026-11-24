"""Explicit auxiliary Gaussian style coordinates (training labels only)."""
import torch
from .env import TASK


def encode_style(mode, amplitude, speed):
    p = ((amplitude - TASK.min_amplitude) / (TASK.max_amplitude - TASK.min_amplitude)).clamp(1e-4, 1-1e-4)
    half = 2**0.5 * torch.erfinv(p)
    q = torch.where(speed < TASK.median_speed,
                    0.5*(speed-TASK.min_speed)/(TASK.median_speed-TASK.min_speed),
                    0.5+0.5*(speed-TASK.median_speed)/(TASK.max_speed-TASK.median_speed)).clamp(1e-4, 1-1e-4)
    return torch.stack([half * (mode*2-1), 2**0.5*torch.erfinv(2*q-1)], -1)


def decode_style(z):
    p = torch.erf(z[..., 0].abs()/2**0.5)
    q = 0.5*(1+torch.erf(z[..., 1]/2**0.5))
    speed = torch.where(q < 0.5, TASK.min_speed+2*q*(TASK.median_speed-TASK.min_speed),
                        TASK.median_speed+2*(q-0.5)*(TASK.max_speed-TASK.median_speed))
    return (z[..., 0]>=0).long(), TASK.min_amplitude+p*(TASK.max_amplitude-TASK.min_amplitude), speed
