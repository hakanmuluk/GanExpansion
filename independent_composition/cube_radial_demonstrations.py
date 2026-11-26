"""Training-only radial variation using an existing base latent coordinate.

The angular teacher ignores base-style amplitude at a fixed own region. This
variant gives that existing degree of freedom a useful role: clearance for
keepout, and entry radius within the waypoint. No network/interface changes,
joint instructions, joint-success labels, or inference planner are introduced.
"""
import numpy as np
from cube.cube_gan.env import TASK
from cube_addons.geometry import CARRY_Z, CUBE_BOUND_RADIUS, axis_frame
from .cube_diverse_demonstrations import DiverseSkillExpert
from .inference_guard import inference_active


class RadialSkillExpert(DiverseSkillExpert):
    def __init__(self, initial, goal, base_style, spec, new_z, skill):
        assert not inference_active(), 'Training-only expert'
        super().__init__(initial, goal, base_style, spec, new_z, skill)
        fraction = float(np.clip((base_style.amplitude - TASK.min_amplitude) /
            (TASK.max_amplitude - TASK.min_amplitude), 0., 1.))
        _, _, axis, perp, _ = axis_frame(initial, goal, initial)
        along = np.r_[axis, 0.]
        side = np.r_[perp, 0.] * (-1. if base_style.mode == 0 else 1.)
        direction = np.cos(self.angle) * side + np.array([0., 0., np.sin(self.angle)])
        start = np.r_[self.initial[:2], CARRY_Z]; end = np.r_[self.goal[:2], CARRY_Z]
        center = np.asarray(spec[:3])
        if skill == 'keepout':
            radius = spec[3] + CUBE_BOUND_RADIUS + .008 + .012 * abs(np.sin(self.angle))
            radius += .045 * fraction
            radial = radius * direction; extent = 1.25 * radius
            self.points = np.stack([start, center - extent * along + radial,
                center + radial, center + extent * along + radial, end])
        else:
            touch = center + (.15 + .70 * fraction) * spec[3] * direction
            self.points = np.stack([start, touch - .060 * along + .030 * direction,
                touch, touch + .060 * along - .010 * direction, end])

    def action(self, *args, **kwargs):
        assert not inference_active(), 'Training-only expert'
        return super().action(*args, **kwargs)
