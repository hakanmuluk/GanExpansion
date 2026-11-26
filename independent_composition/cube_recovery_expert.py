"""Training-only own-keepout recovery teacher for counterfactual code changes.

Original expert phases and code semantics remain. A short desired carry segment
that crosses the own forbidden sphere is replaced by a tangent shell step.
This code is never a policy input, inference controller or shared network.
"""
import numpy as np
from cube_addons.geometry import CUBE_BOUND_RADIUS, segment_distance
from .cube_diverse_demonstrations import DiverseSkillExpert
from .inference_guard import inference_active


class RecoverySkillExpert(DiverseSkillExpert):
    def __init__(self, *args, **kwargs):
        assert not inference_active(), 'Recovery expert is training-only'
        super().__init__(*args, **kwargs)
        assert self.skill == 'keepout'
        self.guard_interventions = 0

    def action(self, info, observer, visited_override=None):
        assert not inference_active(), 'Recovery expert is training-only'
        action, phase = super().action(info, observer, visited_override)
        if phase != 4:
            return action, phase
        cube = np.asarray(info['privileged/block_0_pos'])
        offset = cube - self.spec[:3]
        distance = max(float(np.linalg.norm(offset)), 1e-9)
        normal = offset / distance
        radius = self.spec[3] + CUBE_BOUND_RADIUS + .006
        delta = action[:3] / (15 * self.style.speed)
        if distance >= radius and segment_distance(cube, cube + delta, self.spec[:3]) >= radius:
            return action, phase
        self.guard_interventions += 1
        if distance < radius:
            target = self.spec[:3] + (radius + .002) * normal
        else:
            tangent = delta - float(delta @ normal) * normal
            if np.linalg.norm(tangent) < 1e-6:
                toward_goal = self.goal - cube
                tangent = toward_goal - float(toward_goal @ normal) * normal
            if np.linalg.norm(tangent) < 1e-6:
                basis = np.eye(3)[int(np.argmin(np.abs(normal)))]
                tangent = basis - float(basis @ normal) * normal
            tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
            shell_radius = max(distance, radius + .002)
            radial = normal + .018 / shell_radius * tangent
            radial /= np.linalg.norm(radial)
            target = self.spec[:3] + shell_radius * radial
        action[:3] = np.clip((target - cube) * 15 * self.style.speed, -self.style.speed, self.style.speed)
        return action, phase
