"""Route-selectable feedback expert for real OGBench cube physics.

Inspired by OGBench's CubeMarkovOracle. It chooses a curved left/right carry
route, retains a fixed amplitude/speed per episode, and completes release and
retreat instead of stopping as soon as the cube enters the native goal radius.
This controller is for demonstration collection and training feedback only.
"""

from dataclasses import dataclass
import numpy as np
from .env import TASK, goal_position


PHASES = ["approach", "descend", "grasp", "lift", "carry", "lower", "release", "retreat"]


@dataclass(frozen=True)
class Style:
    mode: int
    amplitude: float
    speed: float


def select_action(env, info, initial_cube, style: Style):
    ee = np.asarray(info["proprio/effector_pos"])
    cube = np.asarray(info["privileged/block_0_pos"])
    goal = goal_position(env)
    yaw = float(info["proprio/effector_yaw"][0])
    cube_yaw = float(info["privileged/block_0_yaw"][0])
    closing = float(info["proprio/gripper_opening"][0])
    contact = float(info["proprio/gripper_contact"][0])
    distance = np.linalg.norm(ee - cube)
    xy_distance = np.linalg.norm(ee[:2] - cube[:2])
    goal_xy = np.linalg.norm(cube[:2] - goal[:2])
    holding = closing > 0.5 and distance < 0.055 and (contact > 0.25 or cube[2] > 0.045)
    action = np.zeros(5, dtype=np.float64)
    grip = -1.0
    yaw_error = ((cube_yaw - yaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4

    if goal_xy < 0.014 and cube[2] < 0.035:
        # Always release at the target, then move clear while the cube settles.
        target = np.r_[goal[:2], 0.20]
        phase = 6 if closing > 0.25 else 7
        yaw_error = 0.0
    elif holding:
        grip = 1.0
        yaw_error = 0.0
        offset = ee - cube
        if goal_xy < 0.025:
            target = ee + (goal - cube)
            phase = 5
        elif cube[2] < TASK.carry_height - 0.035:
            target = ee + np.array([0.0, 0.0, TASK.carry_height - cube[2]])
            phase = 3
        else:
            delta = goal[:2] - initial_cube[:2]
            length = max(np.linalg.norm(delta), 1e-6)
            direction = delta / length
            perpendicular = np.array([-direction[1], direction[0]])
            progress = np.clip(np.dot(cube[:2] - initial_cube[:2], direction) / length, 0, 1)
            lookahead = min(1.0, progress + 0.18)
            side = -1.0 if style.mode == 0 else 1.0
            xy = initial_cube[:2] + delta * lookahead + side * style.amplitude * np.sin(np.pi * lookahead) * perpendicular
            target = np.r_[xy, TASK.carry_height] + offset
            phase = 4
    elif xy_distance > 0.014:
        target = cube + np.array([0.0, 0.0, 0.15])
        phase = 0
    elif distance > 0.012:
        target = cube + np.array([0.0, 0.0, 0.002])
        phase = 1
    else:
        target = cube + np.array([0.0, 0.0, 0.002])
        grip = 1.0
        phase = 2

    action[:3] = (target - ee) * 15.0 * style.speed
    action[:3] = np.clip(action[:3], -style.speed, style.speed)
    action[3] = np.clip(yaw_error * 3.0, -1, 1)
    action[4] = grip
    return action.astype(np.float32), phase
