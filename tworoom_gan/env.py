"""TwoRoom task configuration and a batched replica of its actual dynamics.

Reference: stable-worldmodel commit 21446f1ede6d5284e981bd7b47f432b994e6d812,
envs/two_room/env.py (MIT). The PyPI 0.1.1 wheel predates this implementation.
The GPU transition below intentionally preserves the reference collision
semantics, including its door-margin convention. See verify_reference().
Coordinates are pixels internally; policy observations are normalized to [-1,1].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Task:
    image_size: int = 224
    wall_x: float = 112.0
    wall_thickness: int = 10
    radius: float = 7.0
    border: int = 14
    speed: float = 5.0
    door_y: tuple[float, float] = (56.0, 168.0)
    door_half_width: float = 21.0
    success_radius: float = 16.0
    max_steps: int = 160

    def to_dict(self):
        return asdict(self)


TASK = Task()
OBS_DIM = 10
HORIZON = 16
EXECUTE = 8


def observation(pos: Tensor, goal: Tensor) -> Tensor:
    doors = pos.new_tensor([112.0, 56.0, 112.0, 168.0, 0.0, 0.0])
    return torch.cat([pos, goal, doors.expand(*pos.shape[:-1], 6)], -1) / 112.0 - 1.0


def transition(pos: Tensor, action: Tensor, task: Task = TASK) -> tuple[Tensor, Tensor]:
    """Vectorized reference step; supports arbitrary leading dimensions."""
    proposed = pos + action.clamp(-1, 1) * task.speed
    bounded = proposed.clamp(task.border + task.radius, task.image_size - task.border - task.radius)
    door_y = pos.new_tensor(task.door_y)
    in_door = ((bounded[..., 1, None] - door_y).abs() <= task.door_half_width + 1.75).any(-1)
    left = pos[..., 0] < task.wall_x
    left_edge = task.wall_x - task.wall_thickness // 2 - task.radius
    right_edge = task.wall_x + task.wall_thickness // 2 + task.radius
    blocked = (~in_door) & torch.where(left, bounded[..., 0] > left_edge, bounded[..., 0] < right_edge)
    clamp_x = torch.where(left, pos.new_tensor(left_edge - 0.5), pos.new_tensor(right_edge + 0.5))
    x = torch.where(blocked, clamp_x, bounded[..., 0])
    return torch.stack([x, bounded[..., 1]], -1), blocked


def expert_action(pos: Tensor, goal: Tensor, mode: Tensor, speed: Tensor, offset: Tensor | None = None) -> Tensor:
    """A route-selectable feedback expert with two genuinely distinct paths.

    Route stays fixed for the whole episode. The approach/crossing/exit
    waypoints are inferred from state, without an inaccessible phase variable.
    Speed is a per-episode action magnitude, not per-step noise.
    """
    y = pos.new_tensor(TASK.door_y)[mode.long()]
    if offset is not None:
        y = y + offset
    target_side = torch.where(goal[..., 0] > TASK.wall_x, 1.0, -1.0)
    same_room = (pos[..., 0] > TASK.wall_x) == (goal[..., 0] > TASK.wall_x)
    aligned = (pos[..., 1] - y).abs() < 3.0
    approach = torch.stack([TASK.wall_x - target_side * 27.0, y], -1)
    exit_point = torch.stack([TASK.wall_x + target_side * 29.0, y], -1)
    opposite_waypoint = torch.where(aligned[..., None], exit_point, approach)
    # Clearance must be smaller than the nearest permitted goal's wall distance
    # (24px), otherwise a goal near the wall causes approach/exit oscillation.
    cleared = (pos[..., 0] - TASK.wall_x).abs() >= 20.0
    same_waypoint = torch.where(cleared[..., None], goal, exit_point)
    waypoint = torch.where(same_room[..., None], same_waypoint, opposite_waypoint)
    delta = waypoint - pos
    distance = delta.norm(dim=-1, keepdim=True)
    action = delta / distance.clamp_min(1e-6) * speed[..., None]
    action = action * (distance / (TASK.speed * speed[..., None]).clamp_min(1e-6)).clamp(max=1)
    return torch.where(((goal - pos).norm(dim=-1) < TASK.success_radius)[..., None], 0.0, action)


def expert_chunk(pos: Tensor, goal: Tensor, mode: Tensor, speed: Tensor, horizon: int = HORIZON, offset: Tensor | None = None) -> Tensor:
    actions = []
    for _ in range(horizon):
        a = expert_action(pos, goal, mode, speed, offset=offset)
        actions.append(a)
        pos, _ = transition(pos, a)
    return torch.stack(actions, -2)


def make_reference():
    """Original SWM environment, with controlled geometry and goal sampling."""
    from stable_worldmodel.envs.two_room import TwoRoomEnv
    env = TwoRoomEnv(render_target=True)
    # Upstream Dict.set_init_value requires a COMPLETE nested dictionary.
    # Configure individual leaves to preserve all other documented defaults.
    leaves = {("door", "number"): 2,
              ("door", "position"): np.array([56, 168, 0]),
              ("door", "size"): np.array([21, 21, 21]),
              ("agent", "radius"): np.array([7.0], np.float32),
              ("agent", "speed"): np.array([5.0], np.float32),
              ("wall", "axis"): 1, ("wall", "thickness"): 10}
    for (group, name), value in leaves.items():
        env.variation_space[group][name].set_init_value(value)
    return env


def reset_reference(env, start, goal):
    return env.reset(seed=0, options={"state": np.asarray(start, np.float32), "target_state": np.asarray(goal, np.float32), "variation": []})


def verify_reference(samples: int = 3000, seed: int = 17) -> dict:
    """Compare arbitrary transitions AND complete expert rollouts to SWM."""
    env = make_reference()
    reset_reference(env, [56, 112], [168, 112])
    rng = np.random.default_rng(seed)
    pos = rng.uniform(21, 203, (samples, 2)).astype(np.float32)
    action = rng.uniform(-1.5, 1.5, (samples, 2)).astype(np.float32)
    expected, _ = transition(torch.from_numpy(pos), torch.from_numpy(action))
    observed = []
    for p, a in zip(pos, action):
        env._set_state(p)
        s, _, _, _, _ = env.step(a)
        observed.append(np.asarray(s[:2]))
    max_error = float(np.abs(np.stack(observed) - expected.numpy()).max())
    assert max_error < 3e-5, f"Reference transition mismatch: {max_error}"
    outcomes = []
    for direction in [0, 1]:
        start, goal = ([40, 112], [184, 112]) if direction == 0 else ([184, 112], [40, 112])
        for mode in [0, 1]:
            reset_reference(env, start, goal)
            p = torch.tensor([start], dtype=torch.float32)
            g = torch.tensor([goal], dtype=torch.float32)
            for step in range(TASK.max_steps):
                a = expert_action(p, g, torch.tensor([mode]), torch.tensor([0.7]))
                p, _ = transition(p, a)
                s, _, done, _, _ = env.step(a[0].numpy())
                assert np.allclose(np.asarray(s[:2]), p[0].numpy(), atol=3e-5)
                if done:
                    break
            assert done, (direction, mode)
            outcomes.append({"direction": direction, "mode": mode, "steps": step + 1})
    env.close()
    return {"random_transitions": samples, "max_position_error_pixels": max_error, "expert_rollouts": outcomes}
