"""Real MuJoCo OGBenchCube environment, reset snapshots, and policy features.

Reference: stable-worldmodel commit 21446f1ede6d5284e981bd7b47f432b994e6d812.
No approximate simulator is used for data collection or evaluation.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass, asdict
import numpy as np
import mujoco
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv


@dataclass(frozen=True)
class Task:
    max_steps: int = 200
    horizon: int = 16
    execute: int = 8
    action_dim: int = 5
    observation_dim: int = 47
    goal_tolerance: float = 0.04
    stable_steps: int = 10
    carry_height: float = 0.20
    min_amplitude: float = 0.035
    max_amplitude: float = 0.075
    min_speed: float = 0.65
    median_speed: float = 0.85
    max_speed: float = 1.0

    def to_dict(self):
        return asdict(self)


TASK = Task()
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def make_env(render_size=224):
    return CubeEnv(env_type="single", ob_type="states", permute_blocks=False,
                   terminate_at_goal=False, success_timing="post", mode="task",
                   reward_task_id=None, height=render_size, width=render_size)


def reset_context(env, seed: int, task_id: int = 0):
    """Native tasks 1..5, or a randomized goal-conditioned task (ID 0).

    Upstream SWM reset does not forward its explicit seed to the parent. Seed
    the actual generator explicitly; full integration snapshots then guarantee
    exactly identical initial physics for every style in a paired context.
    """
    rng = np.random.default_rng(seed)
    env.np_random = np.random.default_rng(seed)
    np.random.seed(seed)
    options = {"variation": []}
    if task_id:
        options["task_id"] = int(task_id)
        arm = np.array([0.425, 0.0, 0.275])
    else:
        start = np.r_[rng.uniform([0.34, -0.20], [0.51, 0.20]), 0.02]
        for _ in range(1000):
            goal = np.r_[rng.uniform([0.34, -0.20], [0.51, 0.20]), 0.02]
            if np.linalg.norm(start[:2] - goal[:2]) >= 0.15:
                break
        else:
            raise RuntimeError("Unable to sample a nontrivial cube transfer")
        options["task_info"] = {"task_name": "random_transfer", "init_xyzs": start[None], "goal_xyzs": goal[None]}
        arm = rng.uniform([0.32, -0.22, 0.23], [0.53, 0.22, 0.32])
    env.variation_space["agent"]["ee_start_position"].set_init_value(arm.astype(np.float64))
    ob, info = env.reset(options=options)
    # Upstream creates its goal observation using two random temporary actions,
    # then restores qpos/qvel but not ctrl/warm-start. Remove those irrelevant
    # remnants so the saved initial integration state is reproducible.
    env._data.time = 0.0
    env._data.qacc_warmstart[:] = 0
    env._data.ctrl[env._arm_actuator_ids] = env._data.qpos[env._arm_joint_ids]
    env._data.ctrl[env._gripper_actuator_ids] = 255.0 * np.clip(env._data.qpos[env._gripper_opening_joint_id] / 0.8, 0, 1)
    mujoco.mj_forward(env._model, env._data)
    env.pre_step()
    env.post_step()
    return env.compute_observation(), env.get_reset_info()


def snapshot(env):
    state = np.empty(mujoco.mj_stateSize(env._model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(env._model, env._data, state, STATE_SPEC)
    return state


def restore(env, state):
    mujoco.mj_setState(env._model, env._data, np.asarray(state, dtype=np.float64), STATE_SPEC)
    mujoco.mj_forward(env._model, env._data)
    env._reset_next_step = False
    env._prev_qpos = env._data.qpos.copy()
    env._prev_qvel = env._data.qvel.copy()
    env.pre_step()
    env.post_step()
    return env.compute_observation(), env.get_reset_info()


def goal_position(env):
    return env._data.mocap_pos[env._cube_target_mocap_ids[0]].copy()


def features(env, ob, info, initial_cube, previous_ee=None):
    """47 observed features: native state, task context, and relative geometry.

    No expert phase, latent target, or planned waypoint is an input. Initial
    cube position is explicit episode/task conditioning, retained across replans.
    """
    ee = info["proprio/effector_pos"]
    cube = info["privileged/block_0_pos"]
    goal = goal_position(env)
    center = np.array([0.425, 0.0, 0.0])
    cube_velocity = env._data.joint("object_joint_0").qvel[:3]
    ee_velocity = np.zeros(3) if previous_ee is None else (ee - previous_ee) / 0.05
    result = np.concatenate([
        np.asarray(ob), (goal - center) * 10, (initial_cube - center) * 10,
        (cube - ee) * 10, (goal - cube) * 10, cube_velocity * 5,
        np.asarray(info["proprio/gripper_vel"]) * 0.2, ee_velocity * 5,
    ]).astype(np.float32)
    assert result.shape == (TASK.observation_dim,), result.shape
    return result


def completion(env, info):
    """Native goal success plus independently measured released placement."""
    cube = info["privileged/block_0_pos"]
    ee = info["proprio/effector_pos"]
    native = bool(np.linalg.norm(cube - goal_position(env)) <= TASK.goal_tolerance)
    velocity = env._data.joint("object_joint_0").qvel[:3]
    released = float(info["proprio/gripper_opening"][0]) < 0.25
    contact_free = float(info["proprio/gripper_contact"][0]) < 0.15
    clear = ee[2] - cube[2] > 0.08
    settled = cube[2] < 0.035 and np.linalg.norm(velocity) < 0.04
    return native, bool(native and released and contact_free and clear and settled)


def close_env(env):
    # OGBench's generic close can leave the EGL renderer alive until interpreter
    # teardown; explicitly close it to avoid shutdown-time EGL exceptions.
    for name in ["_renderer", "renderer"]:
        renderer = getattr(env, name, None)
        if renderer is not None and hasattr(renderer, "close"):
            renderer.close()
    env.close()
