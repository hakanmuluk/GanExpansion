"""Restore native observations as well as integration state, outside inference.

Integration snapshots omit derived force and kinematics caches read by native
observations/control. At episode resets recompute native contact output; at
mid-episode training branches copy complete native data. Neither operation
integrates dynamics or modifies actions.
"""
import importlib
import numpy as np
import mujoco
from cube.cube_gan.env import restore as integration_restore
from cube_composition.scene import prepare as integration_prepare
from .inference_guard import inference_active


def refresh_observation(env):
    assert not inference_active(), 'Observation replay is environment bookkeeping, not planning'
    mujoco.mj_rnePostConstraint(env._model, env._data)
    env.pre_step()
    env.post_step()
    return env.compute_observation(), env.get_reset_info()


def restore_observation(env, state):
    assert not inference_active()
    integration_restore(env, state)
    return refresh_observation(env)


def restore_full_native_observation(env, data_snapshot):
    """Exact training branch: includes cached kinematics used by native control.

    Integration-only restoration calls mj_forward, which changes the cached
    end-effector pose relative to the environment's real post-step observation.
    mj_copyData preserves the actual source data, including those derived fields.
    """
    assert not inference_active(), 'Full native branching is training-only'
    mujoco.mj_copyData(env._data, env._model, data_snapshot)
    env._reset_next_step = False
    env.pre_step()
    env.post_step()
    return env.compute_observation(), env.get_reset_info()


def prepare(env, case, state=None):
    assert not inference_active()
    integration_prepare(env, case, state)
    return refresh_observation(env)


def install_single_skill_restore():
    """For training/validation workers; leave original source files unchanged."""
    assert not inference_active()
    importlib.import_module('cube_addons.rollout').restore = restore_observation
