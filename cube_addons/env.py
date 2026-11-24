"""The original MuJoCo dynamics with new randomized single-cube contexts."""
import numpy as np
import mujoco
from cube.cube_gan.env import make_env,snapshot,restore,features,completion,goal_position,close_env,reset_context


def step_traced(env, action):
    """Run the native controller, recording cube positions every physics step.

    Splitting MuJoCo's nstep loop performs the same integration operations.
    There is no action projection, contact injection, or constraint enforcement.
    This function is used serially inside each independent worker process.
    """
    original = mujoco.mj_step
    path = [env._data.joint("object_joint_0").qpos[:3].copy()]
    def traced(model, data, nstep=1):
        if data is not env._data:
            return original(model, data, nstep=nstep)
        for _ in range(nstep):
            original(model, data, nstep=1)
            path.append(data.joint("object_joint_0").qpos[:3].copy())
    mujoco.mj_step = traced
    try:
        result = env.step(action)
    finally:
        mujoco.mj_step = original
    return result, np.asarray(path)


def reset_case(env,seed):
    rng=np.random.default_rng(seed);sign=float(rng.choice([-1,1]))
    start=np.array([rng.uniform(.39,.45),sign*rng.uniform(.14,.20),.02])
    goal=np.array([rng.uniform(.39,.45),-sign*rng.uniform(.14,.20),.02])
    env.np_random=np.random.default_rng(seed);np.random.seed(seed)
    arm=rng.uniform([.35,-.20,.25],[.50,.20,.31])
    env.variation_space["agent"]["ee_start_position"].set_init_value(arm)
    env.reset(options={"variation":[],"task_info":{"task_name":"addon_transfer","init_xyzs":start[None],"goal_xyzs":goal[None]}})
    env._data.time=0.;env._data.qacc_warmstart[:]=0
    env._data.ctrl[env._arm_actuator_ids]=env._data.qpos[env._arm_joint_ids]
    env._data.ctrl[env._gripper_actuator_ids]=255*np.clip(env._data.qpos[env._gripper_opening_joint_id]/.8,0,1)
    mujoco.mj_forward(env._model,env._data);env.pre_step();env.post_step()
    return env.compute_observation(),env.get_reset_info()
