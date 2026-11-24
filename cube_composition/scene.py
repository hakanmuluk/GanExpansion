"""Native scene resets; integration snapshots never replace task-cache reset."""
import numpy as np
import mujoco
from cube_addons.env import reset_case,snapshot,restore
from cube.cube_gan.env import goal_position


def reset_scene(env,seed,ood=False):
    if not ood:return reset_case(env,seed)
    rng=np.random.default_rng(seed);sign=float(rng.choice([-1,1]))
    start=np.r_[rng.uniform(.36,.48),sign*rng.uniform(.12,.22),.02]
    goal=np.r_[rng.uniform(.36,.48),-sign*rng.uniform(.12,.22),.02]
    arm=rng.uniform([.34,-.21,.25],[.51,.21,.31]);env.np_random=np.random.default_rng(seed);np.random.seed(seed)
    env.variation_space['agent']['ee_start_position'].set_init_value(arm)
    env.reset(options={'variation':[],'task_info':{'task_name':'joint_cube_transfer','init_xyzs':start[None],'goal_xyzs':goal[None]}})
    env._data.time=0.;env._data.qacc_warmstart[:]=0
    env._data.ctrl[env._arm_actuator_ids]=env._data.qpos[env._arm_joint_ids]
    env._data.ctrl[env._gripper_actuator_ids]=255*np.clip(env._data.qpos[env._gripper_opening_joint_id]/.8,0,1)
    mujoco.mj_forward(env._model,env._data);env.pre_step();env.post_step()
    return env.compute_observation(),env.get_reset_info()


def prepare(env,case,state=None):
    key=(int(case['seed']),str(case['kind']))
    if getattr(env,'_composition_case_key',None)!=key:
        reset_scene(env,int(case['seed']),str(case['kind'])=='coupled_ood');env._composition_case_key=key
    ob,info=restore(env,case['initial_state'] if state is None else state)
    assert np.allclose(goal_position(env),case['goal'],atol=1e-12)
    return ob,info
