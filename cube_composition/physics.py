"""Compact exact control-boundary forks, preserving native cached observations.

Native control reads site_xpos/site_xmat left by the last integrator substep.
Calling mj_forward alone advances these derived observations to the new qpos.
Keep the real cached values and initial observations with the integration state.
Contact observations at the boundary are retained, never recomputed from a
different solver state. Subsequent controls execute the unchanged native step.
"""
from copy import deepcopy
import numpy as np
from cube_addons.env import snapshot,restore


def capture(env,ob=None,info=None):
    return {'integration':snapshot(env),'site_xpos':env._data.site_xpos.copy(),'site_xmat':env._data.site_xmat.copy(),
        'sensordata':env._data.sensordata.copy(),'ob':deepcopy(env.compute_observation() if ob is None else ob),
        'info':deepcopy(env.get_step_info() if info is None else info),
        'python':{k:deepcopy(getattr(env,k)) for k in ['_prev_qpos','_prev_qvel','_prev_ob_info','_target_effector_pose','_success'] if hasattr(env,k)}}


def restore_branch(env,state):
    restore(env,state['integration'])
    for name in ['site_xpos','site_xmat','sensordata']:getattr(env._data,name)[:]=state[name]
    for name,value in state['python'].items():setattr(env,name,deepcopy(value))
    env._reset_next_step=False
    return deepcopy(state['ob']),deepcopy(state['info'])
