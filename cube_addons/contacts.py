"""Simulator contact evidence for transported-cube support, without actuation."""
import numpy as np
import mujoco

SUPPORT_FORCE_N=.05


def contact_info(env,info):
    """Positive solved normal force from both Robotiq finger pad groups."""
    if not hasattr(env,'_addon_contact_groups'):
        groups={'left':set(),'right':set()}
        for i in range(env._model.ngeom):
            name=mujoco.mj_id2name(env._model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
            for side in groups:
                if f'robotiq/{side}_pad' in name:groups[side].add(i)
        if not all(groups.values()):raise RuntimeError('Cannot identify native left/right gripper pads')
        env._addon_contact_groups=(env._model.geom('object_0').id,groups)
    obj,groups=env._addon_contact_groups;force={'left':0.,'right':0.};wrench=np.zeros(6)
    for index,contact in enumerate(env._data.contact):
        if contact.geom1==obj:other=int(contact.geom2)
        elif contact.geom2==obj:other=int(contact.geom1)
        else:continue
        for side,ids in groups.items():
            if other in ids:
                mujoco.mj_contactForce(env._model,env._data,index,wrench)
                force[side]+=max(0.,float(wrench[0]))
    info['addon/left_cube_normal_force']=force['left']
    info['addon/right_cube_normal_force']=force['right']
    info['addon/bilateral_cube_contact']=bool(min(force.values())>=SUPPORT_FORCE_N)
    return info


def force_holding(info):
    if 'addon/bilateral_cube_contact' not in info:raise ValueError('Native contact evidence is required')
    distance=np.linalg.norm(info['proprio/effector_pos']-info['privileged/block_0_pos'])
    return bool(info['addon/bilateral_cube_contact'] and distance<.065)
