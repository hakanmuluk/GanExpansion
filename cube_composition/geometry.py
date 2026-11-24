"""Joint monitoring reuses the independently audited single-skill predicates."""
import numpy as np
import mujoco
from cube_addons.geometry import Observer,axis_frame,segment_distance
from cube_addons.contacts import contact_info,force_holding


class JointObserver:
    def __init__(self,case):
        self.k=Observer(case['initial_cube'],case['goal'],case['keepout'],'keepout',force_holding)
        self.w=Observer(case['initial_cube'],case['goal'],case['waypoint'],'waypoint',force_holding)
        self.case=case;self.grasped=False;self.drop=False;self.grasp_losses=0;self.robot_table=False
        self.ee_violation=False;self.minimum_ee_clearance=float('inf');self.previous_held=False;self.previous_ee=None

    def features(self,info,time):return self.k.features(info,time),self.w.features(info,time)

    def update(self,before,after,info,path,env=None):
        held=force_holding(info);self.grasped|=held
        goal_distance=np.linalg.norm(after[:2]-self.case['goal'][:2]);separation=np.linalg.norm(info['proprio/effector_pos']-after)
        if self.previous_held and not held and goal_distance>.06 and self.k.has_lifted:self.grasp_losses+=1
        if self.k.has_lifted and goal_distance>.06 and (separation>.09 or after[2]<.065):self.drop=True
        self.previous_held=held;self.k.update(before,after,info,path);self.w.update(before,after,info,path)
        ee=np.asarray(info['proprio/effector_pos']);a=ee if self.previous_ee is None else self.previous_ee
        clearance=float(segment_distance(a,ee,self.case['keepout'][:3])-self.case['keepout'][3]);self.minimum_ee_clearance=min(self.minimum_ee_clearance,clearance)
        self.ee_violation|=clearance<0;self.previous_ee=ee.copy()
        p=axis_frame(self.case['initial_cube'],self.case['goal'],after)[0]
        if env is not None and self.k.has_lifted and .2<p<.8:
            for index,contact in enumerate(env._data.contact):
                names=[(env._model.geom(int(i)).name or '')+' '+(env._model.body(int(env._model.geom_bodyid[int(i)])).name or '') for i in [contact.geom1,contact.geom2]]
                if any(any(t in name.lower() for t in ['table','floor','ground']) for name in names) and any(any(t in name.lower() for t in ['ur5','robotiq']) for name in names):
                    wrench=np.zeros(6);mujoco.mj_contactForce(env._model,env._data,index,wrench)
                    self.robot_table|=wrench[0]>.1

    def result(self,released,native):
        constraints=self.k.satisfied() and self.w.satisfied()
        return {'success':bool(released and constraints and not self.drop and not self.robot_table),
            'joint_without_drop_collision_extension':bool(released and constraints),'released_success':bool(released),'native_success':bool(native),
            'keepout_satisfied':self.k.satisfied(),'waypoint_satisfied':self.w.satisfied(),'visited':self.w.visited,
            'keepout_violation':self.k.violated,'table_violation':self.k.table_violation or self.w.table_violation,
            'min_cube_clearance':self.k.minimum_clearance,'ee_region_violation_diagnostic':bool(self.ee_violation),'min_ee_clearance_diagnostic':self.minimum_ee_clearance,
            'grasped':bool(self.grasped),'has_lifted':self.k.has_lifted,'dropped':bool(self.drop),'grasp_losses':self.grasp_losses,
            'robot_table_contact_in_transport':bool(self.robot_table),'supported_transport_path_fraction':self.k.supported_fraction()}
