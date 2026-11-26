"""Evaluation-only tripwires; these are not a policy runtime dependency."""
from contextlib import contextmanager
from contextvars import ContextVar
import sys

_active=ContextVar('composition_policy_inference',default=False)


def inference_active():return _active.get()


def forbidden(*args,**kwargs):
    raise AssertionError('Simulator or analytic training predicate called inside policy inference')


@contextmanager
def tripwires():
    token=_active.set(True)
    targets={
        'mujoco':['mj_step','mj_forward','mj_setState','mj_rnePostConstraint','mj_copyData'],
        'cube_addons.env':['step_traced','make_env','restore','reset_case'],
        'cube.cube_gan.env':['make_env','restore','reset_context'],
        'cube_addons.geometry':['segment_distance'],
        'cube_agreement.manifolds':['oracle','segment_gap'],
        'independent_composition.train_routes':['gap','sample'],
        'tworoom_gan.env':['transition'],
        'tworoom_composition.planner':['astar','grid_astar','clear_segment','clear_segments','witness','native_step'],
        'tworoom_composition.geometry':['update','wall_clearance'],
        'tworoom_addons.geometry':['segment_distance','constraint_loss','demo_paths','update_constraint'],
        'independent_composition.collect_tworoom_routes':['label'],
        'independent_composition.train_tworoom_routes':['margins','distances'],
        'independent_composition.finetune_tworoom_routes':['worst_margin'],
        'independent_composition.train_path_encoder':['margins'],
        'independent_composition.train_path_support':['own_labels','examples'],
        'independent_composition.train_path_support_continuous':['continuous_examples','continuous_margin'],
        'independent_composition.coded_action_labels':['cube_counterfactuals'],
        'independent_composition.coded_action_physics':['tworoom_own_loss'],
        'independent_composition.refinement_labels':['cone_edit','training_pair'],
    }
    saved=[]
    for name,attributes in targets.items():
        module=sys.modules.get(name)
        if module is None:continue
        for attribute in attributes:
            if hasattr(module,attribute):
                saved.append((module,attribute,getattr(module,attribute)))
                setattr(module,attribute,forbidden)
    try:yield
    finally:
        for module,attribute,value in reversed(saved):setattr(module,attribute,value)
        _active.reset(token)


class GuardedPolicy:
    def __init__(self,policy):self.policy=policy

    def __getattr__(self,name):return getattr(self.policy,name)

    def sample(self,*args,**kwargs):
        with tripwires():return self.policy.sample(*args,**kwargs)
