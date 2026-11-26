"""Fresh benchmark generation with complete native observation restoration.

Uses the existing independent feasibility witnesses during dataset construction
only. No learned-policy outcomes select scenes. Witnesses never reach a policy.
"""
import importlib
from cube_composition import data
from . import generate_final_cube as generation
from .observation_replay import prepare

_original_initialize = data.init_worker


def initialize():
    _original_initialize()
    importlib.import_module('cube_composition.rollout').prepare = prepare


def main():
    data.init_worker = initialize
    generation.main()


if __name__ == '__main__':
    main()
