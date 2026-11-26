"""Original development distribution with complete native reset observations."""
import importlib
from cube_composition import data
from .observation_replay import prepare

_original = data.init_worker


def initialize():
    _original()
    importlib.import_module('cube_composition.rollout').prepare = prepare


def main():
    data.init_worker = initialize
    data.main()


if __name__ == '__main__':main()
