"""Verbatim original training with complete observation replay in eval workers.

The original main and neural classes are unchanged. Module-level installation
also runs in spawned validation workers, before their original rollout starts.
"""
from .observation_replay import install_single_skill_restore
install_single_skill_restore()
from cube_addons.train import main


if __name__ == '__main__':
    main()
