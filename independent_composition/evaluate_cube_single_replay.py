"""Original single-skill evaluation with complete native observation resets."""
from .observation_replay import install_single_skill_restore
install_single_skill_restore()
from cube_addons.evaluate import main


if __name__ == '__main__':
    main()
