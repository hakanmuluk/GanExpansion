"""Matched angular-only reference for radial diversity data experiments."""
import json
from pathlib import Path
import sys
from . import collect_cube_diverse as collection
from .cube_diverse_demonstrations import DiverseSkillExpert
from .collect_cube_coverage import coverage_spec as waypoint_spec
from .collect_cube_keepout_coverage import coverage_spec as keepout_spec
from .observation_replay import install_single_skill_restore
from .legacy import digest_file

_original_initialize = collection.initialize
_original_worker = collection.worker


def initialize():
    install_single_skill_restore()
    _original_initialize()


def worker(job):
    collection.DiverseSkillExpert = DiverseSkillExpert
    collection.independent_spec = keepout_spec if job[1] == 'keepout' else waypoint_spec
    return _original_worker(job)


def main():
    collection.initialize = initialize
    collection.worker = worker
    collection.__doc__ = __doc__
    collection.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]); p = out / 'config.json'
    config = json.loads(p.read_text())
    config.update(coverage_source_sha256=digest_file(__file__),
        corrected_native_contact_observation_at_reset=True,
        old_angular_teacher_unchanged=True)
    p.write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()
