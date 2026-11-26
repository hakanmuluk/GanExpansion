"""Independent keepout data with wider constraint placements, original networks.

Half the regions follow the original distribution. Half vary progress .35-.65,
lateral offset +/- .13 m, height .17-.23 m and radius .014-.030 m. The existing
angular own-skill expert and Gaussian code labels are unchanged. Other skill
regions and joint-success labels are never observed during collection.
"""
import json
from pathlib import Path
import sys
import numpy as np
from cube_addons.probe import constraint
from cube_addons.geometry import axis_frame
from . import collect_cube_diverse as collection
from .legacy import digest_file

_original_worker = collection.worker


def coverage_spec(case, style, skill, seed):
    assert skill == 'keepout'
    if seed % 2 == 0:
        return constraint(case, style, skill, seed)
    rng = np.random.default_rng(seed + 6029)
    initial, goal = case['initial_cube'], case['goal']
    _, _, _, perp, _ = axis_frame(initial, goal, initial)
    progress = rng.uniform(.35, .65)
    sign = 1. if style.mode else -1.
    xy = initial[:2] + (goal[:2] - initial[:2]) * progress
    xy += (sign * style.amplitude * np.sin(np.pi * progress) + rng.uniform(-.13, .13)) * perp
    return np.r_[xy, rng.uniform(.17, .23), rng.uniform(.014, .030)]


def worker(job):
    collection.independent_spec = coverage_spec
    return _original_worker(job)


def main():
    collection.worker = worker
    collection.__doc__ = __doc__
    collection.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    f = out / 'config.json'
    config = json.loads(f.read_text())
    config['coverage_source_sha256'] = digest_file(__file__)
    config['original_region_distribution_fraction'] = .5
    config['same_stage_A_angular_code_semantics'] = True
    f.write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()
