"""Training-only diverse paths on a mixture retaining original task difficulty.

Half of contexts use the original independent skill-region distribution.
Half use broader placements; waypoint radius includes both smaller and larger
targets (.016-.040 m). Latent semantics are the same angular family as Stage A.
Neither expert sees another skill or joint-success labels. Original collector,
neural classes, inputs and losses remain unchanged.
"""
import json
from pathlib import Path
import sys
import numpy as np
from cube_addons.probe import constraint
from . import collect_cube_diverse as collection
from .cube_diverse_demonstrations import independent_spec
from .legacy import digest_file

_original_worker = collection.worker


def coverage_spec(case, style, skill, seed):
    if seed % 2 == 0:
        return constraint(case, style, skill, seed)
    region = independent_spec(case, style, skill, seed)
    if skill == 'waypoint':
        region[3] = np.random.default_rng(seed + 4027).uniform(.016, .040)
    return region


def worker(job):
    # Replacement affects only the training data expert's own region sampler.
    collection.independent_spec = coverage_spec
    return _original_worker(job)


def main():
    collection.worker = worker
    collection.__doc__ = __doc__
    collection.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    config = json.loads((out / 'config.json').read_text())
    config['coverage_source_sha256'] = digest_file(__file__)
    config['coverage'] = {'original_region_distribution_fraction': .5,
                          'broader_waypoint_radius_m': [.016, .040],
                          'same_stage_A_angular_code_semantics': True}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()
