"""Unfiltered paired benchmark construction with corrected native observations."""
import argparse
import json
from pathlib import Path
import sys
import time
from . import create_cube_paired_tasks as collection
from . import audit_cube_data_support as support
from .observation_replay import prepare
from .legacy import digest_file


def initialize():
    support.prepare = prepare
    support.initialize()


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--supplement', required=True)
    a, remaining = p.parse_known_args()
    supplement = json.loads(Path(a.supplement).read_text())
    created = time.time(); assert supplement['created'] < created
    bank = remaining[remaining.index('--bank') + 1]
    definition = supplement['paired_definitions'][remaining[remaining.index('--vary') + 1]]
    assert digest_file(bank) == definition['bank_sha256']
    assert int(remaining[remaining.index('--offset') + 1]) == definition['offset']
    assert int(remaining[remaining.index('--count') + 1]) == definition['worlds']
    assert digest_file(__file__) == supplement['new_source_sha256'][Path(__file__).name]
    sys.argv = [sys.argv[0], *remaining]
    collection.initialize = initialize
    collection.main()
    out = Path(remaining[remaining.index('--out') + 1])
    path = out / 'manifest.json'; manifest = json.loads(path.read_text())
    manifest.update(created=created, supplement_sha256=digest_file(a.supplement),
                    wrapper_sha256=digest_file(__file__), complete_native_observation_restore=True,
                    original_hard_geometry_unchanged=True, all_constructed_cases_retained=True)
    path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
