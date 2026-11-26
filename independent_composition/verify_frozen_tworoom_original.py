"""Check a fresh TwoRoom test against the frozen original-model protocol."""
import json
from pathlib import Path
from .legacy import digest_file


def verify(args):
    p = json.loads(Path(args.frozen_protocol).read_text())
    b = json.loads(Path(args.bank_manifest).read_text())
    assert args.env == p['environment'] == 'tworoom'
    assert b['protocol_sha256'] == digest_file(args.frozen_protocol)
    assert b['created'] > p['created']
    assert args.count == p['scene_count'] == 64
    assert args.execute == p['execute'] == 8 and args.seed == p['evaluation_seed']
    assert args.model_seed == 42 and args.space == 'z'
    assert not args.keepout_checkpoint and not args.waypoint_checkpoint
    assert digest_file(args.config) in p['configuration_sha256'].values()
    for path, checksum in p['models'].items():
        assert digest_file(path) == checksum, ('Frozen model changed', path)
    for path, checksum in p['source_sha256'].items():
        assert digest_file(path) == checksum, ('Frozen source changed', path)
    entry = next(v for v in b['strata'].values() if Path(v['path']).resolve() == Path(args.tasks).resolve())
    assert entry['count'] == args.count and digest_file(args.tasks) == entry['sha256']
    return {'parent_protocol_sha256': digest_file(args.frozen_protocol),
            'bank_manifest_sha256': digest_file(args.bank_manifest),
            'models_source_and_configuration_verified': True, 'fresh_scenes': 64}
