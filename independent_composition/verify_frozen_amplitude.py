"""Verify frozen original-generator composition evaluation provenance."""
import json
from pathlib import Path
from .legacy import digest_file


def verify(args):
    protocol = json.loads(Path(args.frozen_protocol).read_text())
    bank = json.loads(Path(args.bank_manifest).read_text())
    assert protocol['environment'] == 'cube'
    assert bank['protocol_sha256'] == digest_file(args.frozen_protocol)
    assert bank['created'] > protocol['created']
    assert bank['selection'] == 'Geometry and independent native witness only; no learned-policy outcome filtering'
    assert len(bank['strata']) == 3
    assert args.seed == protocol['evaluation_seed']
    assert args.execute == protocol['execute']
    assert digest_file(args.config) == protocol['configuration_sha256']
    assert digest_file(args.base) == protocol['base_sha256']
    actual = [digest_file(args.keepout), digest_file(args.waypoint)]
    assert actual in list(protocol['model_pairs'].values())
    assert [args.keepout_gate, args.waypoint_gate] in protocol['gates']
    assert not args.paired
    for name, checksum in protocol['source_sha256'].items():
        assert digest_file(Path(__file__).parent / name) == checksum, ('Frozen source changed', name)
    for name, checksum in protocol['original_dependencies'].items():
        assert digest_file(name) == checksum, ('Original dependency changed', name)
    task = Path(args.tasks).resolve()
    entry = next(v for v in bank['strata'].values() if Path(v['path']).resolve() == task)
    assert args.count == entry['count'] == protocol['scene_count'] >= 64
    assert digest_file(task) == entry['sha256']
    return {'protocol_sha256': digest_file(args.frozen_protocol),
            'bank_manifest_sha256': digest_file(args.bank_manifest),
            'frozen_configuration_and_weights_verified': True,
            'fresh_scene_count': args.count, 'native_queries_in_policy_forbidden': True}
