"""Read-only preservation audit, SHA256 delivery index, and portable archive."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'cube_composition_artifacts'
INDEX = ART / 'delivery_index.json'
RECEIPT = ART / 'delivery_verification.json'
ARCHIVE = ART / 'backups/cube_composition_complete.tar.gz'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def own_files():
    files = []
    for name in ['cube_composition', 'cube_composition_tests', 'cube_composition_artifacts']:
        for path in (ROOT / name).rglob('*'):
            if (path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts
                    and 'backups' not in path.relative_to(ROOT / name).parts
                    and path not in [INDEX, RECEIPT]):
                files.append(path)
    files.extend(ROOT / p for p in ['CUBE_COMPOSITION_README.md', 'CUBE_COMPOSITION_RESULTS.md'])
    return sorted(set(files))


def dependencies():
    names = json.loads((ART / 'dependency_copy.json').read_text())['files']
    files = [ROOT / p for p in names]
    for name in ['cube/cube_gan', 'cube_addons']:
        files.extend((ROOT / name).rglob('*.py'))
    files.extend([ROOT / 'cube/pyproject.toml', ROOT / 'requirements.txt'])
    files.extend(ROOT / p for p in ['CUBE_ADDONS_README.md', 'CUBE_ADDONS_RESULTS.md'])
    files.extend(ROOT / f'cube_addon_artifacts/test/{skill}_s{seed}/summary.json'
                 for skill in ['keepout', 'waypoint'] for seed in [42, 43, 44])
    return sorted(set(files))


def preservation():
    reports = []
    for name, base in [('artifacts/delivery_index.json', ROOT),
                       ('cube/artifacts/delivery_index.json', ROOT / 'cube'),
                       ('addon_artifacts/delivery_index.json', ROOT),
                       ('cube_addon_artifacts/delivery_index.json', ROOT),
                       ('composition_artifacts/delivery_index.json', ROOT)]:
        path = ROOT / name
        expected = json.loads(path.read_text())['files']
        missing, changed = [], []
        for relative, record in expected.items():
            actual = base / relative
            if not actual.is_file():
                missing.append(relative)
            elif info(actual) != record:
                changed.append(relative)
        reports.append({'index': name, 'index_sha256': sha(path), 'files': len(expected),
                        'bytes': sum(x['bytes'] for x in expected.values()),
                        'all_match': not (missing or changed), 'missing': missing, 'changed': changed})
    expected = json.loads((ART / 'dependency_copy.json').read_text())['files']
    changed = [p for p, digest in expected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    result = {'prior_deliveries': reports, 'copied_dependency_files': len(expected),
              'copied_dependencies_match': not changed, 'changed_dependencies': changed,
              'all_match': all(x['all_match'] for x in reports) and not changed,
              'scope': 'Reads prior files/indexes only. Writes this new study receipt only.'}
    (ART / 'preservation_final.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['all_match'], result
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preservation', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--archive', action='store_true')
    parser.add_argument('--verify-archive', action='store_true')
    args = parser.parse_args()
    if args.preservation:
        preservation()
        return
    if args.verify_archive:
        expected = json.loads(INDEX.read_text())['files']
        errors = []
        with tarfile.open(ARCHIVE, 'r:gz') as archive:
            members = archive.getmembers()
            actual_names = {m.name for m in members}
            if actual_names != set(expected) | {str(INDEX.relative_to(ROOT))}:
                errors.append('Archive member set differs from delivery index')
            for member in members:
                if member.name == str(INDEX.relative_to(ROOT)):
                    record = info(INDEX)
                else:
                    record = expected[member.name]
                digest = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                if member.size != record['bytes'] or digest.hexdigest() != record['sha256']:
                    errors.append(member.name)
        result = {'verified': not errors, 'members': len(members), 'errors': errors, **info(ARCHIVE)}
        (ARCHIVE.parent / 'archive_verification.json').write_text(json.dumps(result, indent=2) + '\n')
        assert result['verified'], result
        print(json.dumps(result))
        return
    if args.verify:
        expected = json.loads(INDEX.read_text())
        changed = [p for p, record in expected['files'].items()
                   if not (ROOT / p).is_file() or info(ROOT / p) != record]
        extra = sorted(set(str(p.relative_to(ROOT)) for p in own_files()) - set(expected['files']))
        result = {'verified': not (changed or extra), 'files': len(expected['files']),
                  'bytes': sum(v['bytes'] for v in expected['files'].values()),
                  'changed_or_missing': changed, 'extra_study_files': extra}
        RECEIPT.write_text(json.dumps(result, indent=2) + '\n')
        assert result['verified'], result
        print(json.dumps(result))
        return
    files = sorted(set(own_files() + dependencies()))
    payload = {'files': {str(p.relative_to(ROOT)): info(p) for p in files},
               'scope': 'Entire isolated Cube composition study plus pinned original runtime source, '
                        'all seven frozen neural checkpoints, original single-skill physical TRAIN banks, '
                        'and original context banks. Excludes bytecode, archive copies and index/verification receipts.'}
    INDEX.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'files': len(files), 'bytes': sum(v['bytes'] for v in payload['files'].values())}))
    if args.archive:
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ARCHIVE, 'w:gz', compresslevel=4) as archive:
            for path in files + [INDEX]:
                archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        (ARCHIVE.parent / 'archive_receipt.json').write_text(json.dumps({
            'archive': str(ARCHIVE.relative_to(ROOT)), **info(ARCHIVE),
            'members': len(files) + 1, 'index_sha256': sha(INDEX)}, indent=2) + '\n')
        print('Archive created', ARCHIVE)


if __name__ == '__main__':
    main()
