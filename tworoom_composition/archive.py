"""Lossless complete-study archive with streaming, per-member verification."""
import argparse,hashlib,json,tarfile
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def digest_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(archive,receipt):
    record=json.loads(receipt.read_text())
    with archive.open('rb') as f:assert digest_stream(f)==record['archive_sha256']
    with tarfile.open(archive,'r|*') as tar:
        members={}
        for member in tar:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            with tar.extractfile(member) as f:members[member.name]={'bytes':member.size,'sha256':digest_stream(f)}
    manifest_path=archive.parent/'complete_manifest.json'
    expected=json.loads(manifest_path.read_text())['files']
    expected['composition_artifacts/complete_manifest.json']={'bytes':manifest_path.stat().st_size,'sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    assert members==expected,{'missing':sorted(set(expected)-set(members)),'extra':sorted(set(members)-set(expected)),'changed':[p for p in expected.keys()&members.keys() if expected[p]!=members[p]]}
    return {'verified':True,'members':len(members),'uncompressed_bytes':sum(x['bytes'] for x in members.values()),'archive_bytes':archive.stat().st_size,'archive_sha256':record['archive_sha256']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args();root=ROOT/'composition_artifacts'
    archive=root/'full_study.tar.xz';receipt=root/'archive_receipt.json';manifest=root/'complete_manifest.json'
    if not args.verify:
        files=[]
        excluded={'full_study.tar.gz','full_study.tar.gz.partial','full_study.tar.xz','full_study.tar.xz.partial','complete_manifest.json','archive_receipt.json','archive_verification.json','delivery_index.json','delivery_verification.json','requirements_audit.json'}
        for directory in ['tworoom_composition','composition_tests','composition_artifacts']:
            files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in excluded)
        files.extend(ROOT/p for p in ['COMPOSITION_README.md','COMPOSITION_RESULTS.md'])
        index={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
        manifest.write_text(json.dumps({'files':index,'scope':'Complete composition-only study; original model dependencies remain in their preserved original deliveries.'},indent=2)+'\n')
        temporary=archive.with_suffix(archive.suffix+'.partial')
        with tarfile.open(temporary,'w:xz',preset=9) as tar:
            for path in sorted(files+[manifest]):tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
        temporary.replace(archive)
        with archive.open('rb') as f:h=digest_stream(f)
        receipt.write_text(json.dumps({'archive_filename':archive.name,'archive_sha256':h,'archive_bytes':archive.stat().st_size,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()},indent=2)+'\n')
    measured=verify(archive,receipt);(root/'archive_verification.json').write_text(json.dumps(measured,indent=2)+'\n');print(json.dumps(measured,indent=2))


if __name__=='__main__':main()
