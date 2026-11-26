"""Verify compact exports and share byte-identical, immutable RNG snapshots.

Hard links preserve every extracted filename and checksum. Original archives,
checkpoints and inputs are retained. This only manages local study backups.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import tarfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def safe(name):
    path=PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts and '\\' not in name,name
    return path


def verified_rng(root):
    for path in sorted(root.glob('completed_*/EXPORT_MANIFEST.json')):
        if not (root/(path.parent.name+'_verified.json')).is_file():continue
        manifest=json.loads(path.read_text())
        for row in manifest.get('reference_rng',[]):
            relative=safe(row['path']);assert relative.parts[0]=='reference_rng'
            p=path.parent/str(relative)
            assert p.is_file() and not p.is_symlink(),str(p)
            assert manifest['file_sha256'][str(relative)]==row['sha256']
            yield p,row['sha256']


def deduplicate(root):
    before=shutil.disk_usage(root).free;canonical={};links=0;logical=0;checked=0
    for path,expected in verified_rng(root):
        assert digest(path)==expected,('RNG copy changed',str(path));checked+=1
        if expected not in canonical:canonical[expected]=path;continue
        source=canonical[expected]
        if os.path.samefile(source,path):continue
        size=path.stat().st_size;temporary=path.with_name(path.name+'.rng-link-tmp')
        assert not temporary.exists()
        try:
            os.link(source,temporary);os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()
        assert os.path.samefile(source,path)
        links+=1;logical+=size
    result={'verified_rng_files':checked,'new_hard_links':links,
        'duplicate_logical_bytes_shared':logical,'unique_rng_contents':len(canonical),
        'free_bytes_before':before,'free_bytes_after':shutil.disk_usage(root).free,
        'scope':'Hash-identical extracted RNG files share inodes. All paths/content and original archives retained; treat extracted backups as immutable.'}
    (root/'RNG_DEDUPLICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def extract(archive,expected,root):
    archive=archive.resolve();assert archive.parent==root and archive.name.startswith('completed_') and archive.name.endswith('.tar.gz')
    assert digest(archive)==expected,'Archive checksum mismatch'
    out=archive.with_suffix('').with_suffix('');assert not out.exists(),'Use a fresh extraction destination'
    candidates={}
    for path,h in verified_rng(root):candidates.setdefault(h,path)
    out.mkdir();shared=0
    with tarfile.open(archive) as bundle:
        members=bundle.getmembers();names=[m.name for m in members]
        assert len(names)==len(set(names))
        for m in members:assert m.isfile() and safe(m.name)
        manifest=json.load(bundle.extractfile('EXPORT_MANIFEST.json'))
        assert not manifest['raw_episode_data_exported']
        assert set(names)==set(manifest['file_sha256'])|{'EXPORT_MANIFEST.json'}
        for m in members:
            dest=out/m.name;dest.parent.mkdir(parents=True,exist_ok=True)
            wanted=manifest['file_sha256'].get(m.name)
            source=candidates.get(wanted) if m.name.startswith('reference_rng/') else None
            if source is not None:
                assert digest(source)==wanted and source.stat().st_size==m.size
                os.link(source,dest);shared+=1
            else:
                with bundle.extractfile(m) as reader,dest.open('wb') as writer:shutil.copyfileobj(reader,writer,2**20)
    for name,h in manifest['file_sha256'].items():assert digest(out/name)==h,('Member checksum mismatch',name)
    result={'archive_sha256':expected,'files':len(manifest['file_sha256']),
        'models':len(manifest['models']),'raw_episode_data_exported':False,'shared_rng_files':shared}
    (root/(out.name+'_verified.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--backups',default='ablation_results/backups')
    p.add_argument('--deduplicate-existing',action='store_true');p.add_argument('--archive');p.add_argument('--expected-sha256')
    args=p.parse_args();root=Path(args.backups).resolve();assert root.is_dir()
    result={}
    if args.deduplicate_existing:result['deduplication']=deduplicate(root)
    if args.archive:
        assert args.expected_sha256 and len(args.expected_sha256)==64
        result['extraction']=extract(Path(args.archive),args.expected_sha256,root)
    assert result,'Choose --deduplicate-existing and/or --archive with its expected SHA256'
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
