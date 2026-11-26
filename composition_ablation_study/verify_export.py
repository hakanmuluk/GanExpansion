"""Stdlib-only verification for aggregate-only backup artifacts."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
    source=Path(a.archive);assert digest(source)==a.sha256;out=source.with_suffix('').with_suffix('');assert not out.exists()
    with tarfile.open(source) as t:
        members=t.getmembers();names=[m.name for m in members];assert len(set(names))==len(names)
        for m in members:
            path=PurePosixPath(m.name)
            assert m.isfile() and not path.is_absolute() and '..' not in path.parts
            assert path.suffix in ['.py','.txt','.json','.csv','.pdf','.png','.tex']
            assert path.name not in ['records.json','planning.json'] and not path.name.startswith('rollout_')
        record=json.loads(t.extractfile('EXPORT_MANIFEST.json').read())
        assert not record['raw_episode_data_exported'] and set(names)==set(record['file_sha256'])|{'EXPORT_MANIFEST.json'}
        out.mkdir()
        for m in members:
            target=out/m.name;target.parent.mkdir(parents=True,exist_ok=True);content=t.extractfile(m).read()
            if m.name!='EXPORT_MANIFEST.json':assert hashlib.sha256(content).hexdigest()==record['file_sha256'][m.name],m.name
            target.write_bytes(content)
    proof={'archive_sha256':a.sha256,'files_verified':len(record['file_sha256']),'completed_cells':record['completed_cells'],
           'partial':record['partial'],'raw_episode_data_exported':False}
    out.with_name(out.name+'_verified.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))

if __name__=='__main__':main()
