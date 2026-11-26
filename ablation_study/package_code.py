"""Package original scientific source and new wrappers, excluding all data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    files=[]
    for folder in ['tworoom_gan','tworoom_addons','cube/cube_gan','cube_addons','ablation_study']:
        files.extend(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and p.suffix in ['.py','.json','.txt','.tex'])
    files.extend(Path('independent_composition')/p for p in ['__init__.py','legacy.py','collect_cube_diverse.py',
        'cube_diverse_demonstrations.py','collect_cube_base_recovery.py'])
    files.extend(Path(p) for p in ['requirements.txt','pyproject.toml','cube/pyproject.toml','cube/setup.sh'])
    files=sorted(set(files));assert all(p.is_file() for p in files)
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in files:archive.write(path,str(path))
        archive.writestr('SOURCE_MANIFEST.json',json.dumps({'file_sha256':manifest,
            'scope':'Original policy, environment and data-generation source plus controlled ablation wrappers/configs. No datasets, episode records, model weights or credentials.'},indent=2)+'\n')
    print(json.dumps({'archive':str(out),'files':len(files),'bytes':out.stat().st_size,
        'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
