"""Local SHA256 index for the new delivery; previous deliveries are read-only."""
import argparse
import datetime
import hashlib
import json
import fnmatch
from pathlib import Path


INDEX=Path('cube_addon_artifacts/delivery_index.json')
RECEIPT=Path('cube_addon_artifacts/delivery_verification.json')
SOURCE_INDEX=Path('cube_addon_artifacts/source_index.json')
SOURCE_RECEIPT=Path('cube_addon_artifacts/source_verification.json')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def paths():
    result=[]
    for root in ['cube_addons','cube_addon_tests','cube_addon_artifacts']:
        result.extend(p for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                      and p not in [INDEX,RECEIPT,SOURCE_RECEIPT] and not p.name.startswith('.') and p.suffix!='.tmp')
    result.extend(Path(p) for p in ['CUBE_ADDONS_README.md','CUBE_ADDONS_RESULTS.md'])
    return sorted(set(result))


def source_paths():
    """Required reproducible bundle; raw simulator/debug duplicates stay remote."""
    selected=[]
    for path in paths():
        if path in [SOURCE_INDEX,SOURCE_RECEIPT] or path.name=='source_missing_files.txt':continue
        if path.parts[0]!='cube_addon_artifacts':selected.append(path);continue
        rel=path.relative_to('cube_addon_artifacts');parts=rel.parts
        if parts[0]=='data' or any(p.endswith('_raw') for p in parts):continue
        if path.suffix=='.npz':
            canonical=parts[0] in ['data_force','macro_data','pilot_data','pilot_force'] and len(parts)<=3
            compact=path.name in ['episodes.npz','feedback.npz','original_test_contexts.npz']
            example=fnmatch.fnmatch(path.name,'rollout_000[0-3]_*.npz')
            probe=parts[0] in ['probe','probe_v2_targeted']
            if canonical or compact or example or probe:selected.append(path)
        elif path.suffix=='.pt':
            if path.name in ['best.pt','latest.pt','adapter.pt']:selected.append(path)
        elif path.suffix in ['.py','.sh','.json','.jsonl','.png','.gif','.xml','.log','.txt','.md']:selected.append(path)
    # Also retain every code used in each rendered same-context comparison.
    for selection in Path('cube_addon_artifacts/visuals').glob('*/selection.json'):
        skill=selection.parent.name;context=json.loads(selection.read_text())['context']
        selected.extend(Path('cube_addon_artifacts/test',skill+'_s42').glob(f'rollout_{context:04d}_*.npz'))
    return sorted(set(selected))


def source_index():
    files={str(path):{'bytes':path.stat().st_size,'sha256':sha(path)} for path in source_paths()}
    SOURCE_INDEX.write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Canonical demonstrations, correction banks, exact calibration inputs, all checkpoints/resume states and per-attempt metrics, figures/videos, and representative full traces. Full raw debug traces remain remote and are reproducible.',
        'files':files},indent=2)+'\n')
    print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'index':str(SOURCE_INDEX)}))


def verify_source(index):
    data=json.loads(Path(index).read_text());failures=[]
    for name,info in data['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid source index path')
        if not path.is_file() or path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:failures.append(name)
    Path('cube_addon_artifacts/source_missing_files.txt').write_text(''.join(name+'\n' for name in failures))
    result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(data['files']),
            'bytes':sum(x['bytes'] for x in data['files'].values()),'all_match':not failures,'mismatched_or_missing':failures}
    SOURCE_RECEIPT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_match':not failures,'files':len(data['files']),'mismatched_or_missing_count':len(failures),
                      'first_failures':failures[:20]},indent=2))
    if failures:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true')
    p.add_argument('--source-index',action='store_true');p.add_argument('--verify-source');args=p.parse_args()
    if args.source_index:source_index();return
    if args.verify_source:verify_source(args.verify_source);return
    if args.verify:
        data=json.loads(INDEX.read_text());failures=[]
        for name,info in data['files'].items():
            path=Path(name)
            if not path.is_file() or path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:failures.append(name)
        result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(data['files']),
                'bytes':sum(x['bytes'] for x in data['files'].values()),'all_match':not failures,'failures':failures}
        RECEIPT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        if failures:raise SystemExit(1)
    else:
        files={str(path):{'bytes':path.stat().st_size,'sha256':sha(path)} for path in paths()}
        INDEX.write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'scope':'New independent Cube add-on source and artifacts only. Prior delivery indexes remain separate.',
                                    'files':files},indent=2)+'\n')
        print(f'Indexed {len(files)} files, {sum(x["bytes"] for x in files.values())} bytes')


if __name__=='__main__':main()
