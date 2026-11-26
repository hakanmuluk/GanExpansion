"""Archive exact source/weights for reproduction, explicitly omitting case arrays."""
import argparse,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
from .analyze import read,sha
from .evaluate import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    study_path=Path(a.study);study=read(study_path);verify(study)
    old=Path(study['original_root']);new=Path(__file__).resolve().parent.parent;out=Path(a.out)
    assert old.parent==new.parent==Path('/content')
    assert not out.exists();out.mkdir(parents=True)
    protected=dict(study['original_hashes'])
    for e in study['environments'].values():
        for name,digest in read(e['old_freeze'])['original_sources'].items():
            path=Path(name) if Path(name).is_absolute() else old/name
            assert sha(path)==digest;protected[str(path)]=digest
    source=set();weights=set();omitted=[]
    for name,digest in protected.items():
        path=Path(name);assert sha(path)==digest
        if path.suffix=='.npz':omitted.append({'path':name,'sha256':digest,'reason':'Protected native task bank, not exported.'})
        elif path.suffix=='.pt':weights.add(path)
        else:
            assert path.suffix in ['.py','.json','.txt'];source.add(path)
    for folder in old.iterdir():
        if not folder.is_dir() or folder.name.startswith('.') or folder.name.endswith('_artifacts'):continue
        if (folder/'__init__.py').is_file():
            source.update(p for p in folder.rglob('*.py') if '__pycache__' not in p.parts)
    source.update((old/'cube/cube_gan').rglob('*.py'))
    for name in ['requirements.txt','pyproject.toml','README.md','cube/setup.sh','cube/pyproject.toml','cube/README.md']:
        if (old/name).is_file():source.add(old/name)
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ['.py','.txt','.json','.tex']:source.add(p)
    source.update((study_path.parent/'configs').rglob('*.json'))
    source.update([study_path,study_path.parent/'preflight.json'])
    assert len(weights)==10 and len(omitted)==6
    def name(path):
        path=path.resolve();assert path.is_relative_to('/content')
        return str(path.relative_to('/content'))
    def bundle(path,files):
        manifest={name(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(files):z.write(p,name(p))
            z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        return {'bytes':path.stat().st_size,'sha256':sha(path),'members':len(files)}
    # Safe aggregate expectations let a fresh host regenerate the original
    # reference before running the instrumented sweep, without exporting traces.
    expected={}
    for env,e in study['environments'].items():
        expected[env]={}
        for kind in ['supported_easy','supported_conflict','coupled_ood']:
            r=read(Path(e['old_evaluations'])/kind/'summary.json')
            expected[env][kind]={k:r[k] for k in ['episodes','success','goal_success','keepout_satisfied','waypoint_satisfied']}
    expectation=study_path.parent/'reproduction_reference_expectations.json'
    content=json.dumps(expected,indent=2)+'\n'
    if expectation.exists():assert expectation.read_text()==content
    else:expectation.write_text(content)
    source.add(expectation)
    record={'source':bundle(out/'reference_code_and_protocol.zip',source),
        'weights':bundle(out/'unchanged_reference_weights.zip',weights),'omitted_protected_task_banks':omitted,
        'study_sha256':sha(study_path),'original_root':str(old),'study_code_root':str(new),
        'raw_episode_data_exported':False,
        'scope':'Exact original source and10unchanged model/encoder checkpoints, plus new controlled wrappers and frozen configs. Canonical paths restored by extraction under/content. Six raw case banks, native rollouts and episode-level records excluded.'}
    (out/'REFERENCE_INPUT_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'source':record['source'],'weights':record['weights'],'raw_banks_omitted':6,'out':str(out)}))

if __name__=='__main__':main()
