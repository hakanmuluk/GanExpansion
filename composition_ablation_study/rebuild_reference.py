"""Fresh-host only: reconstruct original reference outputs, with no overwrites."""
import argparse,json,subprocess,sys
from pathlib import Path
from .analyze import read,sha
from .evaluate import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    study_path=Path(a.study);study=read(study_path);verify(study)
    expectations=read(study_path.parent/'reproduction_reference_expectations.json');commands=[]
    for env,e in study['environments'].items():
        freeze=read(e['old_freeze']);protocol=Path(e['old_freeze']).parent
        for kind,tasks in e['tasks'].items():
            out=Path(e['old_evaluations'])/kind
            argv=[sys.executable,'-u','-m','independent_composition.evaluate_regularized_latents','--env',env,
                '--base',freeze['role_model_paths']['base'],'--keepout',freeze['role_model_paths']['keepout'],
                '--waypoint',freeze['role_model_paths']['waypoint'],'--tasks',tasks,'--config',str(protocol/'prior.json'),
                '--out',str(out),'--count','64','--execute','8','--seed','241','--freeze',e['old_freeze'],
                '--bank-manifest',str(Path(tasks).parent/'manifest.json'),'--method-name','prior']
            commands.append((env,kind,out,argv))
    if a.run:
        assert all(not out.exists() for _,_,out,_ in commands),'Original reference outputs already exist; preserve them.'
        for env,kind,out,argv in commands:
            subprocess.run(argv,cwd=study['original_root'],check=True)
            actual=read(out/'summary.json')
            assert all(actual[k]==v for k,v in expectations[env][kind].items()),(env,kind,'Reference aggregate mismatch; do not start new comparisons.')
    print(json.dumps({'commands':6,'executed':a.run,'original_files_verified':True,
        'scope':'Uses original CLI, original frozen models/config and exact bank hashes. Existing outputs never overwritten.'}))

if __name__=='__main__':main()
