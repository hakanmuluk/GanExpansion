"""Create fresh output paths, preserving every numeric scientific setting."""
import argparse,copy,json,shutil
from pathlib import Path
from .analyze import read,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    original=read(a.study);target=Path(a.out).resolve();assert not target.exists();target.mkdir(parents=True)
    result=copy.deepcopy(original);mapping={}
    for key,row in result['configurations'].items():
        src=Path(row['path']);dest=target/'configs'/key.split('/')[0]/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
        assert sha(src)==row['sha256'];shutil.copy2(src,dest);assert sha(dest)==row['sha256']
        mapping[row['path']]=str(dest);row['path']=str(dest)
    for job in result['jobs']:
        job['config']=mapping[job['config']];job['out']=str(target/'runs'/job['environment']/job['setting']/job['category'])
    # Only the destination addresses change. Input/source hashes, RNG seeds,
    # grids, model versions, objectives and execution all retain exact values.
    check=copy.deepcopy(result)
    for key,row in check['configurations'].items():row['path']=original['configurations'][key]['path']
    for row,old in zip(check['jobs'],original['jobs']):row['config']=old['config'];row['out']=old['out']
    assert check==original
    (target/'study.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(Path(a.study).parent/'preflight.json',target/'preflight.json')
    (target/'path_copy_proof.json').write_text(json.dumps({'source_study_sha256':sha(a.study),'only_config_and_output_paths_changed':True,'cells':len(result['jobs'])},indent=2)+'\n')
    print(json.dumps({'study':str(target/'study.json'),'unchanged_scientific_settings':True,'cells':len(result['jobs'])}))

if __name__=='__main__':main()
