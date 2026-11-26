"""Make output/config-path-only copies for reruns on the preserved input host."""
import argparse,copy,shutil,json
from pathlib import Path
from .freeze import read,write,sha,verify
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);a=p.parse_args();old=read(a.study);verify(old);s=copy.deepcopy(old);out=Path(a.out).resolve();assert not out.exists();out.mkdir(parents=True)
    for j in s['jobs']:
        f=out/'configs'/j['environment']/Path(j['config']).name;f.parent.mkdir(parents=True,exist_ok=True)
        if not f.exists():shutil.copy2(j['config'],f)
        assert sha(f)==j['config_sha256'];j['config']=str(f);j['out']=str(out/'evaluations'/j['id'])
    for x,y in zip(old['jobs'],s['jobs']):assert {k:v for k,v in x.items() if k not in ['config','out']}=={k:v for k,v in y.items() if k not in ['config','out']}
    write(out/'study.json',s);pre=read(Path(a.study).parent/'preflight.json');pre['inherited_after_output_path_only_copy']={'original_study_sha256':sha(a.study),'all_scientific_fields_equal':True};pre['study_sha256']=sha(out/'study.json');write(out/'preflight.json',pre)
    write(out/'path_copy_proof.json',{'only_output_and_config_paths_changed':True,'jobs':90,'original_study_sha256':sha(a.study),'new_study_sha256':sha(out/'study.json')});verify(s);print({'study':str(out/'study.json'),'jobs':90,'scientific_fields_unchanged':True})
if __name__=='__main__':main()
