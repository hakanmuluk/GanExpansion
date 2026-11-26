"""Durable serial evaluation queue; observation timeouts never relaunch jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();study_path=Path(a.study).resolve();study=json.loads(study_path.read_text())
    assert (study_path.parent/'preflight.json').is_file()
    assert json.loads((study_path.parent/'preflight.json').read_text())['passed']
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    state={'pid':os.getpid(),'started':time.time(),'state':'running','jobs':{}}
    def save():(out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    priorities=['h8_n20','h8_n0','h4_n20','h12_n20','h16_n20','h8_n5','h8_n10','h8_n40','h8_n0_parameterized']
    def key(j):
        index=(priorities.index(j['setting']) if j['setting'] in priorities else
               20 if j['search']=='single' else 9+study['random_search_seeds'].index(j['seed']))
        return index,['supported_easy','supported_conflict','coupled_ood'].index(j['category']),j['environment']
    for job in sorted(study['jobs'],key=key):
        assert not Path(job['out']).exists(),('Output already exists; investigate before a new queue',job['out'])
        record={'state':'running','started':time.time(),'out':job['out']};state['jobs'][job['id']]=record;save()
        with (out/(job['id']+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','composition_ablation_study.evaluate','--study',str(study_path),'--job',job['id']],
                                   stdout=log,stderr=subprocess.STDOUT)
            record['pid']=child.pid;save();code=child.wait()
        record.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());save()
        print(json.dumps({'job':job['id'],'state':record['state'],'seconds':record['finished']-record['started']}),flush=True)
        if code:state['state']='failed';save();raise SystemExit(code)
        assert (Path(job['out'])/'DONE.json').exists()
    state.update(state='complete',finished=time.time());save()
    print(json.dumps({'complete':True,'cells':len(study['jobs']),'episodes':64*len(study['jobs'])}),flush=True)

if __name__=='__main__':main()
