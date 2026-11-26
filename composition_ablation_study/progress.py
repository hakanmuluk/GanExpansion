"""Liveness and aggregate work counts only; never echo episode outcomes."""
import argparse,json
from pathlib import Path
from .status import live

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    state=json.loads((root/'queue/status.json').read_text());study=json.loads((root/'study.json').read_text())
    current=[]
    for name,row in state['jobs'].items():
        if row['state']!='running':continue
        r={'job':name,'pid':row['pid'],'live':live(row['pid'])}
        for line in reversed((root/'queue'/(name+'.log')).read_text().splitlines()):
            try:
                value=json.loads(line)
                if 'episode' in value:r['episodes_completed']=value['episode']+1;break
            except (ValueError,KeyError):pass
        current.append(r)
    print(json.dumps({'state':state['state'],'supervisor_pid':state['pid'],'supervisor_live':live(state['pid']),
        'complete':sum(v['state']=='complete' for v in state['jobs'].values()),'total':len(study['jobs']),
        'failed':[n for n,v in state['jobs'].items() if v['state']=='failed'],'current':current},indent=2))

if __name__=='__main__':main()
