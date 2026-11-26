"""Safe counts/liveness only: never print individual episode outcomes."""
import argparse,os,json
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root);s=read(r/'queue/status.json');cur=[]
    for name,j in s['jobs'].items():
        if j['state']!='running':continue
        folder=Path(j['out']);item={'job':name,'pid':j['pid'],'live':live(j['pid'])}
        f=folder/'records.json'
        if f.exists():
            try:item['episodes_completed']=len(read(f))
            except json.JSONDecodeError:pass
        cur.append(item)
    print(json.dumps({'state':s['state'],'supervisor_pid':s['pid'],'supervisor_live':live(s['pid']),
        'complete':sum(j['state']=='complete' for j in s['jobs'].values()),'total':90,
        'failed':[n for n,j in s['jobs'].items() if j['state']=='failed'],'current':cur},indent=2))
if __name__=='__main__':main()
