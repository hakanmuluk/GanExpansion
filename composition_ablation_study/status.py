"""Read authoritative queue state plus actual Linux process liveness."""
import argparse,json
from pathlib import Path

def live(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].strip().split()[0]
        return stat!='Z' and bool(Path(f'/proc/{pid}/cmdline').read_bytes())
    except FileNotFoundError:return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    s=json.loads((root/'queue/status.json').read_text());m=json.loads((root/'study.json').read_text())
    current=[]
    for name,row in s['jobs'].items():
        if row['state']=='running':
            r={'job':name,'pid':row['pid'],'live':live(row['pid'])}
            log=root/'queue'/(name+'.log')
            if log.exists():
                for line in reversed(log.read_text().splitlines()):
                    try:
                        v=json.loads(line)
                        if 'episode' in v:r['last_episode_progress']={k:v[k] for k in ['episode','joint','goal','keepout','waypoint']};break
                    except (ValueError,KeyError):pass
            current.append(r)
    print(json.dumps({'queue_state':s['state'],'supervisor_pid':s['pid'],'supervisor_live':live(s['pid']),
        'complete':sum(r['state']=='complete' for r in s['jobs'].values()),'total':len(m['jobs']),
        'failed':[n for n,r in s['jobs'].items() if r['state']=='failed'],'current':current},indent=2))

if __name__=='__main__':main()
