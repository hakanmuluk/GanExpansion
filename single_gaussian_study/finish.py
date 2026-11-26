"""Wait for the existing90-cell queue; audit, time, analyze, render, export."""
import argparse,subprocess,sys,time,os,json
from pathlib import Path
from .freeze import read,write

def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out');a=p.parse_args();root=Path(a.study).parent;out=Path(a.out) if a.out else root/'finishing';assert not out.exists();out.mkdir()
    status={'pid':os.getpid(),'state':'waiting_for_existing_queue','stages':[]}
    def save():write(out/'status.json',status)
    save();stale=0
    while True:
        q=read(root/'queue/status.json')
        assert q['state']!='failed' and not any(j['state']=='failed' for j in q['jobs'].values()),'Queue failed; inspect without restarting'
        if q['state']=='complete':break
        stale=0 if live(q['pid']) else stale+1;assert stale<3,'Queue process missing on repeated direct checks'
        time.sleep(20)
    def run(name,args):
        row={'stage':name,'state':'running','started':time.time()};status['state']='running';status['stages'].append(row);save()
        with (out/(name+'.log')).open('w') as log:
            module='audit_cli' if name=='audit' else name
            child=subprocess.Popen([sys.executable,'-u','-m','single_gaussian_study.'+module]+args,stdout=log,stderr=subprocess.STDOUT);row['pid']=child.pid;save();code=child.wait()
        row.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());save()
        if code:status['state']='failed';save();raise SystemExit(code)
    run('audit',['--study',a.study,'--out',str(root/'final_audit.json'),'--require-complete'])
    run('timing',['--study',a.study,'--out',str(root/'latency')])
    run('analyze',['--study',a.study,'--out',str(root/'aggregate')])
    run('figures',['--aggregate',str(root/'aggregate'),'--latency',str(root/'latency/summary.json'),'--out',str(root/'aggregate/figures')])
    run('representatives',['--study',a.study,'--out',str(root/'aggregate/representatives')])
    run('export',['--study',a.study,'--out',str(root.parent/'complete_export_v1')])
    status.update(state='complete',finished=time.time());save()

if __name__=='__main__':main()
