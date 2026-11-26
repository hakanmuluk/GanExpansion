"""Aggregate-only milestones before final uncontended timing."""
import argparse,os,time,sys,subprocess
from pathlib import Path
from .freeze import read,write
from .finish import live
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--exports');a=p.parse_args();root=Path(a.study).parent;out=Path(a.exports) if a.exports else root.parent/'milestones';out.mkdir(exist_ok=False)
    state={'pid':os.getpid(),'state':'waiting','completed':[]}
    def save():write(out/'status.json',state)
    save()
    for n in [18,36,54,72]:
        while True:
            q=read(root/'queue/status.json');assert q['state']!='failed' and not any(x['state']=='failed' for x in q['jobs'].values())
            if sum(x['state']=='complete' for x in q['jobs'].values())>=n:break
            assert live(q['pid']),'Queue no longer live; inspect before continuing';time.sleep(30)
        name=f'milestone_{n:03d}';aggregate=root/name;state['state']='creating_'+name;save()
        for module,args in [('audit_cli',['--study',a.study,'--out',str(out/(name+'_audit.json'))]),
            ('analyze',['--study',a.study,'--out',str(aggregate),'--partial']),
            ('export',['--study',a.study,'--aggregate',str(aggregate),'--out',str(out/name),'--partial'])]:
            with (out/(name+'_'+module+'.log')).open('w') as log:
                subprocess.run([sys.executable,'-u','-m','single_gaussian_study.'+module]+args,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(name);state['state']='waiting';save()
    state['state']='complete';save()
if __name__=='__main__':main()
