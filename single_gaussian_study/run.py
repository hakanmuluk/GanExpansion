"""Independent-process queue; no per-episode restarts or result selection."""
import argparse,subprocess,sys,os,time,json
from pathlib import Path
from .freeze import read,write,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();s=read(a.study);verify(s)
    root=Path(a.study).parent;assert read(root/'preflight.json')['passed'];out=root/'queue';assert not out.exists();out.mkdir()
    status={'pid':os.getpid(),'state':'running','jobs':{},'started':time.time()}
    def save():write(out/'status.json',status)
    save()
    pending=list(s['jobs']);active={};failed=False;pilot_verified=not bool(s.get('serial_pilot'))
    status['independent_evaluation_processes']=s['evaluation_workers'];save()
    while pending or active:
        for name,(child,j,log) in list(active.items()):
            code=child.poll()
            if code is None:continue
            log.close();r=status['jobs'][name]
            r.update(state='complete' if code==0 else 'failed',finished=time.time(),returncode=code)
            failed=failed or code!=0
            if code==0:
                assert (Path(j['out'])/'DONE.json').exists()
                if j['environment']=='cube' and j['steps']==0 and j['seed']==241 and j['category']=='supported_easy' and not pilot_verified:
                    from composition_ablation_study.evaluate import reference_check
                    proof=reference_check(Path(j['out']),Path(s['serial_pilot']),'cube',64)
                    write(root/'serial_parallel_native_parity.json',proof);pilot_verified=True
            del active[name];save()
            print(json.dumps({'completed':sum(x['state']=='complete' for x in status['jobs'].values()),'total':90,'job':name,'state':r['state']}),flush=True)
        if failed:
            if active:time.sleep(1);continue
            status.update(state='failed',finished=time.time());save();raise SystemExit(1)
        for j in list(pending):
            if len(active)>=s['evaluation_workers']:break
            if j['steps']>0:
                baseline=next(x for x in s['jobs'] if x['environment']==j['environment'] and x['category']==j['category'] and x['seed']==j['seed'] and x['steps']==0)
                if status['jobs'].get(baseline['id'],{}).get('state')!='complete' or not pilot_verified:continue
            assert not Path(j['out']).exists()
            r={'state':'running','started':time.time(),'out':j['out']};status['jobs'][j['id']]=r
            log=(out/(j['id']+'.log')).open('w')
            child=subprocess.Popen([sys.executable,'-u','-m','single_gaussian_study.evaluate','--study',a.study,'--job',j['id']],stdout=log,stderr=subprocess.STDOUT)
            r['pid']=child.pid;active[j['id']]=(child,j,log);pending.remove(j);save()
        if active:time.sleep(1)
        elif pending:raise RuntimeError('No eligible job; inspect baseline dependencies')
    status.update(state='complete',finished=time.time());save()

if __name__=='__main__':main()
