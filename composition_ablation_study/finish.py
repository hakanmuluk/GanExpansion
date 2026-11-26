"""After the live sweep, run final controls and uncontended timing/analysis."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .status import live

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();study=Path(a.study).resolve();root=study.parent
    target=root/'finishing';assert not target.exists();target.mkdir()
    record={'pid':os.getpid(),'state':'waiting_for_existing_queue','started':time.time(),'stages':[]}
    def save():(target/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    save();stale=0;last=None
    while True:
        state=json.loads((root/'queue/status.json').read_text())
        assert state['state']!='failed','The evaluation queue failed; inspect it without restarting.'
        complete=sum(r['state']=='complete' for r in state['jobs'].values())
        if complete!=last:print(json.dumps({'completed_cells':complete,'waiting':state['state']!='complete'}),flush=True);last=complete
        if state['state']=='complete':break
        if not live(state['pid']):
            stale+=1
            assert stale<3,'Queue supervisor missing on repeated direct process checks; no automatic restart.'
        else:stale=0
        time.sleep(30)
    def run(name,module,argv):
        row={'stage':name,'state':'running','started':time.time()};record['stages'].append(row);record['state']='running';save()
        with (target/(name+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',module]+argv,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=child.pid;save();code=child.wait()
        row.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());save()
        if code:record['state']='failed';save();raise SystemExit(code)
        print(json.dumps({'stage':name,'complete':True}),flush=True)
    run('audit','composition_ablation_study.audit',['--study',str(study),'--out',str(root/'final_audit.json'),'--require-complete'])
    run('timing','composition_ablation_study.timing',['--study',str(study),'--out',str(root/'latency')])
    run('analysis','composition_ablation_study.analyze',['--study',str(study),'--out',str(root/'aggregate')])
    run('figures','composition_ablation_study.figures',['--aggregate',str(root/'aggregate'),
        '--latency',str(root/'latency/summary.json'),'--out',str(root/'aggregate/figures')])
    record.update(state='complete',finished=time.time());save()
    print(json.dumps({'all_evaluations_audited_timed_and_aggregated':True}),flush=True)

if __name__=='__main__':main()
