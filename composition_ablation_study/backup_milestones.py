"""Periodic aggregate-only checkpoints; no intervention in the GPU queue."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .status import live

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--exports',required=True);a=p.parse_args()
    study=Path(a.study).resolve();root=study.parent;exports=Path(a.exports).resolve();exports.mkdir(parents=True,exist_ok=True)
    receipt={'pid':os.getpid(),'state':'waiting','completed_backups':[]};status=exports/'milestone_status.json'
    def save():status.write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    for threshold in [30,42,54,66]:
        while True:
            state=json.loads((root/'queue/status.json').read_text())
            if state['state']=='failed':raise RuntimeError('Evaluation queue failed; no automatic restart')
            n=sum(r['state']=='complete' for r in state['jobs'].values())
            if n>=threshold:break
            assert live(state['pid']),'Queue no longer live; investigate authoritative status'
            time.sleep(30)
        # These milestones precede the final12 single-skill cells. All final
        # GPU timing waits until the complete queue, so this CPU-only processing
        # is not concurrent with the dedicated uncontended latency benchmark.
        name=f'milestone_{threshold:03d}';receipt['state']='creating_'+name;save()
        commands=[['composition_ablation_study.audit','--study',str(study),'--out',str(root/(name+'_audit.json'))],
            ['composition_ablation_study.analyze','--study',str(study),'--out',str(root/'aggregate_working'),'--partial'],
            ['composition_ablation_study.export','--study',str(study),'--aggregate',str(root/'aggregate_working'),
             '--out',str(exports/name),'--partial']]
        for argv in commands:subprocess.run([sys.executable,'-u','-m']+argv,check=True)
        receipt['completed_backups'].append(name);receipt['state']='waiting';save()
    receipt['state']='complete';save()

if __name__=='__main__':main()
