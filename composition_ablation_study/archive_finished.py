"""Preserve complete data-free evidence after the existing finish pipeline."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .status import live

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
    study=Path(a.study).resolve();root=study.parent;outer=root.parent
    out=root/'preservation';assert not out.exists();out.mkdir()
    record={'pid':os.getpid(),'state':'waiting_for_existing_finish','stages':[]}
    def save():(out/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    save();stale=0
    while True:
        state=json.loads((root/'finishing/status.json').read_text())
        assert state['state']!='failed','Final processing failed; investigate without restarting it.'
        if state['state']=='complete':break
        stale=0 if live(state['pid']) else stale+1
        assert stale<3,'Finish supervisor absent on repeated checks; no automatic restart.'
        time.sleep(30)
    def run(name,module,args):
        row={'stage':name,'state':'running','started':time.time()};record['stages'].append(row);record['state']='running';save()
        with (out/(name+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',module]+args,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=child.pid;save();code=child.wait()
        row.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());save()
        if code:record['state']='failed';save();raise SystemExit(code)
        print(json.dumps({'stage':name,'complete':True}),flush=True)
    snapshot=outer/'reference_snapshot_b'
    run('representatives','composition_ablation_study.representatives',[
        '--study',str(study),'--out',str(root/'aggregate/representatives')])
    run('source_and_weights','composition_ablation_study.preserve_reference',[
        '--study',str(study),'--out',str(snapshot)])
    run('isolated_source_check','composition_ablation_study.check_source_bundle',[
        '--archive',str(snapshot/'reference_code_and_protocol.zip'),'--study',str(study),
        '--out',str(snapshot/'import_check')])
    run('aggregate_export','composition_ablation_study.export',[
        '--study',str(study),'--out',str(outer/'finished_export_v1')])
    record.update(state='complete',finished=time.time());save()
    print(json.dumps({'complete':True,'raw_episode_data_exported':False,
        'scope':'Complete aggregate/figure/source/weight preservation. Report interpretation and visual QA still required.'}),flush=True)

if __name__=='__main__':main()
