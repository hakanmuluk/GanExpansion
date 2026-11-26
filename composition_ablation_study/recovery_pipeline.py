"""Reconstruct exact frozen inputs after a confirmed runtime reset, then rerun."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
from .analyze import read,sha
from .evaluate import verify
from .status import live

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',required=True);a=p.parse_args()
    control=Path(a.control).resolve();status=control/'pipeline_status.json'
    assert not status.exists(),'Inspect existing recovery; never launch a duplicate.'
    root=Path('/content/cs585_composition_ablation_20261126');study_path=root/'study_v1/study.json'
    study=read(study_path);study_digest=sha(study_path);original=Path(study['original_root'])
    assert read(control/'restore_proof.json')['restored'] and read(control/'native_check.json')['cube_native_reset_passed']
    env=dict(os.environ,PYTHONPATH=str(root)+':'+str(original)+':'+str(original/'cube'),
             MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    record={'pid':os.getpid(),'state':'running','started':time.time(),'stages':[],
            'study_sha256':study_digest,'reason':'Confirmed Colab reset removed all native records. Exact-hash restoration; complete rerun, no new independent contexts.'}
    def save():status.write_text(json.dumps(record,indent=2)+'\n')
    save()
    def run(name,module,args):
        row={'stage':name,'state':'running','started':time.time()};record['stages'].append(row);save()
        with (control/(name+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',module]+args,cwd=original,env=env,
                                   stdout=log,stderr=subprocess.STDOUT)
            row['pid']=child.pid;save();code=child.wait()
        row.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());save()
        if code:raise RuntimeError(f'{name} exited with code {code}; inspect log without restarting earlier stages')
    def launch(name,module,args,receipt):
        with (study_path.parent/(name+'_supervisor.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',module]+args,cwd=original,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        proof={'pid':child.pid,'module':module,'args':args,'recovery_run':True}
        (study_path.parent/receipt).write_text(json.dumps(proof,indent=2)+'\n')
        return child.pid
    try:
        for name,module,workers in [('tworoom','independent_composition.generate_final_tworoom',[]),
                                    ('cube','independent_composition.generate_cube_frozen_bank',['--workers','4'])]:
            e=study['environments'][name];bank=Path(next(iter(e['tasks'].values()))).parent
            assert not bank.exists(),'Preserve existing banks; this recovery requires empty destinations.'
            run(name+'_bank_restoration',module,['--protocol',e['old_freeze'],'--out',str(bank),'--count','64']+workers)
        verify(study)
        bank_proof={name:{kind:sha(path) for kind,path in e['tasks'].items()} for name,e in study['environments'].items()}
        assert all(bank_proof[name]==e['bank_sha256'] for name,e in study['environments'].items())
        (control/'case_restoration_proof.json').write_text(json.dumps({'all_six_exact_hashes_match':True,'bank_sha256':bank_proof,'raw_arrays_exported':False},indent=2)+'\n')
        # The preparation CLI appends random jobs. Validate on a disposable copy,
        # then compare all settings; never append to the frozen 84-job descriptor.
        validation=control/'preflight_validation';validation.mkdir()
        clone=copy.deepcopy(study);clone['jobs']=[j for j in clone['jobs'] if j['search']!='random']
        assert len(clone['jobs'])==66
        clone_path=validation/'study.json';clone_path.write_text(json.dumps(clone,indent=2)+'\n')
        run('restored_gpu_preflight','composition_ablation_study.preflight',['--study',str(clone_path)])
        checked=read(clone_path);byid={j['id']:j for j in study['jobs']}
        for j in checked['jobs']:j['out']=byid[j['id']]['out']
        assert checked==study,'Restored profiling/settings differ; do not retune or launch comparisons.'
        assert read(validation/'preflight.json')['passed'] and sha(study_path)==study_digest
        run('original_reference_restoration','composition_ablation_study.rebuild_reference',['--study',str(study_path),'--run'])
        verify(study);assert sha(study_path)==study_digest
        assert not (study_path.parent/'queue').exists()
        pids={}
        pids['queue']=launch('queue','composition_ablation_study.run_queue',[
            '--study',str(study_path),'--out',str(study_path.parent/'queue')],'queue_process.json')
        deadline=time.time()+60
        while not (study_path.parent/'queue/status.json').exists():
            assert live(pids['queue']) and time.time()<deadline
            time.sleep(.25)
        pids['finish']=launch('finish','composition_ablation_study.finish',['--study',str(study_path)],'finish_process.json')
        deadline=time.time()+60
        while not (study_path.parent/'finishing/status.json').exists():
            assert live(pids['finish']) and time.time()<deadline
            time.sleep(.25)
        pids['milestones']=launch('milestone','composition_ablation_study.backup_milestones',[
            '--study',str(study_path),'--exports',str(root/'milestone_exports')],'milestone_process.json')
        # The milestone utility's older receipt convention is at the outer root.
        (root/'milestone_process.json').write_text(json.dumps({'pid':pids['milestones'],'recovery_run':True},indent=2)+'\n')
        pids['preservation']=launch('preservation','composition_ablation_study.archive_finished',[
            '--study',str(study_path)],'preservation_process.json')
        record.update(state='evaluations_launched',pids=pids,finished=time.time(),all_six_case_hashes_restored=True)
        save();print(json.dumps({'recovery_complete':True,'evaluations_launched':True,'pids':pids}),flush=True)
    except BaseException as exc:
        record.update(state='failed',error=repr(exc),finished=time.time());save();raise

if __name__=='__main__':main()
