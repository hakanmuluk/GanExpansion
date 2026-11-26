"""Aggregate original-code agreement-window ablations; no trajectory export."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .summarize_execution_study import read_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='independent_composition_artifacts')
    p.add_argument('--out',required=True);args=p.parse_args();root=Path(args.root);out=Path(args.out)
    assert not out.exists(),out
    report={'created_unix':time.time(),'scope':'Development ablation of action-agreement window only, within each execution rate. Original frozen models generate16actions;20code-Adam steps/4starts;no inference simulator queries.','queues':{}}
    files={}
    for name in ['cube_e8','cube_e4','tworoom_e8']:
        queue=root/('original_agreement_window_v27_'+name+'_queue');assert (queue/'DONE.json').exists(),queue
        status=json.loads((queue/'status.json').read_text());result=[]
        for job in status['jobs']:
            assert job['status']=='complete'
            directory=Path(job['result']).parent;c,s,rows,counts=read_run(directory)
            group=next(g for g in ['supported_easy','supported_conflict'] if job['id'].endswith(g))
            if name=='cube_e8':reference=root/('original_composition_v23_cube/cube_prior_'+group)
            elif name=='tworoom_e8':reference=root/('original_composition_v23_tworoom/tw_history_'+group)
            else:reference=queue.parent/('original_agreement_window_v27_cube_e4/e4_h16_'+group)
            rc,rs,rr,rn=read_run(reference)
            assert c['task_sha256']==rc['task_sha256']
            assert c['original_model_files']==rc['original_model_files']
            for key in ['execute','count','seed','model_seed']:assert c['args'][key]==rc['args'][key]
            clean=lambda d:{k:v for k,v in d.items() if k!='agreement_horizon'}
            assert clean(c['composition'])==clean(rc['composition'])
            assert rc['composition'].get('agreement_horizon',16)==16 and len(rows)==len(rr)
            files.update(c['original_model_files'])
            result.append({'id':job['id'],'agreement_horizon':c['composition']['agreement_horizon'],
                'execute':c['args']['execute'],'group':group,'summary':s,'counts':counts,
                'full16_reference':{'summary':rs,'counts':rn},
                'paired_joint_outcomes':{'prefix_only':sum(a['success'] and not b['success'] for a,b in zip(rows,rr)),
                    'full16_only':sum(b['success'] and not a['success'] for a,b in zip(rows,rr)),
                    'both':sum(a['success'] and b['success'] for a,b in zip(rows,rr))},
                'configuration':c,'reproduction_command':job['command']})
        report['queues'][name]={'jobs':result,'manifest_sha256':status['manifest_sha256']}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in files.items())
    report['original_weight_hashes_verified']=True
    report['prefix_gradient_smoke']=json.loads((root/'original_agreement_window_v27_smoke.json').read_text())
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'out':str(out),'jobs':sum(len(q['jobs']) for q in report['queues'].values())}))


if __name__=='__main__':main()
