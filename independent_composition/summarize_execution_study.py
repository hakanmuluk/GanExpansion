"""Aggregate matched execution-rate runs without exporting trajectory arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import time


REFERENCES={
    'waypoint':'original_composition_v25_cube/cube_waypoint',
    'keepout':'original_composition_v25_cube/cube_keepout',
    'residual025':'original_composition_v22_cube/cube_xyz025',
    'blend_z050':'original_composition_v24_cube/cube_blend_z050',
}


def read_run(directory):
    assert (directory/'DONE.json').exists(),directory
    config=json.loads((directory/'config.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    rows=json.loads((directory/'records.json').read_text())
    done=json.loads((directory/'DONE.json').read_text())
    assert done['original_models_unchanged'] and not done['inference_environment_queries']
    count_keys=['success','goal_success','keepout_full_success','waypoint_full_success',
                'keepout_satisfied','waypoint_satisfied','dropped','robot_table_contact_in_transport']
    return config,summary,rows,{key:sum(bool(row.get(key,False)) for row in rows) for key in count_keys}


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();queue=Path(args.queue);root=queue.parent;out=Path(args.out)
    assert (queue/'DONE.json').exists(),'Wait for the existing queue; do not restart it'
    assert not out.exists(),out
    status=json.loads((queue/'status.json').read_text());results=[];original_files={}
    for job in status['jobs']:
        assert job['status']=='complete'
        directory=Path(job['result']).parent
        config,summary,rows,counts=read_run(directory)
        ident=job['id'];execute=int(ident.split('_')[0][1:])
        group=next(g for g in ['supported_easy','supported_conflict'] if ident.endswith(g))
        method=ident[len('e'+str(execute))+1:-len('_'+group)]
        ref=root/(REFERENCES[method]+'_'+group)
        rc,rs,rr,rn=read_run(ref)
        assert config['task_sha256']==rc['task_sha256']
        assert config['original_model_files']==rc['original_model_files']
        assert config['composition']==rc['composition']
        assert len(rows)==len(rr) and config['args']['seed']==rc['args']['seed']
        assert config['args']['execute']==execute and rc['args']['execute']==8
        original_files.update(config['original_model_files'])
        paired={'shorter_only':sum(a['success'] and not b['success'] for a,b in zip(rows,rr)),
                'eight_only':sum(b['success'] and not a['success'] for a,b in zip(rows,rr)),
                'both':sum(a['success'] and b['success'] for a,b in zip(rows,rr))}
        results.append({'id':ident,'method':method,'group':group,'execute':execute,
            'summary':summary,'counts':counts,'reference_execute8':{'summary':rs,'counts':rn},
            'paired_joint_outcomes':paired,'configuration':config,'reproduction_command':job['command']})
    for path,sha in original_files.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    result={'created_unix':time.time(),'scope':'Development pilot. Same scenes/checkpoints/codes/output rule/16-action prediction; only executed chunk length changes. No trajectory arrays exported.',
        'original_weights_verified_unchanged':True,'manifest_sha256':status['manifest_sha256'],
        'jobs':results,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'out':str(out),'jobs':len(results),'all_original_weights_unchanged':True}))


if __name__=='__main__':main()
