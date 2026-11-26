"""Save aggregate-only evidence for original Cube Z/W composition trials."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .summarize_execution_study import read_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--queues',nargs='+',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);assert not out.exists()
    root=Path('independent_composition_artifacts');report={'created_unix':time.time(),
        'scope':'Cube-only development pilots; original network modules/weights fixed, no inference simulator/task score/new trained network. W optimization explicitly requested by user. Raw trajectories are not exported.','queues':{}}
    files={}
    for queue in map(Path,args.queues):
        assert (queue/'DONE.json').exists(),queue
        status=json.loads((queue/'status.json').read_text());jobs=[]
        for job in status['jobs']:
            assert job['status']=='complete'
            directory=Path(job['result']).parent;c,s,rows,counts=read_run(directory)
            group=next(g for g in ['supported_easy','supported_conflict'] if job['id'].endswith(g))
            merge=c['composition']['merge']
            reference=root/(('original_composition_v25_cube/cube_waypoint_' if merge=='waypoint' else 'original_composition_v22_cube/cube_xyz025_')+group)
            rc,rs,rr,rn=read_run(reference)
            assert c['args']['env']=='cube' and c['args']['execute']==8 and c['args']['count']==8
            assert c['task_sha256']==rc['task_sha256'] and c['original_model_files']==rc['original_model_files']
            assert c['args']['seed']==rc['args']['seed'] and len(rows)==len(rr)
            files.update(c['original_model_files'])
            plan=json.loads((directory/'planning.json').read_text());flat=[item for episode in plan for item in episode]
            losses={key:float(np.mean([item[key][0] for item in flat if key in item])) for key in ['agreement','style_tether','style_continuity'] if any(key in item for item in flat)}
            sign_switches=None
            if all('selected_codes' in item for item in flat):
                sign_switches=sum(int((np.sign(np.asarray([x['selected_codes'][0][0] for x in ep]))[1:]!=np.sign(np.asarray([x['selected_codes'][0][0] for x in ep]))[:-1]).sum()) for ep in plan)
            jobs.append({'id':job['id'],'summary':s,'counts':counts,'configuration':c,
                'fixed_code_reference':{'summary':rs,'counts':rn},
                'paired_joint_outcomes':{'trial_only':sum(a['success'] and not b['success'] for a,b in zip(rows,rr)),
                    'fixed_only':sum(b['success'] and not a['success'] for a,b in zip(rows,rr)),
                    'both':sum(a['success'] and b['success'] for a,b in zip(rows,rr))},
                'observed_trajectory_mean_optimizer_losses':losses,'keepout_Z_sign_switches':sign_switches,
                'reproduction_command':job['command']})
        report['queues'][str(queue)]={'jobs':jobs,'manifest_sha256':status['manifest_sha256']}
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in files.items())
    report['original_weights_verified_unchanged']=True
    report['original_weight_sha256']=files
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'out':str(out),'queues':len(report['queues']),'jobs':sum(len(q['jobs']) for q in report['queues'].values())}))


if __name__=='__main__':main()
