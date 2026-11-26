"""Record four independent adapter trainings and development-only comparisons."""
import json
import argparse
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-reference',action='store_true');args=parser.parse_args()
    root=Path('independent_composition_artifacts');name='action_adapter_denoising_v15' if args.base_reference else 'action_adapter_denoising_v14'
    plan=root/(name+'_plan.json')
    if plan.exists():raise FileExistsError(plan)
    train_queues={};eval_queues={}
    for env,dirname in [('tworoom','tworoom_diverse_merged_v5'),('cube','cube_broad_v3')]:
        train_queues[env]=[]
        for skill in ['keepout','waypoint']:
            queue=root/f'{name}_train_{env}_{skill}'
            model=root/name/'models'/env/skill
            command=[sys.executable,'-m','independent_composition.train_action_adapter_diffusion',
                     '--env',env,'--skill',skill,'--train',str(root/'data'/dirname/skill/'train.npz'),
                     '--validation',str(root/'data'/dirname/skill/'val.npz'),'--out',str(model),
                     '--steps','6000','--batch','256']
            if args.base_reference:
                command+=['--score-residual','--initialize',str(root/'action_adapter_denoising_v14/models'/env/skill/'best.pt')]
            manifest=Path(str(queue)+'_manifest.json')
            manifest.write_text(json.dumps({'scope':'Single-skill original base + LoRA/FiLM adapter only',
                'jobs':[{'id':'train','command':command,'result':str(model/'DONE.json')}]},indent=2)+'\n')
            train_queues[env].append({'directory':str(queue),'manifest':str(manifest)})
        queue=root/f'{name}_evaluate_{env}';ready=root/name/f'{env}_ready.json'
        jobs=[{'id':'wait_two_own_trainings','command':[sys.executable,'-m','independent_composition.wait_jobs',
             '--queues',*[x['directory'] for x in train_queues[env]],'--out',str(ready)],'result':str(ready)}]
        models=[str(root/name/'models'/env/s/'best.pt') for s in ['keepout','waypoint']]
        taskroot='composition_artifacts/tasks' if env=='tworoom' else 'cube_composition_artifacts/tasks_v2'
        for method in ['keepout','waypoint','ratio','mean']:
            for kind in ['supported_easy','supported_conflict','coupled_ood']:
                ident=method+'_'+kind;output=root/name/'results'/env/ident
                command=[sys.executable,'-m','independent_composition.evaluate_action_adapters',
                    '--env',env,'--checkpoints',*models,'--tasks',f'{taskroot}/val_{kind}.npz',
                    '--out',str(output),'--method',method,'--count',str(16 if env=='tworoom' else 8),
                    '--execute',str(4 if env=='tworoom' else 8),'--sampling-steps','20','--cold-time','70','--warm-time','70']
                jobs.append({'id':ident,'command':command,'result':str(output/'summary.json')})
        manifest=Path(str(queue)+'_manifest.json')
        manifest.write_text(json.dumps({'scope':'Development only. Judge and route-free base plus adapters. Own-skill checks precede joint/mean comparisons; no final bank access.','jobs':jobs},indent=2)+'\n')
        eval_queues[env]={'directory':str(queue),'manifest':str(manifest)}
    plan.write_text(json.dumps({'train_queues':train_queues,'eval_queues':eval_queues},indent=2)+'\n')
    print(str(plan))


if __name__=='__main__':main()
