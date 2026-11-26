"""Make a recorded, single-skill-trained action-manifold experiment queue."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--name',required=True);p.add_argument('--data',required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--count',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--strata',nargs='+',default=['supported_easy','supported_conflict'])
    args=p.parse_args();root=Path('independent_composition_artifacts');manifest=root/'manifests'/(args.name+'.json')
    if manifest.exists():raise FileExistsError(manifest)
    modelroot=root/'manifolds'/args.name
    models=[str(modelroot/skill/'best.pt') for skill in ['keepout','waypoint']];jobs=[]
    for skill,path in zip(['keepout','waypoint'],models):
        output=str(Path(path).parent)
        command=[sys.executable,'-u','-m','independent_composition.train_cvae','--env',args.env,'--skill',skill,
                 '--train',f'{args.data}/{skill}/train.npz','--validation',f'{args.data}/{skill}/val.npz',
                 '--out',output,'--steps',str(args.steps),'--seed',str(args.seed)]
        jobs.append({'id':f'train_{skill}','command':command,'result':output+'/DONE.json'})
    taskroot='composition_artifacts/tasks' if args.env=='tworoom' else 'cube_composition_artifacts/tasks_v2'
    variants=[('keepout','keepout',.25,1.),('waypoint','waypoint',.25,1.),('fixed','fixed',.25,1.),
              ('agreement','compose',0.,0.),('cycle','compose',.25,1.)]
    for label,method,cycle,reconstruction in variants:
        for kind in args.strata:
            assert kind in ['supported_easy','supported_conflict','coupled_ood']
            name=f'{label}_{kind}';output=str(root/args.name/'results'/name)
            command=[sys.executable,'-u','-m','independent_composition.evaluate_cvae','--env',args.env,
                     '--tasks',f'{taskroot}/val_{kind}.npz','--checkpoints',*models,'--out',output,
                     '--method',method,'--cycle',str(cycle),'--reconstruction',str(reconstruction),
                     '--iterations','16','--starts','4','--count',str(args.count)]
            jobs.append({'id':name,'command':command,'result':output+'/summary.json'})
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({'args':vars(args),'jobs':jobs,
        'scope':'Development validation only. Own-skill success precedes composition comparisons. New learned components remain completely private to each skill.'},indent=2)+'\n')
    print(manifest)


if __name__=='__main__':main()
