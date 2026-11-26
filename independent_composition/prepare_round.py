"""Create explicit development-only residual-score experiment manifests."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=['tworoom', 'cube'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--data')
    p.add_argument('--model-root', required=True)
    p.add_argument('--steps', type=int, default=16000)
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--strata', nargs='+', default=['supported_easy', 'supported_conflict'])
    p.add_argument('--executes', nargs='+', type=int, default=[8, 4])
    p.add_argument('--warm-times', nargs='+', type=int, default=[99, 50])
    args = p.parse_args()
    root = Path('independent_composition_artifacts')
    manifest_path = root / 'manifests' / (args.name + '.json')
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    jobs = []
    models = [str(Path(args.model_root) / f'{args.env}_{s}_v1' / 'best.pt')
              for s in ['keepout', 'waypoint']]
    if args.data:
        for skill, model in zip(['keepout', 'waypoint'], models):
            output = str(Path(model).parent)
            command = [sys.executable, '-u', '-m', 'independent_composition.train_diffusion',
                       '--env', args.env, '--skill', skill, '--train', f'{args.data}/{skill}/train.npz',
                       '--validation', f'{args.data}/{skill}/val.npz', '--out', output,
                       '--steps', str(args.steps), '--score-residual']
            jobs.append({'id': f'train_{skill}', 'command': command, 'result': output + '/DONE.json'})
    taskroot = 'composition_artifacts/tasks' if args.env == 'tworoom' else 'cube_composition_artifacts/tasks_v2'
    variants = [('keepout', 99, 8), ('waypoint', 99, 8)]
    variants += [('ratio', warm, execute) for warm in args.warm_times for execute in args.executes]
    for method, warm, execute in variants:
        for kind in args.strata:
            assert kind in ['supported_easy', 'supported_conflict', 'coupled_ood']
            name = f'{method}_warm{warm}_execute{execute}_{kind}'
            output = str(root / args.name / 'results' / name)
            command = [sys.executable, '-u', '-m', 'independent_composition.evaluate_diffusion',
                       '--env', args.env, '--tasks', f'{taskroot}/val_{kind}.npz',
                       '--checkpoints', *models, '--out', output, '--method', method,
                       '--warm-time', str(warm), '--execute', str(execute), '--count', str(args.count),
                       '--seed', str(args.seed)]
            jobs.append({'id': name, 'command': command, 'result': output + '/summary.json'})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({'args': vars(args), 'jobs': jobs,
        'scope': 'Development validation only. No final test bank access; no simulator in inference.'}, indent=2) + '\n')
    print(manifest_path)


if __name__ == '__main__':
    main()
