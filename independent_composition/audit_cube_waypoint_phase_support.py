"""Measure existing waypoint teacher/adapter code choices by transport phase.

Own-task training-distribution diagnostic only. The teacher calculation is its
existing phase-four translation formula at recorded native cube positions; no
new rollout, joint geometry, simulator search, or model selection is performed.
Raw states, poses and model outputs remain on the machine running the audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cube.cube_gan.expert import Style
from cube.cube_gan.style import decode_style
from cube_addons.evaluate import load_adapter
from cube_addons.geometry import axis_frame
from .cube_diverse_demonstrations import DiverseSkillExpert, lookahead
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'checkpoint', 'base', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--rollouts', type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(1)
    model, _, _ = load_adapter(args.checkpoint, args.base, 'cpu')
    model.eval().requires_grad_(False)
    codes = np.array([-2., -1., -.2, .2, 1., 2.], np.float32)[:, None]
    records = {name: [] for name in ['before_visit', 'after_visit_near', 'after_visit_late']}
    paths = sorted(Path(args.source).glob('learner_*.npz'))[:args.rollouts]
    assert paths
    with torch.no_grad():
        for path in paths:
            with np.load(path) as archive:
                data = {k: archive[k] for k in ['states', 'contexts', 'poses', 'initial',
                                               'goal', 'spec', 'base_z']}
            mode, amplitude, speed = decode_style(torch.as_tensor(data['base_z']))
            style = Style(int(mode), float(amplitude), float(speed))
            teachers = [DiverseSkillExpert(data['initial'], data['goal'], style,
                                          data['spec'], u, 'waypoint') for u in codes]
            _, _, axis, _, _ = axis_frame(data['initial'], data['goal'], data['initial'])
            along = np.r_[axis, 0.]
            for i in range(0, len(data['states']), 2):
                context = data['contexts'][i]
                cube = data['poses'][i, 3:]
                if not (context[13] > .5 and context[14] > .5 and cube[2] > .165
                        and np.linalg.norm(cube[:2] - data['goal'][:2]) > .035):
                    continue
                visited = bool(context[15] > .5)
                after_distance = float((cube - data['spec'][:3]) @ along)
                name = ('before_visit' if not visited else
                        'after_visit_late' if after_distance > .08 else 'after_visit_near')
                target = np.stack([np.clip((lookahead(t.points[2:] if visited else t.points[:3],
                    cube, .032) - cube) * 15 * style.speed, -style.speed, style.speed)
                    for t in teachers])
                repeat = lambda x: torch.as_tensor(np.repeat(x[None], len(codes), axis=0))
                actions = model(repeat(data['states'][i]), repeat(data['base_z']),
                                torch.as_tensor(codes), repeat(context)).numpy()[:, 0, :3]
                records[name].append((float(target.std(axis=0).mean()),
                                      float(actions.std(axis=0).mean())))
    report = {'scope': __doc__, 'checkpoint_sha256': digest_file(args.checkpoint),
              'base_sha256': digest_file(args.base), 'source_sha256': digest_file(__file__),
              'rollouts': len(paths), 'private_codes': codes[:, 0].tolist(),
              'first_control_translation_only': True, 'native_rollouts_during_audit': 0,
              'joint_data_or_selection': False, 'bins': {}}
    for name, rows in records.items():
        if not rows:
            report['bins'][name] = {'states': 0}
            continue
        values = np.array(rows)
        report['bins'][name] = {'states': len(rows),
            'teacher_mean_code_std': float(values[:, 0].mean()),
            'generator_mean_code_std': float(values[:, 1].mean()),
            'teacher_fraction_code_independent': float((values[:, 0] < 1e-6).mean()),
            'generator_fraction_code_independent': float((values[:, 1] < 1e-6).mean())}
    out = Path(args.out)
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
