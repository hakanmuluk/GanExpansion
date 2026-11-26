"""Aggregate disagreement on actual observed states; no simulation or planning.

Checks whether agreement on two controls persists through the eight controls
actually executed. This diagnostic never selects controls or changes a policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .cube_checkpoint_bank import CheckpointBank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'base', 'keepout', 'waypoint', 'out']:
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    bank = CheckpointBank(args.keepout, args.waypoint, args.base, 'cpu')
    folder = Path(args.run)
    plans = json.loads((folder / 'planning.json').read_text())
    records = json.loads((folder / 'records.json').read_text())
    result = {'scope': __doc__, 'groups': {}}
    values = {}
    for row, episode in zip(records, plans):
        with np.load(folder / ('rollout_%04d.npz' % row['context'])) as data:
            selected = [plan for plan in episode if 'selected_codes' in plan]
            times = [plan['step'] for plan in selected]
            if not times:
                continue
            state = torch.as_tensor(data['states'][times], dtype=torch.float32)
            ctx = torch.as_tensor(data['contexts'][times], dtype=torch.float32)
            bz = torch.as_tensor(data['base_z'], dtype=torch.float32)[None].expand(len(times), -1)
            z = torch.tensor([plan['selected_codes'][0] for plan in selected])
            with torch.no_grad():
                k, w = bank.pair(state, bz, z, (ctx[:, 0], ctx[:, 1]))
            square = (k[:, :, :3] - w[:, :, :3]).square()
            carry = (ctx[:, 0, 13] > .5) & (ctx[:, 0, 14] > .5)
            label = 'goal_succeeded' if row['goal_success'] else 'goal_failed'
            for phase, mask in [('all', torch.ones_like(carry)), ('held_transport', carry)]:
                if not mask.any():
                    continue
                entry = values.setdefault(label + '_' + phase, [])
                for i in torch.nonzero(mask).flatten().tolist():
                    entry.append([float(square[i, :2].mean()), float(square[i, :8].mean()),
                                  float((k[i, :8, 4].sign() != w[i, :8, 4].sign()).float().mean())])
    for name, rows in values.items():
        x = np.array(rows)
        result['groups'][name] = {'plans': len(x), 'xyz_rmse_first2': float(np.sqrt(x[:, 0].mean())),
            'xyz_rmse_first8': float(np.sqrt(x[:, 1].mean())), 'gripper_sign_disagreement_first8': float(x[:, 2].mean())}
    bank.assert_frozen()
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
