"""Aggregate diagnostic of conditional code/instruction use on own teacher data.

This is a training-support diagnostic, not a closed-loop success estimate or a
checkpoint-selection criterion. No native environment is constructed. Raw
teacher arrays and model predictions remain on the machine running the audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cube_addons.evaluate import load_adapter
from .legacy import digest_file


def groups(data, varied):
    fixed = ['states', 'base_z', 'contexts' if varied == 'new_z' else 'new_z']
    buckets = {}
    for i in range(len(data['states'])):
        key = tuple(np.round(np.concatenate([data[k][i] for k in fixed]), 5))
        buckets.setdefault(key, []).append(i)
    return [v for v in buckets.values() if len(v) >= 2
            and np.ptp(data[varied][v], axis=0).max() > .001]


def variation(values, indices, horizon):
    return float(np.mean([values[g, :horizon].std(axis=0).mean() for g in indices]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['feedback', 'checkpoint', 'base', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--horizon', type=int, default=8)
    args = p.parse_args()
    assert 1 <= args.horizon <= 16
    torch.set_num_threads(1)
    with np.load(args.feedback) as archive:
        data = {k: archive[k] for k in archive.files}
    code_groups, instruction_groups = groups(data, 'new_z'), groups(data, 'contexts')
    assert code_groups, 'Requires different-code queries at identical inputs'
    model, _, _ = load_adapter(args.checkpoint, args.base, 'cpu')
    model.eval().requires_grad_(False)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(data['states']), 256):
            batch = {k: torch.as_tensor(data[k][start:start + 256])
                     for k in ['states', 'base_z', 'new_z', 'contexts']}
            predictions.append(model(batch['states'], batch['base_z'],
                                     batch['new_z'], batch['contexts']).numpy())
    pred = np.concatenate(predictions)
    assert pred.shape == data['actions'].shape and np.isfinite(pred).all()
    report = {'scope': __doc__, 'checkpoint_sha256': digest_file(args.checkpoint),
              'base_sha256': digest_file(args.base),
              'feedback_sha256': digest_file(args.feedback),
              'source_sha256': digest_file(__file__),
              'samples': len(pred), 'horizon': args.horizon,
              'training_support_only': True, 'simulator_calls': 0,
              'used_for_checkpoint_selection': False, 'raw_data_transferred': False}
    mask = data['mask'][:, :args.horizon, None]
    report['action_rmse'] = float(np.sqrt(
        (((pred[:, :args.horizon] - data['actions'][:, :args.horizon]) ** 2) * mask).sum()
        / (mask.sum() * pred.shape[-1])))
    for name, indices in [('code', code_groups), ('instruction', instruction_groups)]:
        if not indices:
            report[name] = {'groups': 0, 'measured': False,
                            'reason': 'No paired instruction changes in this dataset'}
            continue
        report[name] = {'groups': len(indices),
                        'teacher_mean_action_std': variation(data['actions'], indices, args.horizon),
                        'generator_mean_action_std': variation(pred, indices, args.horizon),
                        'maximum_fixed_state_difference': max(float(np.ptp(
                            data['states'][g], axis=0).max()) for g in indices)}
    out = Path(args.out)
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
