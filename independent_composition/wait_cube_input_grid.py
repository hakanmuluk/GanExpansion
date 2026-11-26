"""Wait for native same-state collections and verify matched feedback tensors."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .collect_cube_input_grid_feedback import KEYS
from .counterfactual_action_loss import build_pairs
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); output = Path(a.out); assert not output.exists()
    receipt = json.loads(Path(a.receipt).read_text()); locations = {}
    for row in receipt['launches']:
        command = row['command']; locations[row['skill']] = Path(command[command.index('--out') + 1])
    while True:
        pending = []
        for row in receipt['launches']:
            folder = locations[row['skill']]
            if (folder / 'DONE.json').exists():
                continue
            proc = Path('/proc') / str(row['pid']) / 'cmdline'
            assert proc.exists() and proc.read_bytes(), ('Collector stopped before DONE; inspect log', folder)
            command = proc.read_bytes().replace(b'\0', b' ').decode()
            assert 'collect_cube_input_grid_feedback' in command and str(folder) in command
            pending.append(row['skill'])
        if not pending:
            break
        time.sleep(5)
    result = {'matched_original_feedback_verified': True, 'roles': {}, 'raw_arrays_remain_remote': True}
    for skill, folder in locations.items():
        summary = json.loads((folder / 'summary.json').read_text())
        config = json.loads((folder / 'config.json').read_text())
        assert config['source_sha256'] == receipt['source_sha256']
        assert summary['actor_unchanged'] and summary['no_joint_labels']
        assert max(summary['maximum_state_replay_error'], summary['maximum_context_replay_error']) < 2e-6
        n = summary['new_matched_rows_per_arm']; assert n >= 1000
        paths = {f: folder / f / 'feedback.npz' for f in ['grid', 'applied']}
        with np.load(paths['grid']) as g, np.load(paths['applied']) as c:
            count = len(g['states']); assert count == len(c['states']) == summary['total_rows_per_arm']
            expected = {'states': (count, 47), 'contexts': (count, 20), 'actions': (count, 16, 5),
                'mask': (count, 16), 'base_z': (count, 2), 'new_z': (count, 1)}
            for k in KEYS:
                assert g[k].shape == c[k].shape == expected[k]
                assert g[k].dtype == c[k].dtype == np.float32
                assert np.isfinite(g[k]).all() and np.isfinite(c[k]).all()
                assert np.array_equal(g[k][n:], c[k][n:]), ('Different retained old feedback', skill, k)
            for k in ['states', 'contexts', 'mask']:
                assert np.array_equal(g[k], c[k]), (skill, k)
            assert np.all(g['mask'][:n] == 1)
            assert np.array_equal(g['base_z'][:, 1], c['base_z'][:, 1])
            assert np.array_equal(np.sign(g['base_z'][:, 0]), np.sign(c['base_z'][:, 0]))
            pairs = {}
            if skill == 'keepout':
                for family, data in [('grid', g), ('applied', c)]:
                    values = {k: torch.from_numpy(data[k].copy()) for k in KEYS}
                    index, description = build_pairs(values)
                    assert len(index) >= 128
                    pairs[family] = description
                    del values, index
        result['roles'][skill] = {'new_matched_rows_per_arm': n, 'total_rows_per_arm': count,
            'source_observation_parity_pass': True, 'states_contexts_masks_equal': True,
            'base_sign_speed_preserved': True, 'native_targets_full16': True,
            'feedback_sha256': {f: digest_file(path) for f, path in paths.items()},
            'private_code_pair_groups': pairs}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
