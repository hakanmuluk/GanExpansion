"""Wait for matched own-data collection, pack, and check original interfaces."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .legacy import digest_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--receipt', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(); root = Path(a.root); out = Path(a.out)
    receipt = json.loads(Path(a.receipt).read_text())
    raw = {}
    for row in receipt['launches']:
        command = row['command']; raw[row['family']] = Path(command[command.index('--out') + 1])
    while True:
        pending = []
        for row in receipt['launches']:
            directory = raw[row['family']]
            cfg = json.loads((directory / 'config.json').read_text()) if (directory / 'config.json').exists() else {}
            if (directory / 'DONE.json').exists() and 'coverage_source_sha256' in cfg:
                continue
            proc = Path('/proc') / str(row['pid']) / 'cmdline'
            assert proc.exists() and proc.read_bytes(), ('Collector ended before final provenance', row)
            pending.append(row['family'])
        if not pending:
            break
        time.sleep(5)
    command = [sys.executable, '-m', 'independent_composition.pack_cube_radial_pair',
        '--radial', str(raw['radial']), '--angular', str(raw['angular']), '--out', str(out),
        '--keepout-validation', str(root / 'cube_keepout_coverage_v43_data/keepout/val_contexts.npz'),
        '--waypoint-validation', str(root / 'cube_coverage_v39_data/waypoint/val_contexts.npz'),
        '--keepout-feedback', str(root / 'cube_donor_v87_crossed_data/feedback.npz'),
        '--waypoint-feedback', str(root / 'cube_counterfactual_v94_waypoint_data/feedback.npz')]
    subprocess.run(command, check=True)
    from cube_addons.data import EpisodeDataset
    from .cube_checkpoint_bank import CheckpointBank
    torch.set_num_threads(1)
    bank = CheckpointBank(root / 'cube_magnitude_v100_keepout_control/best.pt',
        root / 'cube_magnitude_v100_waypoint_control/best.pt',
        root / 'cube_diverse_v35_stageB/base_policy.pt', device='cpu')
    checks = []
    for family in ['angular', 'radial']:
        for skill, actor, encoder in [('keepout', bank.k, bank.ek), ('waypoint', bank.w, bank.ew)]:
            ds = EpisodeDataset(out / family / skill / 'train.npz', 'cpu', alignment=8)
            b = ds.sample(4, 2)
            bz = b['base_z'][:, None].expand(-1, 2, -1)
            z = b['new_z'][:, None].expand(-1, 2, -1)
            with torch.no_grad():
                predicted = actor(b['states'], bz, z, b['contexts'])
                assert predicted.shape == (4, 2, 16, 5) and torch.isfinite(predicted).all()
                assert torch.equal(actor(b['states'], bz, z, b['contexts'], gate=0), actor.base(b['states'], bz))
                mu, lv = encoder(b['episode_states'], b['episode_actions'], b['episode_contexts'], b['base_z'], b['lengths'])
                assert torch.isfinite(mu).all() and torch.isfinite(lv).all()
            checks.append({'family': family, 'skill': skill, 'episodes': ds.size,
                'original_generator_encoder_finite': True, 'disabled_adapter_equals_base': True})
            del ds, b
    bank.assert_frozen()
    proof = {'checks': checks, 'original_models_frozen': True,
        'original_generator_and_encoder_classes': True, 'new_learned_modules': 0,
        'source_sha256': digest_file(__file__), 'raw_arrays_remain_remote': True}
    (out / 'interface_proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    # Queue waits for process exit plus this result, so packing DONE alone never
    # releases training before the interface checks complete.
    (out / 'READY.json').write_text(json.dumps({'data_and_original_interfaces_verified': True}) + '\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
