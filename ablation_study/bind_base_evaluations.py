"""Bind completed base scores to selected weights using saved first commands.

The historical original evaluator recorded paths, not checkpoint digests. This
retrospective check compares the first executed action chunk of EVERY original
Gaussian test episode with the currently selected checkpoint. It neither runs
the environment nor changes scores. Raw trajectories remain on the host.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .export_completed import state_hash


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    assert not out.exists(), 'Use a fresh receipt'
    torch.set_num_threads(1)
    checks = []
    for env, seeds in [('tworoom', [42, 123]), ('cube', [42, 43])]:
        for method in ['full', 'adversarial']:
            for seed in seeds:
                case = f'{env}_'+('latest_' if env == 'cube' else '')+f'base_{method}_s{seed}'
                folder = Path('ablation_results/evaluation')/f'original_gaussian_{env}_{method}_s{seed}'
                assert (folder/'DONE.json').exists()
                summary = json.loads((folder/'study_summary.json').read_text())
                cfg = summary['original_cli_configuration']
                checkpoint = Path('ablation_results/training')/case/('snapshots/step_005000.pt' if env == 'tworoom' else 'best.pt')
                assert Path(cfg['checkpoint']) == checkpoint
                checkpoint_sha = sha(checkpoint)
                archive = folder/('independent_rollouts.npz' if env == 'tworoom' else 'rollouts.npz')
                with np.load(archive) as data:
                    state = np.ascontiguousarray(data['states'][:, 0])
                    z = np.ascontiguousarray(data['z'])
                    actions = data['actions'][:, :8].copy()
                    lengths = data['lengths'].copy()
                count = 8192 if env == 'tworoom' else 1200
                assert len(state) == len(z) == len(actions) == count
                assert len(lengths) == count and np.all(lengths > 0)
                if env == 'tworoom':
                    from tworoom_gan.evaluate import load_checkpoint
                    # CPU-only audit avoids contending with live GPU trainers.
                    # Float32 cross-device differences use fixed atol1e-5.
                    device = 'cpu'
                    model, _, ck = load_checkpoint(checkpoint, device, use_ema=not cfg.get('raw', False))
                    with torch.inference_mode():
                        predicted = model(torch.from_numpy(state).to(device), torch.from_numpy(z).to(device))[:, :8].cpu().numpy()
                else:
                    from cube.cube_gan.evaluate import load_models
                    model, _, ck = load_models(checkpoint, 'cpu', ema=not cfg.get('raw', False))
                    # Original native workers predict one episode at a time.
                    predicted = np.empty_like(actions)
                    with torch.inference_mode():
                        for i in range(count):
                            predicted[i] = model(torch.from_numpy(state[i:i+1]), torch.from_numpy(z[i:i+1]))[0, :8].numpy()
                    predicted[..., 4] = np.where(predicted[..., 4] >= 0, 1, -1)
                    device = 'cpu'
                valid = np.arange(8)[None] < lengths[:, None]
                error = np.abs(predicted-actions)[valid]
                assert np.isfinite(error).all()
                maximum = float(error.max())
                assert maximum <= 1e-5, (case, maximum)
                assert sha(checkpoint) == checkpoint_sha, 'Checkpoint changed during binding'
                slot = 'generator' if cfg.get('raw', False) or 'generator_ema' not in ck else 'generator_ema'
                row = {'case': case, 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
                       'generator_slot': slot, 'generator_tensor_sha256': state_hash(ck[slot]),
                       'evaluation_summary_sha256': sha(folder/'study_summary.json'),
                       'raw_rollout_archive_sha256': sha(archive), 'episodes_checked': count,
                       'executed_commands_checked': int(valid.sum()), 'device': device,
                       'original_evaluation_device': cfg.get('device', 'cpu'),
                       'max_abs_error': maximum, 'absolute_tolerance': 1e-5,
                       'bitwise_equal': bool(np.array_equal(predicted[valid], actions[valid])),
                       'passed': True}
                checks.append(row)
                print(json.dumps(row), flush=True)
                del model, ck, predicted, actions
    result = {'complete': True, 'models_checked': 8, 'checks': checks,
              'scope': 'Retrospective checkpoint binding: first executed eight-command chunk, masked at termination, for all37568original Gaussian test episodes. Not a rerun or validation of later commands; no scores changed.',
              'raw_episode_data_exported': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
