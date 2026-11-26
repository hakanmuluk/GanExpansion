"""Retrain the original Cube base, then transfer the original adapters to it.

No model classes, dimensions, forward interfaces or loss definitions change.
New files only. The old published checkpoints are immutable controls. Adapter
transfer initialization is explicit: its weights are copied, and the copied
artifact is associated with the newly trained base before further training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'source', 'out', 'base-data', 'stage-a']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--base-steps', type=int, default=6000)
    p.add_argument('--adapter-steps', type=int, default=6000)
    args = p.parse_args()
    root, source, out, data, stage_a = [Path(v).resolve() for v in
        [args.root, args.source, args.out, args.base_data, args.stage_a]]
    assert not out.exists()
    out.mkdir(parents=True)
    (out / 'logs').mkdir()
    env = {**os.environ, 'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': str(source) + ':' + str(source / 'cube')}
    base = root / 'cube/artifacts/final/base_policy.pt'
    disc = root / 'cube/artifacts/final/discriminator.pt'
    originals = [base, disc, *[root / f'cube_addon_artifacts/final/{s}_s42/adapter.pt'
                              for s in ['keepout', 'waypoint']]]
    state = {'pid': os.getpid(), 'args': vars(args), 'scope': __doc__, 'stages': [],
             'original_sha256': {str(f): sha(f) for f in originals}}

    def save():
        temp = out / 'status.tmp.json'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(out / 'status.json')

    def run(name, command):
        row = {'name': name, 'command': command, 'cwd': str(source),
               'started_unix': time.time(), 'status': 'running'}
        with (out / 'logs' / (name + '.log')).open('w') as log:
            proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = proc.pid
            state['stages'].append(row)
            save()
            code = proc.wait()
        row.update(returncode=code, status='complete' if code == 0 else 'failed',
                   finished_unix=time.time())
        save()
        if code:
            raise RuntimeError('Failed stage: ' + name + '; inspect saved log')

    state['waiting_for'] = str(data / 'DONE.json')
    save()
    while not (data / 'DONE.json').exists():
        status = json.loads((data / 'status.json').read_text())
        if not Path('/proc/' + str(status['pid']) + '/cmdline').exists():
            raise RuntimeError('Data collector exited before DONE; inspect its log')
        time.sleep(10)
    assert (stage_a / 'DONE.json').exists()
    state.pop('waiting_for')
    # The original evaluator also measures held-out reconstruction. Generate
    # its required val.npz from the same fresh, unfiltered validation bank.
    # It is never used by the training dataset or optimizer.
    if not (data / 'val.npz').exists():
        code = ('from concurrent.futures import ProcessPoolExecutor\n'
                'import multiprocessing as mp\nfrom pathlib import Path\n'
                'from cube_gan.data import initialize_worker,collect_context,context_specs,save_split\n'
                'if __name__ == "__main__":\n'
                ' with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context("spawn"),initializer=initialize_worker) as pool:\n'
                '  results=list(pool.map(collect_context,context_specs(64,2871,.5)))\n'
                '  stats,_=save_split(results,Path(' + repr(str(data)) + '),"val")\n'
                '  print(stats)\n')
        script = out / 'collect_heldout_base.py'
        script.write_text(code)
        run('collect_heldout_base', [sys.executable, str(script)])

    from cube.cube_gan.models import ModelConfig, Generator, TrajectoryEncoder, Discriminator
    old = torch.load(base, map_location='cpu', weights_only=False)
    cfg = ModelConfig(**old['model'])
    g, e, d = Generator(cfg), TrajectoryEncoder(cfg), Discriminator(cfg)
    g.load_state_dict(old['generator'])
    e.load_state_dict(old['encoder'])
    d.load_state_dict(torch.load(disc, map_location='cpu', weights_only=False)['discriminator'])
    opt_ge = torch.optim.Adam(list(g.parameters()) + list(e.parameters()), lr=1e-4, betas=(.5, .99))
    opt_d = torch.optim.Adam(d.parameters(), lr=5e-5, betas=(.5, .99))
    init = out / 'base_transfer_initialization.pt'
    torch.save({'model': asdict(cfg), 'generator': g.state_dict(), 'generator_ema': g.state_dict(),
                'encoder': e.state_dict(), 'discriminator': d.state_dict(),
                'optimizer_ge': opt_ge.state_dict(), 'optimizer_d': opt_d.state_dict(),
                'step': 0, 'best_score': -1}, init)
    state['initialization'] = {'path': str(init), 'sha256': sha(init),
        'generator_encoder': 'Exact original final weights; fresh Adam optimizers',
        'normalization': 'Original buffers retained on resume; same coordinates for adapter transfer',
        'losses': 'Original auxiliary base objective: rec10, known-rec10, style2, latent1, KL.01, GAN.05; no new loss'}
    save()
    run('original_base_control', [sys.executable, '-m', 'cube_gan.evaluate', '--checkpoint', str(base),
        '--data', str(data), '--split', 'val', '--contexts', '24', '--latents', '4',
        '--workers', '4', '--out', str(out / 'original_base_control')])
    base_run = out / 'train_base'
    run('train_base', [sys.executable, '-m', 'cube_gan.train', '--run', str(base_run),
        '--data', str(data), '--resume', str(init), '--steps', str(args.base_steps),
        '--batch', '64', '--chunks', '8', '--cycle-batch', '8', '--hidden', str(cfg.hidden),
        '--seed', '42', '--lr', '.0001', '--warmup', '500', '--rec-weight', '10',
        '--latent-weight', '1', '--kl-weight', '.01', '--adv-weight', '.05',
        '--style-weight', '2', '--known-rec-weight', '10', '--route-feature',
        '--eval-every', '1500', '--eval-contexts', '24', '--eval-latents', '4'])
    selected = torch.load(base_run / 'best.pt', map_location='cpu', weights_only=False)
    assert selected['model'] == old['model']
    new_base = out / 'base_policy.pt'
    torch.save({'model': selected['model'], 'generator': selected['generator_ema'],
                'encoder': selected['encoder'], 'step': selected['step'], 'execute': 8,
                'source_checkpoint': str(base_run / 'best.pt'),
                'original_initialization_sha256': sha(base)}, new_base)
    for skill in ['keepout', 'waypoint']:
        transfer = torch.load(stage_a / ('train_' + skill) / 'best.pt', map_location='cpu', weights_only=False)
        provenance = {'parent_adapter_sha256': sha(stage_a / ('train_' + skill) / 'best.pt'),
                      'parent_base_sha256': transfer['base_sha256'], 'new_base_sha256': sha(new_base),
                      'scope': 'Explicit transfer initialization, not a previously compatible trained adapter'}
        transfer['base_sha256'] = sha(new_base)
        transfer['transfer_initialization'] = provenance
        transferred = out / (skill + '_transfer_initialization.pt')
        torch.save(transfer, transferred)
        (out / (skill + '_transfer.json')).write_text(json.dumps(provenance, indent=2) + '\n')
        # Reuse the exact Stage A command and all loss/config flags.
        a_status = json.loads((stage_a / 'status.json').read_text())
        command = next(r['command'][:] for r in a_status['stages'] if r['name'] == 'train_' + skill)
        updates = {'--run': str(out / ('train_' + skill)), '--base': str(new_base),
                   '--discriminator-source': str(base_run / 'best.pt'), '--initialize': str(transferred),
                   '--steps': str(args.adapter_steps)}
        for flag, value in updates.items():
            command[command.index(flag) + 1] = value
        run('train_' + skill, command)
    assert all(sha(f) == checksum for f, checksum in state['original_sha256'].items())
    (out / 'DONE.json').write_text(json.dumps({'stage_B_complete': True, 'architecture_unchanged': True,
        'originals_preserved': True, 'base_sha256': sha(new_base),
        'composition_success_not_established': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()
