"""Calibrate the ORIGINAL private SkillEncoder for static observed-state chunks.

The existing encoder consumes a sequence interface. Here the SAME current state
and own context are repeated for every action token, both during calibration and
evaluation. These are not predicted future states. No new neural class, hidden
width, input feature, generator update, joint scene or simulator is introduced.
Whole physical contexts are held out. This is a calibration study, not evidence
that a cycle penalty improves composition.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from cube_addons.evaluate import load_adapter
from cube_addons.models import state_digest
from .legacy import digest_file
from .windows_data import Windows


def encode_chunk(encoder, state, context, base_z, actions, mask=None):
    n, horizon = actions.shape[:2]
    lengths = (torch.full((n,), horizon, device=state.device, dtype=torch.long)
               if mask is None else mask.sum(1).long().clamp_min(1))
    return encoder(state[:, None].expand(-1, horizon, -1), actions,
                   context[:, None].expand(-1, horizon, -1), base_z, lengths)


def nll(mu, lv, code):
    return .5 * ((mu - code).square() * (-lv).exp() + lv).mean()


@torch.no_grad()
def diagnostics(encoder, batch):
    s, c, bz, a, mask, z = [batch[k] for k in
        ['state', 'context', 'base_z', 'actions', 'mask', 'code']]
    mu, lv = encode_chunk(encoder, s, c, bz, a, mask)
    # Every condition is identical for the shuffle control; only actions change.
    shuffled, _ = encode_chunk(encoder, s, c, bz, a.roll(1, 0), mask)
    active = c[:, 14] > .5
    result = {'n': len(s), 'rmse': float((mu-z).square().mean().sqrt()),
        'nll': float(nll(mu, lv, z)),
        'prior_zero_rmse': float(z.square().mean().sqrt()),
        'shuffled_action_rmse': float((shuffled-z).square().mean().sqrt()),
        'sign_accuracy': float((mu.sign()==z.sign()).float().mean()),
        'posterior_std': float((.5*lv).exp().mean()),
        'within_2std': float(((mu-z).abs() <= 2*(.5*lv).exp()).float().mean())}
    if active.any():
        result.update(active_n=int(active.sum()),
            active_rmse=float((mu[active]-z[active]).square().mean().sqrt()),
            active_shuffled_rmse=float((shuffled[active]-z[active]).square().mean().sqrt()),
            active_sign_accuracy=float((mu[active].sign()==z[active].sign()).float().mean()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'base', 'data', 'out']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--seed', type=int, default=2601001)
    p.add_argument('--counterfactual-base-fraction', type=float, default=0.,
        help='Fraction of generated-only examples with independent existing base codes')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    assert 0 <= args.counterfactual_base_fraction <= 1
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    actor, encoder, meta = load_adapter(args.checkpoint, args.base, args.device)
    actor.eval().requires_grad_(False)
    encoder.requires_grad_(True)
    fingerprint = state_digest(actor)
    initial_encoder = state_digest(encoder)
    train = Windows(args.data, args.device, split='train')
    valid = Windows(args.data, args.device, split='validation')
    assert not train.context_fingerprints & valid.context_fingerprints
    rng = torch.Generator(device=args.device).manual_seed(2601002)
    real = valid.sample(2048, rng, alignment=8)
    fake = {k: v.clone() for k, v in real.items()}
    fake['code'] = torch.randn(fake['code'].shape, device=args.device, generator=rng)
    count = int(len(fake['code'])*args.counterfactual_base_fraction)
    if count:
        fake['base_z'][:count] = torch.randn((count, 2), device=args.device, generator=rng)
    with torch.no_grad():
        fake['actions'] = actor(fake['state'], fake['base_z'], fake['code'], fake['context'])
        fake['mask'] = torch.ones_like(fake['mask'])
    config = {'args': vars(args), 'scope': __doc__,
        'checkpoint_sha256': digest_file(args.checkpoint), 'base_sha256': digest_file(args.base),
        'data_sha256': digest_file(args.data), 'original_encoder_class': type(encoder).__name__,
        'encoder_shapes': {k: list(v.shape) for k, v in encoder.state_dict().items()},
        'train_contexts': len(train.context_ids), 'heldout_contexts': len(valid.context_ids),
        'loss': '.5*NLL(teacher code | own real chunk) + .5*NLL(fresh prior code | frozen generator chunk)',
        'generated_base_code_augmentation_fraction': args.counterfactual_base_fraction,
        'normalization': 'Original encoder buffers copied unchanged',
        'joint_data_used': False, 'new_learned_modules': 0,
        'source_sha256': digest_file(__file__)}
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    encoder.eval()
    (out/'initial_diagnostics.json').write_text(json.dumps(
        {'real': diagnostics(encoder, real), 'generated': diagnostics(encoder, fake)}, indent=2)+'\n')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    best = float('inf')
    started = time.time()
    with (out/'train.jsonl').open('w', buffering=1) as log:
        for step in range(1, args.steps+1):
            encoder.train()
            b = train.sample(args.batch, alignment=8)
            with torch.no_grad():
                code = torch.randn_like(b['code'])
                generated_base = b['base_z'].clone()
                count = int(len(code)*args.counterfactual_base_fraction)
                if count:
                    generated_base[:count] = torch.randn_like(generated_base[:count])
                actions = actor(b['state'], generated_base, code, b['context'])
            mu, lv = encode_chunk(encoder, b['state'], b['context'], b['base_z'], b['actions'], b['mask'])
            gm, gl = encode_chunk(encoder, b['state'], b['context'], generated_base, actions)
            loss = .5*nll(mu, lv, b['code']) + .5*nll(gm, gl, code)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.)
            optimizer.step()
            if step % 250 == 0 or step == args.steps:
                encoder.eval()
                metrics = {'step': step, 'seconds': time.time()-started,
                    'real': diagnostics(encoder, real), 'generated': diagnostics(encoder, fake)}
                score = .5*(metrics['real']['nll']+metrics['generated']['nll'])
                log.write(json.dumps(metrics)+'\n')
                print(json.dumps(metrics), flush=True)
                saved = {'encoder': encoder.state_dict(), 'config': config, 'metrics': metrics,
                    'step': step, 'encoder_class': 'SkillEncoder',
                    'calibration_interface': 'static_current_observation_action_chunk'}
                torch.save(saved, out/'latest.pt')
                if score < best:
                    best = score
                    torch.save(saved, out/'best.pt')
    assert state_digest(actor) == fingerprint
    assert state_digest(encoder) != initial_encoder
    assert all(not v.requires_grad and v.grad is None for v in actor.parameters())
    assert digest_file(args.checkpoint) == config['checkpoint_sha256']
    assert digest_file(args.base) == config['base_sha256']
    assert digest_file(__file__) == config['source_sha256']
    (out/'DONE.json').write_text(json.dumps({'steps': args.steps, 'best_validation_nll': best,
        'generator_unchanged': True, 'only_original_private_encoder_retrained': True,
        'architecture_unchanged': True, 'joint_data_used': False,
        'composition_not_yet_verified': True})+'\n')


if __name__ == '__main__':
    main()
