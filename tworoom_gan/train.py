"""BicycleGAN training: full-episode VAE reconstruction + Gaussian latent cycle.

The latent cycle follows the proposal's teacher-state construction: generate
actions at an episode's observed states using ONE sampled latent, then encode
the resulting state/action sequence. Closed-loop latent recovery is measured
separately in evaluation to expose any teacher-state / rollout discrepancy.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F

from .data import EpisodeDataset
from .env import TASK, HORIZON, expert_chunk, observation, transition
from .models import ModelConfig, Generator, TrajectoryEncoder, Discriminator, reparameterize, kl_loss, masked_l1, gaussian_prior_match
from .evaluate import evaluate, offline_metrics
from .rollout_training import differentiable_rollout, collect_feedback_states
from .style import encode_style, decode_style


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="artifacts/data")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--cycle-batch", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-pairs", type=int, default=64)
    p.add_argument("--eval-latents", type=int, default=8)
    p.add_argument("--min-best-step", type=int, default=0,
                   help="Earliest selectable step; 0 means warmup+500 (capped at total steps)")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rec-weight", type=float, default=10.0)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=0.01)
    p.add_argument("--adv-weight", type=float, default=0.05)
    p.add_argument("--smooth-weight", type=float, default=0.01)
    p.add_argument("--mode-weight", type=float, default=0.0)
    p.add_argument("--augment-pixels", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--z-dim", type=int, default=2)
    p.add_argument("--encoder-moments", action="store_true")
    p.add_argument("--canonical", action="store_true")
    p.add_argument("--route-feature", action="store_true")
    p.add_argument("--style-weight", type=float, default=0.0)
    p.add_argument("--prior-rec-weight", type=float, default=0.0)
    p.add_argument("--style-warmup", type=int, default=1000)
    p.add_argument("--prior-match-weight", type=float, default=0.0)
    p.add_argument("--rollout-cycle-weight", type=float, default=0.0)
    p.add_argument("--rollout-cycle-every", type=int, default=8)
    p.add_argument("--rollout-cycle-start", type=int, default=1500)
    p.add_argument("--feedback-weight", type=float, default=0.0)
    p.add_argument("--feedback-every", type=int, default=8)
    p.add_argument("--feedback-start", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume")
    return p


def main():
    args = parser().parse_args()
    run, device = Path(args.run), args.device
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = ModelConfig(z_dim=args.z_dim, hidden=args.hidden, encoder_moments=args.encoder_moments,
                         canonical=args.canonical, route_feature=args.route_feature,
                         semantic_style=args.style_weight > 0 or args.prior_rec_weight > 0)
    if args.style_weight > 0 or args.prior_rec_weight > 0:
        assert args.z_dim == 2, "Auxiliary style coordinates require z_dim=2 and dataset v2"
        assert "offset" in EpisodeDataset(Path(args.data) / "train.npz", "cpu").tensors, "Use the offset-varying dataset v2"
    generator = Generator(config).to(device)
    encoder = TrajectoryEncoder(config).to(device)
    discriminator = Discriminator(config).to(device)
    ema = deepcopy(generator).eval().requires_grad_(False)
    opt_ge = torch.optim.Adam(list(generator.parameters()) + list(encoder.parameters()), lr=args.lr, betas=(0.5, 0.99))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.99))
    train_data = EpisodeDataset(Path(args.data) / "train.npz", device)
    val_path = Path(args.data) / "val.npz"
    start_step, best_score = 0, -1.0
    if args.resume:
        previous = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(previous["generator"])
        encoder.load_state_dict(previous["encoder"])
        discriminator.load_state_dict(previous["discriminator"])
        ema.load_state_dict(previous["generator_ema"])
        opt_ge.load_state_dict(previous["optimizer_ge"])
        opt_d.load_state_dict(previous["optimizer_d"])
        for group in opt_ge.param_groups:
            group["lr"] = args.lr
        for group in opt_d.param_groups:
            group["lr"] = args.lr * 0.5
        start_step = previous["step"]
        # A new run/configuration selects its best model independently.
        if str(Path(args.resume).parent) == str(run):
            best_score = previous.get("best_score", -1.0)
            if "torch_rng" in previous:
                torch.set_rng_state(previous["torch_rng"].cpu())
            if device.startswith("cuda") and previous.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in previous["cuda_rng"]])
    info = {"args": vars(args), "model": asdict(config),
            "parameter_counts": {"generator": sum(p.numel() for p in generator.parameters()),
                                 "encoder": sum(p.numel() for p in encoder.parameters()),
                                 "discriminator": sum(p.numel() for p in discriminator.parameters())},
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu"}
    (run / "config.json").write_text(json.dumps(info, indent=2))
    print("CONFIG", json.dumps(info), flush=True)
    started = time.time()
    running = {}
    with (run / "train.jsonl").open("a", buffering=1) as log:
        for step in range(start_step + 1, args.steps + 1):
            generator.train()
            encoder.train()
            discriminator.train()
            b = train_data.sample(args.batch, args.chunks)
            mu, logvar = encoder(b["episode_states"], b["episode_actions"], b["lengths"])
            z_encoded = reparameterize(mu, logvar, min(1.0, step / 500))
            style_aux = torch.zeros((), device=device)
            if args.style_weight > 0:
                style_target = encode_style(b["mode"], b["speed"], b["offset"])
                style_aux = F.l1_loss(mu, style_target)
                mix = min(1.0, step / max(args.style_warmup, 1))
                z_encoded = (1 - mix) * style_target + mix * z_encoded
            state, target, mask = b["states"], b["actions"], b["action_mask"]
            # Expert relabeling around demonstrated states teaches feedback recovery.
            if args.augment_pixels > 0:
                with torch.no_grad():
                    pos, goal = (state[..., :2] + 1) * 112, (state[..., 2:4] + 1) * 112
                    jitter = torch.randn_like(pos) * args.augment_pixels / TASK.speed
                    perturbed, _ = transition(pos, jitter)
                    augment = torch.rand(state.shape[:-1], device=device) < 0.5
                    pos = torch.where(augment[..., None], perturbed, pos)
                    state = observation(pos, goal)
                    mode = b["mode"][:, None].expand(-1, args.chunks)
                    speed = b["speed"][:, None].expand_as(mode)
                    offset = b["offset"][:, None].expand_as(mode)
                    relabeled = expert_chunk(pos, goal, mode, speed, offset=offset)
                    target = torch.where(augment[..., None, None], relabeled, target)
            z_per_chunk = z_encoded[:, None].expand(-1, args.chunks, -1)
            reconstructed = generator(state, z_per_chunk)
            prior_z = torch.randn_like(z_encoded)
            random_actions = generator(state, prior_z[:, None].expand_as(z_per_chunk))
            gan_scale = args.adv_weight * min(1.0, max(0.0, (step - args.warmup) / 500))
            d_loss = torch.zeros((), device=device)
            r1 = torch.zeros((), device=device)
            if gan_scale > 0:
                discriminator.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                flat_state = state.detach().flatten(0, 1)
                flat_mask = mask.flatten(0, 1)
                real = target.detach().flatten(0, 1).requires_grad_(step % 16 == 0)
                real_logits = discriminator(flat_state, real, flat_mask)
                fakes = torch.cat([reconstructed.detach().flatten(0, 1), random_actions.detach().flatten(0, 1)])
                fake_logits = discriminator(flat_state.repeat(2, 1), fakes, flat_mask.repeat(2, 1))
                d_loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
                if step % 16 == 0:
                    grad = torch.autograd.grad(real_logits.sum(), real, create_graph=True)[0]
                    r1 = grad.square().flatten(1).sum(1).mean()
                    d_loss = d_loss + r1 * 8.0  # lazy R1: 0.5 * gamma(1) * interval(16)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5)
                opt_d.step()
            discriminator.requires_grad_(False)
            opt_ge.zero_grad(set_to_none=True)
            rec = masked_l1(reconstructed, target, mask)
            kl = kl_loss(mu, logvar)
            # Same random latent is applied to every state in this episode.
            cb = min(args.cycle_batch, args.batch)
            episode_states = b["episode_states"][:cb]
            cycle_z = torch.randn((cb, config.z_dim), device=device)
            synthetic = generator(episode_states, cycle_z[:, None].expand(-1, episode_states.shape[1], -1))[:, :, 0]
            recovered, _ = encoder(episode_states, synthetic, b["lengths"][:cb])
            lat = F.l1_loss(recovered, cycle_z)
            prior_rec = torch.zeros((), device=device)
            if args.prior_rec_weight > 0:
                with torch.no_grad():
                    pm, ps, po = decode_style(prior_z)
                    ppos, pgoal = (state[..., :2] + 1) * 112, (state[..., 2:4] + 1) * 112
                    pm, ps, po = [v[:, None].expand(-1, args.chunks) for v in [pm, ps, po]]
                    prior_targets = expert_chunk(ppos, pgoal, pm, ps, offset=po)
                prior_rec = F.l1_loss(random_actions, prior_targets)
            adv = torch.zeros((), device=device)
            if gan_scale > 0:
                all_fake = torch.cat([reconstructed.flatten(0, 1), random_actions.flatten(0, 1)])
                adv = -discriminator(state.flatten(0, 1).repeat(2, 1), all_fake, mask.flatten(0, 1).repeat(2, 1)).mean()
            smooth = (reconstructed[..., 1:, :] - reconstructed[..., :-1, :]).square().mean()
            mode_aux = F.softplus(-(b["mode"].float() * 2 - 1) * mu[:, 0]).mean()
            prior_match = gaussian_prior_match(mu) if args.prior_match_weight > 0 else torch.zeros((), device=device)
            rollout_lat = torch.zeros((), device=device)
            feedback_rec = torch.zeros((), device=device)
            if args.rollout_cycle_weight > 0 and step >= args.rollout_cycle_start and step % args.rollout_cycle_every == 0:
                count = min(16, args.batch)
                starts = (b["episode_states"][:count, 0, :2] + 1) * 112
                goals = (b["episode_states"][:count, 0, 2:4] + 1) * 112
                rz = torch.randn((count, config.z_dim), device=device)
                rs, ra, rl = differentiable_rollout(generator, starts, goals, rz)
                recovered_rollout, _ = encoder(rs, ra, rl)
                rollout_lat = F.l1_loss(recovered_rollout, rz)
            if args.feedback_weight > 0 and step >= args.feedback_start and step % args.feedback_every == 0:
                feedback_states, feedback_z, feedback_targets = collect_feedback_states(generator, encoder, b)
                feedback_pred = generator(feedback_states, feedback_z)
                feedback_rec = F.l1_loss(feedback_pred, feedback_targets)
            loss = args.rec_weight * rec + args.kl_weight * kl + args.latent_weight * lat + gan_scale * adv + args.smooth_weight * smooth + args.mode_weight * mode_aux + args.rollout_cycle_weight * rollout_lat + args.feedback_weight * feedback_rec + args.style_weight * style_aux + args.prior_rec_weight * prior_rec + args.prior_match_weight * prior_match
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(generator.parameters()) + list(encoder.parameters()), 5)
            opt_ge.step()
            with torch.no_grad():
                decay = min(0.995, (step + 1) / (step + 10))
                for p_ema, p in zip(ema.parameters(), generator.parameters()):
                    p_ema.lerp_(p, 1 - decay)
            values = {"loss": loss, "rec": rec, "kl": kl, "latent": lat, "adv": adv, "d_loss": d_loss, "r1": r1, "smooth": smooth, "grad_norm": grad_norm, "rollout_latent": rollout_lat, "feedback_rec": feedback_rec, "style_aux": style_aux, "prior_rec": prior_rec, "prior_match": prior_match}
            for k, v in values.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            if step % 100 == 0 or step == args.steps:
                row = {"step": step, "elapsed_seconds": time.time() - started, "gan_weight": gan_scale,
                       **{k: v / (100 if step % 100 == 0 else step % 100) for k, v in running.items()}}
                log.write(json.dumps(row) + "\n")
                print("TRAIN", json.dumps(row), flush=True)
                running = {}
            if step % args.eval_every == 0 or step == args.steps:
                metrics, _ = evaluate(ema, encoder, val_path, args.eval_pairs, args.eval_latents, device=device)
                metrics.update(offline_metrics(ema, encoder, val_path, device, batches=2))
                score = metrics["success_rate"] + 0.2 * metrics["both_modes_covered_fraction"]
                eligible = step >= min(args.steps, args.min_best_step or args.warmup + 500)
                improved = eligible and score > best_score
                if eligible:
                    best_score = max(best_score, score)
                row = {"step": step, "elapsed_seconds": time.time() - started, "score": score, "eligible_for_best": eligible, **metrics}
                with (run / "validation.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print("VALIDATION", json.dumps(row), flush=True)
                checkpoint = {"step": step, "model_config": asdict(config), "train_config": vars(args),
                              "task": asdict(TASK), "normalization": "pixel / 112 - 1", "execute_actions": 8,
                              "generator": generator.state_dict(), "generator_ema": ema.state_dict(),
                              "encoder": encoder.state_dict(), "discriminator": discriminator.state_dict(),
                              "optimizer_ge": opt_ge.state_dict(), "optimizer_d": opt_d.state_dict(),
                              "best_score": best_score, "validation": metrics,
                              "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None}
                temp = run / "checkpoint.tmp"
                torch.save(checkpoint, temp)
                temp.replace(run / "latest.pt")
                if improved:
                    torch.save(checkpoint, run / "best.tmp")
                    (run / "best.tmp").replace(run / "best.pt")
                # Keep compact per-evaluation inference snapshots for auditing and
                # selecting checkpoints after all adversarial objectives are active.
                snapshots = run / "snapshots"
                snapshots.mkdir(exist_ok=True)
                compact = {k: v for k, v in checkpoint.items() if not k.startswith("optimizer") and k not in ["torch_rng", "cuda_rng", "discriminator"]}
                torch.save(compact, snapshots / f"step_{step:06d}.pt")
    (run / "DONE.json").write_text(json.dumps({"step": args.steps, "best_score": best_score, "elapsed_seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
