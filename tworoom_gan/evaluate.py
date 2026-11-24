"""Closed-loop 16-predict / 8-execute evaluation with fixed episode latents."""

import argparse
import json
from pathlib import Path
import math
import numpy as np
import torch

from .env import TASK, EXECUTE, observation, transition, make_reference, reset_reference
from .models import Generator, TrajectoryEncoder, ModelConfig, masked_l1
from .data import EpisodeDataset
from .style import decode_style


@torch.no_grad()
def rollout(generator, starts, goals, z, resample_latent=False, seed=0):
    generator.eval()
    n, device = len(starts), starts.device
    pos = starts.clone()
    alive = torch.ones(n, dtype=torch.bool, device=device)
    lengths = torch.full((n,), TASK.max_steps, dtype=torch.long, device=device)
    modes = torch.full((n,), -1, dtype=torch.long, device=device)
    collisions = torch.zeros(n, dtype=torch.long, device=device)
    positions = [pos.clone()]
    states, actions = [], []
    rng = torch.Generator(device=device).manual_seed(seed)
    episode_z = z.clone()
    for t in range(TASK.max_steps):
        state = observation(pos, goals)
        states.append(state)
        if t % EXECUTE == 0:
            if resample_latent and t > 0:
                z = torch.randn(z.shape, device=device, generator=rng)
            chunk = generator(state, z)
        a = torch.where(alive[:, None], chunk[:, t % EXECUTE], 0.0)
        actions.append(a)
        next_pos, hit = transition(pos, a)
        crossing = ((next_pos[:, 0] > 112) != (pos[:, 0] > 112)) & alive & (modes < 0)
        route = (next_pos[:, 1, None] - next_pos.new_tensor(TASK.door_y)).abs().argmin(-1)
        modes = torch.where(crossing, route, modes)
        collisions += hit & alive
        pos = next_pos
        success = (pos - goals).norm(dim=-1) < TASK.success_radius
        lengths = torch.where(alive & success, t + 1, lengths)
        alive &= ~success
        positions.append(pos.clone())
    return {"positions": torch.stack(positions, 1), "states": torch.stack(states, 1),
            "actions": torch.stack(actions, 1), "lengths": lengths, "success": ~alive,
            "mode": modes, "collisions": collisions, "z": episode_z,
            "start": starts, "goal": goals}


def wilson(successes, count):
    if not count:
        return [0.0, 1.0]
    p = successes / count
    d = 1 + 1.96 ** 2 / count
    c = (p + 1.96 ** 2 / (2 * count)) / d
    r = 1.96 * math.sqrt(p * (1 - p) / count + 1.96 ** 2 / (4 * count ** 2)) / d
    return [c - r, c + r]


@torch.no_grad()
def summarize_rollouts(result, pairs, latent_count, encoder=None, shared_latents=True):
    ok, mode = result["success"], result["mode"]
    good = ok & (mode >= 0)
    upper = (good & (mode == 0)).reshape(pairs, latent_count)
    lower = (good & (mode == 1)).reshape(pairs, latent_count)
    has_upper, has_lower = upper.any(-1), lower.any(-1)
    latent_valid = (upper | lower).sum(0)
    majority = torch.maximum(upper.sum(0), lower.sum(0))
    agreement = (majority[latent_valid > 0] / latent_valid[latent_valid > 0]).float()
    valid_steps = torch.arange(TASK.max_steps, device=ok.device)[None] < result["lengths"][:, None]
    mean_speeds = (result["actions"].norm(dim=-1) * valid_steps).sum(-1) / result["lengths"].clamp_min(1)
    result["mean_speed"] = mean_speeds
    positions = result["positions"][:, :-1]
    dx = (positions[..., 0] - 112).abs().sub(5).clamp_min(0)
    overlap = torch.zeros_like(valid_steps)
    for lo, hi in [(14, 35), (77, 147), (189, 210)]:
        dy = torch.maximum(lo - positions[..., 1], positions[..., 1] - hi).clamp_min(0)
        overlap |= dx.square() + dy.square() < (TASK.radius ** 2 - 1e-4)
    overlap &= valid_steps
    speeds_by_pair = mean_speeds.reshape(pairs, latent_count)
    ok_by_pair = ok.reshape(pairs, latent_count)
    speed_mean = (speeds_by_pair * ok_by_pair).sum(-1) / ok_by_pair.sum(-1).clamp_min(1)
    speed_variance = ((speeds_by_pair - speed_mean[:, None]).square() * ok_by_pair).sum(-1) / ok_by_pair.sum(-1).clamp_min(1)
    metric = {
        "rollouts": len(ok), "start_goal_pairs": pairs, "latents_per_pair": latent_count,
        "success_rate": float(ok.float().mean()),
        "success_95ci_wilson": wilson(int(ok.sum()), len(ok)),
        "both_modes_covered_fraction": float((has_upper & has_lower).float().mean()),
        "mean_mode_coverage": float((has_upper.float() + has_lower.float()).mean() / 2),
        "successful_mode_counts": [int(upper.sum()), int(lower.sum())],
        "same_latent_route_agreement_across_states": (float(agreement.mean()) if len(agreement) else 0.0) if shared_latents else None,
        "collision_episode_fraction": float((result["collisions"] > 0).float().mean()),
        "mean_collision_steps": float(result["collisions"].float().mean()),
        "strict_body_wall_overlap_episode_fraction": float(overlap.any(-1).float().mean()),
        "mean_successful_steps": float(result["lengths"][ok].float().mean()) if ok.any() else None,
        "mean_successful_action_magnitude": float(mean_speeds[ok].mean()) if ok.any() else None,
        "mean_within_task_speed_std": float(speed_variance.sqrt().mean()),
        "success_by_latent": ok.reshape(pairs, latent_count).float().mean(0).tolist(),
        "upper_route_fraction_by_latent": (upper.sum(0) / latent_valid.clamp_min(1)).tolist(),
    }
    # Shared latents and repeated start/goal configurations are correlated.
    # Bootstrap both grouping axes instead of treating every rollout as IID.
    arr = ok.reshape(pairs, latent_count).float().cpu().numpy()
    rng = np.random.default_rng(419)
    bootstrap = []
    for _ in range(1000):
        rows = rng.integers(pairs, size=pairs)
        if shared_latents and latent_count > 1:
            cols = rng.integers(latent_count, size=latent_count)
            bootstrap.append(arr[np.ix_(rows, cols)].mean())
        else:
            bootstrap.append(arr[rows].mean())
    metric["success_95ci_group_bootstrap"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    metric["shared_latents_across_pairs"] = shared_latents
    if encoder is not None:
        encoder.eval()
        recovered = []
        for i in range(0, len(ok), 128):
            mu, _ = encoder(result["states"][i:i+128], result["actions"][i:i+128], result["lengths"][i:i+128])
            recovered.append(mu)
        recovered = torch.cat(recovered)
        result["recovered_z"] = recovered
        error = recovered - result["z"]
        metric["rollout_latent_rmse_all"] = float(error.square().mean().sqrt())
        metric["rollout_latent_rmse_successful"] = float(error[ok].square().mean().sqrt()) if ok.any() else None
        correlations = []
        for j in range(result["z"].shape[1]):
            x, y = result["z"][ok, j], recovered[ok, j]
            if len(x) > 1 and x.std() > 1e-6 and y.std() > 1e-6:
                correlations.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
            else:
                correlations.append(None)
        metric["rollout_latent_correlations"] = correlations
        if encoder.config.semantic_style:
            intended_mode, intended_speed, intended_offset = decode_style(result["z"])
            crossed = (result["positions"][:, 1:, 0] > 112) != (result["positions"][:, :-1, 0] > 112)
            cross_t = crossed.long().argmax(1) + 1
            cross_y = result["positions"][torch.arange(len(ok), device=ok.device), cross_t, 1]
            observed_offset = cross_y - result["positions"].new_tensor(TASK.door_y)[mode.clamp_min(0)]
            metric["intended_route_accuracy_successful"] = float((mode[ok] == intended_mode[ok]).float().mean()) if ok.any() else None
            metric["crossing_offset_rmse_pixels_successful"] = float((observed_offset[good] - intended_offset[good]).square().mean().sqrt()) if good.any() else None
            metric["mean_speed_mae_successful"] = float((mean_speeds[ok] - intended_speed[ok]).abs().mean()) if ok.any() else None
            result["observed_crossing_offset"] = observed_offset
            result["intended_mode"] = intended_mode
            result["intended_speed"] = intended_speed
            result["intended_offset"] = intended_offset
    return metric


@torch.no_grad()
def evaluate(generator, encoder, data_path, pairs=64, latent_count=8, seed=7001, device="cuda", resample=False, independent_latents=False):
    data = np.load(data_path)
    pairs = min(pairs, len(data["start"]) // 4)
    starts = torch.tensor(data["start"][::4][:pairs], device=device)
    goals = torch.tensor(data["goal"][::4][:pairs], device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    if independent_latents:
        z = torch.randn((pairs * latent_count, generator.config.z_dim), device=device, generator=rng)
    else:
        shared_z = torch.randn((latent_count, generator.config.z_dim), device=device, generator=rng)
        z = shared_z.repeat(pairs, 1)
    result = rollout(generator, starts.repeat_interleave(latent_count, 0), goals.repeat_interleave(latent_count, 0), z,
                     resample_latent=resample, seed=seed + 1)
    metrics = summarize_rollouts(result, pairs, latent_count, encoder, shared_latents=not independent_latents and not resample)
    return metrics, result


@torch.no_grad()
def offline_metrics(generator, encoder, data_path, device="cuda", batches=4):
    data = EpisodeDataset(data_path, device)
    errors, cycle_errors, kl_values = [], [], []
    for _ in range(batches):
        b = data.sample(64, 4)
        mu, lv = encoder(b["episode_states"], b["episode_actions"], b["lengths"])
        z = mu[:, None].expand(-1, 4, -1)
        pred = generator(b["states"], z)
        errors.append(float(masked_l1(pred, b["actions"], b["action_mask"])))
        # Same teacher-state latent cycle as training, reported separately from actual rollout recovery.
        random_z = torch.randn_like(mu[:16])
        st = b["episode_states"][:16]
        generated = generator(st, random_z[:, None].expand(-1, st.shape[1], -1))[:, :, 0]
        rec_z, _ = encoder(st, generated, b["lengths"][:16])
        cycle_errors.append(float((rec_z - random_z).square().mean().sqrt()))
        kl_values.append(float(0.5 * (mu.square() + lv.exp() - 1 - lv).sum(-1).mean()))
    return {"action_reconstruction_l1": float(np.mean(errors)), "teacher_state_latent_rmse": float(np.mean(cycle_errors)), "posterior_kl": float(np.mean(kl_values)), "offline_episodes_sampled": batches * 64, "teacher_cycle_latents_sampled": batches * 16}


@torch.no_grad()
def reconstruction_rollouts(generator, encoder, data_path, device="cuda"):
    """Reconstruct ALL held-out demonstrations by rolling out their encoded means."""
    data = EpisodeDataset(data_path, device).tensors
    codes = []
    for i in range(0, len(data["lengths"]), 128):
        mu, _ = encoder(data["states"][i:i+128, :-1], data["actions"][i:i+128], data["lengths"][i:i+128])
        codes.append(mu)
    result = rollout(generator, data["start"], data["goal"], torch.cat(codes))
    ok = result["success"]
    t = torch.arange(TASK.max_steps, device=device)[None]
    mask = t < torch.minimum(data["lengths"], result["lengths"])[:, None]
    expert_pos = (data["states"][:, :-1, :2] + 1) * 112
    distance_sq = (expert_pos - result["positions"][:, :-1]).square().sum(-1)
    error = ((distance_sq * mask).sum() / mask.sum().clamp_min(1)).sqrt()
    metric = {"episodes": len(ok), "success_rate": float(ok.float().mean()),
              "route_match_fraction_all": float(((result["mode"] == data["mode"]) & ok).float().mean()),
              "time_aligned_position_rmse_pixels": float(error)}
    return metric, result


@torch.no_grad()
def reference_audit(generator, fast_result, count=64):
    """Replay policy actions in original SWM; independently check success/modes."""
    env = make_reference()
    n = len(fast_result["start"])
    # Stratified across configurations AND latents rather than the first pair.
    indices = np.linspace(0, n - 1, min(count, n), dtype=int)
    successes, max_error, modes = 0, 0.0, []
    for i in indices:
        start = fast_result["start"][i].cpu().numpy()
        goal = fast_result["goal"][i].cpu().numpy()
        reset_reference(env, start, goal)
        acts = fast_result["actions"][i].cpu().numpy()
        expected = fast_result["positions"][i].cpu().numpy()
        done, route, prev_x = False, -1, start[0]
        for t, a in enumerate(acts):
            state, _, done, _, _ = env.step(a)
            pos = np.asarray(state[:2])
            max_error = max(max_error, float(np.abs(pos - expected[t + 1]).max()))
            if route < 0 and (pos[0] > 112) != (prev_x > 112):
                route = int(np.argmin(np.abs(pos[1] - np.array(TASK.door_y))))
            prev_x = pos[0]
            if done:
                break
        assert done == bool(fast_result["success"][i]), f"Success mismatch for rollout {i}"
        assert route == int(fast_result["mode"][i]), f"Mode mismatch for rollout {i}"
        successes += int(done)
        modes.append(route)
    env.close()
    assert max_error < 0.002, f"Reference rollout mismatch {max_error}"
    return {"rollouts": len(indices), "success_rate": successes / len(indices),
            "max_position_error_pixels": max_error, "success_and_route_parity": True, "mode_counts": np.bincount(np.asarray(modes) + 1, minlength=3).tolist()}


def load_checkpoint(path, device="cuda", use_ema=True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_config = dict(ckpt["model_config"])
    model_config.setdefault("semantic_style", ckpt.get("train_config", {}).get("style_weight", 0) > 0 or ckpt.get("train_config", {}).get("prior_rec_weight", 0) > 0)
    cfg = ModelConfig(**model_config)
    g, e = Generator(cfg).to(device), TrajectoryEncoder(cfg).to(device)
    g.load_state_dict(ckpt["generator_ema"] if use_ema and "generator_ema" in ckpt else ckpt["generator"])
    e.load_state_dict(ckpt["encoder"])
    return g.eval(), e.eval(), ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="artifacts/data/test.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--latents", type=int, default=16)
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference-count", type=int, default=128)
    p.add_argument("--raw", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g, e, ckpt = load_checkpoint(args.checkpoint, args.device, not args.raw)
    metrics, result = evaluate(g, e, args.data, args.pairs, args.latents, args.seed, args.device)
    metrics.update(offline_metrics(g, e, args.data, args.device))
    metrics["checkpoint_step"] = ckpt["step"]
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["reference_audit"] = reference_audit(g, result, args.reference_count)
    np.savez_compressed(out / "rollouts.npz", **{k: v.cpu().numpy() for k, v in result.items()})
    changed, _ = evaluate(g, e, args.data, args.pairs, args.latents, args.seed, args.device, resample=True)
    metrics["resample_latent_every_replan"] = changed
    # Constant z is evaluated across the same configuration distribution.
    actual_pairs = metrics["start_goal_pairs"]
    z0 = torch.zeros_like(result["z"][:actual_pairs])
    constant = rollout(g, result["start"][::args.latents], result["goal"][::args.latents], z0)
    metrics["constant_zero_latent"] = summarize_rollouts(constant, actual_pairs, 1, e)
    independent, independent_result = evaluate(g, e, args.data, args.pairs, args.latents, args.seed + 73, args.device, independent_latents=True)
    metrics["independent_gaussian_latents"] = independent
    np.savez_compressed(out / "independent_rollouts.npz", **{k: v.cpu().numpy() for k, v in independent_result.items()})
    reconstruction, reconstructed_result = reconstruction_rollouts(g, e, args.data, args.device)
    metrics["encoded_demonstration_rollouts"] = reconstruction
    np.savez_compressed(out / "reconstructed_rollouts.npz", **{k: v.cpu().numpy() for k, v in reconstructed_result.items()})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
