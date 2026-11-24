"""Generate balanced successful episodes on GPU, split by start/goal pair."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from .env import TASK, HORIZON, expert_action, observation, transition, verify_reference


def sample_pairs(count: int, seed: int, device="cpu"):
    rng = np.random.default_rng(seed)
    start = np.stack([rng.uniform(30, 88, count), rng.uniform(28, 196, count)], -1)
    goal = np.stack([rng.uniform(136, 194, count), rng.uniform(28, 196, count)], -1)
    reverse = rng.integers(0, 2, count).astype(bool)
    start[reverse, 0] = 224 - start[reverse, 0]
    goal[reverse, 0] = 224 - goal[reverse, 0]
    return torch.tensor(start, dtype=torch.float32, device=device), torch.tensor(goal, dtype=torch.float32, device=device)


@torch.no_grad()
def collect(pairs: int, seed: int, device="cuda", route_offset: float = 0.0) -> dict[str, np.ndarray]:
    starts, goals = sample_pairs(pairs, seed, device)
    starts, goals = starts.repeat_interleave(4, 0), goals.repeat_interleave(4, 0)
    n = len(starts)
    modes = torch.tensor([0, 0, 1, 1], device=device).repeat(pairs)
    rng = np.random.default_rng(seed + 1)
    speeds_np = np.stack([rng.uniform(0.55, 0.78, pairs), rng.uniform(0.78, 1.0, pairs)], -1)
    speeds = torch.tensor(np.tile(speeds_np, (1, 2)).reshape(-1), dtype=torch.float32, device=device)
    offsets = torch.tensor(rng.uniform(-route_offset, route_offset, n), dtype=torch.float32, device=device)
    states = torch.zeros((n, TASK.max_steps + 1, 10), device=device)
    actions = torch.zeros((n, TASK.max_steps, 2), device=device)
    lengths = torch.full((n,), TASK.max_steps, device=device, dtype=torch.int64)
    alive = torch.ones(n, device=device, dtype=torch.bool)
    pos = starts.clone()
    for t in range(TASK.max_steps):
        states[:, t] = observation(pos, goals)
        a = expert_action(pos, goals, modes, speeds, offsets)
        a = torch.where(alive[:, None], a, 0)
        actions[:, t] = a
        pos, _ = transition(pos, a)
        success = (pos - goals).norm(dim=-1) < TASK.success_radius
        lengths = torch.where(alive & success, t + 1, lengths)
        alive &= ~success
    states[:, -1] = observation(pos, goals)
    assert not alive.any(), f"Expert failed on {int(alive.sum())}/{n} episodes"
    return {k: v.cpu().numpy() for k, v in {
        "states": states, "actions": actions, "lengths": lengths,
        "mode": modes, "speed": speeds, "offset": offsets, "pair_id": torch.arange(pairs, device=device).repeat_interleave(4),
        "start": starts, "goal": goals,
    }.items()}


def quality_stats(data):
    lengths = data["lengths"]
    same_pair_modes = data["mode"].reshape(-1, 4)
    positions = (data["states"][..., :2] + 1) * 112
    crossings = (positions[:, 1:, 0] > 112) != (positions[:, :-1, 0] > 112)
    assert crossings.any(1).all(), "A demonstration did not cross the wall"
    first = crossings.argmax(1) + 1
    crossing_y = positions[np.arange(len(first)), first, 1]
    observed_modes = np.abs(crossing_y[:, None] - np.array(TASK.door_y)).argmin(1)
    assert np.array_equal(observed_modes, data["mode"]), "Expert route labels disagree with actual crossings"
    return {
        "episodes": int(len(lengths)), "start_goal_pairs": int(len(lengths) // 4),
        "success_rate": 1.0, "modes_per_pair": 2,
        "mode_counts": np.bincount(data["mode"], minlength=2).tolist(),
        "paired_mode_balance_verified": bool(np.all(same_pair_modes == [0, 0, 1, 1])),
        "actual_route_labels_verified": True,
        "length_mean": float(lengths.mean()), "length_min": int(lengths.min()), "length_max": int(lengths.max()),
        "speed_min": float(data["speed"].min()), "speed_max": float(data["speed"].max()),
        "crossing_offset_min": float(data.get("offset", np.zeros(1)).min()),
        "crossing_offset_max": float(data.get("offset", np.zeros(1)).max()),
        "left_to_right_fraction": float((data["start"][:, 0] < 112).mean()),
    }


class EpisodeDataset:
    """Complete episodes resident on the training device; masks exclude padding."""
    def __init__(self, path, device="cuda"):
        data = np.load(path)
        self.tensors = {k: torch.as_tensor(data[k], device=device) for k in data.files}
        self.device = device
        self.n = len(self.tensors["lengths"])

    def sample(self, batch_size, chunks_per_episode=4):
        d = self.tensors
        idx = torch.randint(self.n, (batch_size,), device=self.device)
        lengths = d["lengths"][idx]
        starts = (torch.rand((batch_size, chunks_per_episode), device=self.device) * lengths[:, None]).long()
        offsets = torch.arange(HORIZON, device=self.device)
        times = starts[..., None] + offsets
        valid = times < lengths[:, None, None]
        action_idx = times.clamp(max=TASK.max_steps - 1)
        return {
            "episode_states": d["states"][idx, :-1],
            "episode_actions": d["actions"][idx],
            "lengths": lengths,
            "states": d["states"][idx[:, None], starts],
            "actions": d["actions"][idx[:, None, None], action_idx],
            "action_mask": valid,
            "mode": d["mode"][idx], "speed": d["speed"][idx],
            "offset": d["offset"][idx] if "offset" in d else torch.zeros_like(d["speed"][idx]),
            "idx": idx,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/data")
    p.add_argument("--train-pairs", type=int, default=2048)
    p.add_argument("--val-pairs", type=int, default=128)
    p.add_argument("--test-pairs", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--route-offset", type=float, default=12.0,
                   help="Episode-level door-crossing offset range in pixels; zero reproduces dataset v1")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parity = verify_reference()
    print("REFERENCE_PARITY", json.dumps(parity), flush=True)
    report = {"task": asdict(TASK), "reference_commit": "21446f1ede6d5284e981bd7b47f432b994e6d812", "reference_parity": parity, "split_unit": "start_goal_pair", "splits": {}}
    for split, count, seed in [("train", args.train_pairs, 1103), ("val", args.val_pairs, 2207), ("test", args.test_pairs, 3301)]:
        data = collect(count, seed, args.device, args.route_offset)
        np.savez_compressed(out / f"{split}.npz", **data)
        report["splits"][split] = {"seed": seed, **quality_stats(data)}
        print(split, json.dumps(report["splits"][split]), flush=True)
    report["elapsed_seconds"] = time.time() - t0
    (out / "quality.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
