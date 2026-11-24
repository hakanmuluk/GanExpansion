"""Exercise the public policy API against original SWM, with expert disabled."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from . import env as environment
from .policy import TrajectoryPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="artifacts/data_v2/test.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=11627)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(4)
    data = np.load(args.data)
    policy = TrajectoryPolicy(args.checkpoint, args.device)
    env = environment.make_reference()
    def forbidden_expert(*args, **kwargs):
        raise AssertionError("An expert was called during policy inference")
    environment.expert_action = forbidden_expert
    rng = np.random.default_rng(args.seed)
    indices = np.linspace(0, len(data["start"]) // 4 - 1, args.count, dtype=int) * 4
    trials = []
    for idx in indices:
        z = rng.normal(size=policy.generator.config.z_dim).astype(np.float32)
        state, _ = environment.reset_reference(env, data["start"][idx], data["goal"][idx])
        policy.reset(z=z)
        old_x, mode, done = float(state[0]), -1, False
        for t in range(environment.TASK.max_steps):
            action = policy.act(state)
            assert action.shape == (2,) and np.isfinite(action).all() and np.abs(action).max() <= 1
            state, _, done, truncated, _ = env.step(action)
            if mode < 0 and (float(state[0]) > 112) != (old_x > 112):
                mode = int(np.abs(float(state[1]) - np.array(environment.TASK.door_y)).argmin())
            old_x = float(state[0])
            if done or truncated:
                break
        trials.append({"pair": int(idx // 4), "z": z.tolist(), "success": bool(done), "mode": mode, "steps": t + 1})
    env.close()
    report = {"checkpoint": args.checkpoint, "device": args.device, "episodes": args.count,
              "original_environment_success_rate": sum(t["success"] for t in trials) / len(trials),
              "expert_disabled_during_inference": True, "sample_once_per_episode": True,
              "predict_actions": 16, "execute_actions": 8, "trials": trials}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))
    assert report["original_environment_success_rate"] >= 0.95


if __name__ == "__main__":
    main()
