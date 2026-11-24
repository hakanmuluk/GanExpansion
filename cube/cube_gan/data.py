"""Parallel real-MuJoCo collection of balanced multimodal complete episodes."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
from pathlib import Path
import time
import numpy as np

from .env import TASK, make_env, reset_context, snapshot, restore, completion, features, close_env, goal_position
from .expert import Style, select_action

_ENV = None


def initialize_worker():
    global _ENV
    _ENV = make_env()


def context_specs(count, seed, native_fraction=0.5):
    rng = np.random.default_rng(seed)
    native = int(round(count * native_fraction))
    tasks = np.r_[np.arange(native) % 5 + 1, np.zeros(count - native, dtype=int)]
    rng.shuffle(tasks)
    # Each split occupies a disjoint seed bank; no probabilistic overlap.
    seeds = seed * 1_000_000 + rng.choice(900_000, size=count, replace=False)
    return [{"pair_id": j, "seed": int(s), "task_id": int(t)} for j, (s, t) in enumerate(zip(seeds, tasks))]


def collect_context(spec):
    env = _ENV
    ob, info = reset_context(env, spec["seed"], spec["task_id"])
    start = info["privileged/block_0_pos"].copy()
    initial = snapshot(env)
    goal = goal_position(env)
    rng = np.random.default_rng(spec["seed"] + 41)
    speeds = [rng.uniform(TASK.min_speed, TASK.median_speed), rng.uniform(TASK.median_speed, TASK.max_speed)]
    episodes = []
    for mode, speed in [(0, speeds[0]), (0, speeds[1]), (1, speeds[0]), (1, speeds[1])]:
        style = Style(mode, rng.uniform(TASK.min_amplitude, TASK.max_amplitude), speed)
        ob, info = restore(env, initial)
        states = np.zeros((TASK.max_steps + 1, TASK.observation_dim), np.float32)
        actions = np.zeros((TASK.max_steps, TASK.action_dim), np.float32)
        poses = np.zeros((TASK.max_steps + 1, 6), np.float32)
        phases = np.zeros(TASK.max_steps, np.int8)
        stable, native_any, previous_ee = 0, False, None
        for t in range(TASK.max_steps):
            states[t] = features(env, ob, info, start, previous_ee)
            poses[t] = np.r_[info["proprio/effector_pos"], info["privileged/block_0_pos"]]
            a, phase = select_action(env, info, start, style)
            actions[t], phases[t] = a, phase
            previous_ee = info["proprio/effector_pos"].copy()
            ob, _, _, _, info = env.step(a)
            native, strict = completion(env, info)
            native_any |= native
            stable = stable + 1 if strict else 0
            if stable >= TASK.stable_steps:
                break
        length = t + 1
        states[length:] = features(env, ob, info, start, previous_ee)
        poses[length:] = np.r_[info["proprio/effector_pos"], info["privileged/block_0_pos"]]
        phases[length:] = 7
        episodes.append({"states": states, "actions": actions, "poses": poses, "phase": phases,
                         "length": length, "native_success": native_any, "success": stable >= TASK.stable_steps,
                         "mode": mode, "amplitude": style.amplitude, "speed": style.speed})
    return {"spec": spec, "initial_state": initial, "initial_cube": start, "goal": goal, "episodes": episodes}


def save_split(results, out, split):
    contexts = sorted(results, key=lambda x: x["spec"]["pair_id"])
    successful = [c for c in contexts if all(e["success"] for e in c["episodes"])]
    episodes = [e for c in successful for e in c["episodes"]]
    assert episodes, "No successful paired expert demonstrations"
    data = {k: np.stack([e[k] for e in episodes]) for k in ["states", "actions", "poses", "phase"]}
    for k, typ in [("length", np.int64), ("mode", np.int64), ("amplitude", np.float32), ("speed", np.float32)]:
        data["lengths" if k == "length" else k] = np.asarray([e[k] for e in episodes], dtype=typ)
    data["pair_id"] = np.repeat(np.arange(len(successful)), 4)
    data["task_id"] = np.repeat([c["spec"]["task_id"] for c in successful], 4)
    np.savez_compressed(out / f"{split}.npz", **data)
    # Retain ALL fixed contexts for evaluation, including any failed oracle case.
    ctx = {"initial_state": np.stack([c["initial_state"] for c in contexts]),
           "initial_cube": np.stack([c["initial_cube"] for c in contexts]), "goal": np.stack([c["goal"] for c in contexts]),
           "task_id": np.asarray([c["spec"]["task_id"] for c in contexts]), "seed": np.asarray([c["spec"]["seed"] for c in contexts])}
    np.savez_compressed(out / f"{split}_contexts.npz", **ctx)
    all_episodes = [e for c in contexts for e in c["episodes"]]
    stats = {"requested_contexts": len(contexts), "successful_paired_contexts": len(successful), "saved_episodes": len(episodes),
             "oracle_native_success": float(np.mean([e["native_success"] for e in all_episodes])),
             "oracle_released_success": float(np.mean([e["success"] for e in all_episodes])),
             "mode_counts": np.bincount(data["mode"], minlength=2).tolist(), "mean_steps": float(data["lengths"].mean()),
             "max_steps": int(data["lengths"].max()), "task_context_counts": np.bincount(ctx["task_id"], minlength=6).tolist(),
             "failed_context_seeds": [c["spec"]["seed"] for c in contexts if not all(e["success"] for e in c["episodes"])]}
    return stats, data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/data")
    p.add_argument("--train-contexts", type=int, default=1000)
    p.add_argument("--val-contexts", type=int, default=100)
    p.add_argument("--test-contexts", type=int, default=150)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"task": TASK.to_dict(), "environment": "swm/OGBCube-v0, env_type=single", "split_unit": "full initial simulator state and task", "splits": {}}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=initialize_worker) as pool:
        for split, count, seed, native_fraction in [("train", args.train_contexts, 1729, 0.5), ("val", args.val_contexts, 2741, 0.5), ("test", args.test_contexts, 3967, 2/3)]:
            specs = context_specs(count, seed, native_fraction)
            pending = [pool.submit(collect_context, spec) for spec in specs]
            results = []
            for f in as_completed(pending):
                results.append(f.result())
                if len(results) % 50 == 0 or len(results) == count:
                    success = sum(e["success"] for c in results for e in c["episodes"])
                    print(json.dumps({"split": split, "contexts_done": len(results), "total": count, "released_success": success/(4*len(results)), "elapsed_seconds": time.time()-started}), flush=True)
            stats, data = save_split(results, out, split)
            report["splits"][split] = {"seed": seed, **stats}
            if split == "train":
                valid = np.arange(TASK.max_steps)[None] < data["lengths"][:, None]
                x = data["states"][:, :-1][valid].astype(np.float64)
                np.savez(out / "normalization.npz", mean=x.mean(0).astype(np.float32), std=np.maximum(x.std(0), 0.05).astype(np.float32))
            (out / "quality.json").write_text(json.dumps(report, indent=2))
            print("SPLIT_DONE", split, json.dumps(stats), flush=True)
    report["elapsed_seconds"] = time.time() - started
    (out / "quality.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
