"""Calibrate experts and record environment facts before dataset collection."""

import argparse
import json
from pathlib import Path
import time
import numpy as np

from .env import TASK, make_env, reset_context, snapshot, restore, completion, features, close_env, goal_position
from .expert import Style, select_action, PHASES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/probe")
    p.add_argument("--contexts", type=int, default=12)
    p.add_argument("--expert", choices=["styled", "native"], default="styled")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env()
    results = []
    start_time = time.time()
    for j in range(args.contexts):
        task_id = j % 6
        ob, info = reset_context(env, 1000 + j, task_id)
        start = info["privileged/block_0_pos"].copy()
        initial = snapshot(env)
        for mode in [0, 1]:
            ob, info = restore(env, initial)
            style = Style(mode, 0.055, 0.8)
            if args.expert == "native":
                from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
                oracle = CubeMarkovOracle(env, min_norm=0.4)
                native_info = dict(info, **{"privileged/target_block": 0, "privileged/target_block_pos": goal_position(env), "privileged/target_block_yaw": np.array([0.0])})
                oracle.reset(ob, native_info)
            stable, native_any = 0, False
            states, actions, poses, phases, contacts = [], [], [], [], []
            prev_ee = None
            for t in range(TASK.max_steps):
                states.append(features(env, ob, info, start, prev_ee))
                poses.append(np.r_[info["proprio/effector_pos"], info["privileged/block_0_pos"]])
                contacts.append([info["proprio/gripper_opening"][0], info["proprio/gripper_contact"][0]])
                if args.expert == "styled":
                    a, phase = select_action(env, info, start, style)
                else:
                    native_info = dict(info, **{"privileged/target_block": 0, "privileged/target_block_pos": goal_position(env), "privileged/target_block_yaw": np.array([0.0])})
                    a, phase = oracle.select_action(ob, native_info), -1
                actions.append(a)
                phases.append(phase)
                prev_ee = info["proprio/effector_pos"].copy()
                ob, reward, _, _, info = env.step(a)
                native, strict = completion(env, info)
                native_any |= native
                stable = stable + 1 if strict else 0
                if stable >= TASK.stable_steps:
                    break
            row = {"context": j, "task_id": task_id, "mode": mode, "native": native_any, "released_settled": stable >= TASK.stable_steps, "steps": t + 1, "phases": {PHASES[k] if k >= 0 else "native": int(np.sum(np.array(phases) == k)) for k in set(phases)}}
            results.append(row)
            np.savez_compressed(out / f"context{j}_mode{mode}.npz", states=np.asarray(states), actions=np.asarray(actions), poses=np.asarray(poses), phase=phases, contact=contacts, goal=goal_position(env), initial_state=initial)
            print(json.dumps(row), flush=True)
    report = {"task": TASK.to_dict(), "expert": args.expert, "results": results, "native_success": float(np.mean([x["native"] for x in results])), "released_success": float(np.mean([x["released_settled"] for x in results])), "elapsed_seconds": time.time() - start_time}
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    close_env(env)


if __name__ == "__main__":
    main()
