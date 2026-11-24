"""Run a base model, one skill, or composition using the published assets."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def specification(args):
    """Resolve model/data choices without importing Torch or a simulator."""
    method = args.method
    if args.case != "compose" and method is not None:
        raise ValueError("--method applies only to --case compose")
    if args.case == "compose":
        method = method or ("task-aware" if args.env == "tworoom" else "agreement")
        allowed = ["agreement", "task-aware"] if args.env == "tworoom" else ["agreement", "route"]
        if method not in allowed:
            raise ValueError(f"{args.env} composition supports: {', '.join(allowed)}")
    seed = args.model_seed if args.model_seed is not None else (43 if args.env == "cube" and args.case == "compose" else 42)
    valid = [42, 123, 2026] if args.env == "tworoom" else [42, 43, 44]
    if seed not in valid:
        raise ValueError(f"{args.env} model seeds: {valid}")
    if args.count < 1 or args.codes < 1:
        raise ValueError("--count and --codes must be positive")
    files = []
    if args.env == "tworoom":
        base = "artifacts/final/base_policy.pt" if seed == 42 else f"artifacts/final/seeds/{seed}/base_policy.pt"
        skills = {s: f"addon_artifacts/final/{s}.pt" if seed == 42 else f"addon_artifacts/runs/{s}_s{seed}/best.pt" for s in ["keepout", "waypoint"]}
        if args.case == "base":
            files = [base, "artifacts/data_v2/test.npz"]
        elif args.case in skills:
            files = ["artifacts/final/base_policy.pt", skills[args.case], f"addon_artifacts/data/{args.case}/test_contexts.npz"]
        else:
            base = "artifacts/final/base_policy.pt"
            files = [base, *skills.values(), f"composition_artifacts/tasks/test_{args.stratum}.npz", f"composition_artifacts/priors/seed{seed}.json"]
        bundles = ["tworoom"]
    else:
        base = "cube/artifacts/final/base_policy.pt" if seed == 42 else f"cube/artifacts/final/seed{seed}/base_policy.pt"
        skills = {s: f"cube_addon_artifacts/final/{s}_s{seed}/adapter.pt" for s in ["keepout", "waypoint"]}
        if args.case == "base":
            files = [base, "cube/artifacts/data/test_contexts.npz"]
        elif args.case in skills:
            base = "cube/artifacts/final/base_policy.pt"
            files = [base, skills[args.case], f"cube_addon_artifacts/data_force/{args.case}/test_contexts.npz"]
        else:
            base = "cube/artifacts/final/base_policy.pt"
            actor = f"cube_agreement_artifacts/{'action_adapter_v3' if seed == 42 else 'action_adapter_v3_s'+str(seed)}/best.pt"
            skills = {s: f"cube_agreement_artifacts/decoders/{s}_s{seed}/best.pt" for s in ["keepout", "waypoint"]}
            files = [base, *skills.values(), actor, f"cube_agreement_artifacts/tasks_v1/test_{args.stratum}.npz"]
        bundles = ["cube-agreement" if args.case == "compose" else "cube"]
    return {"env": args.env, "case": args.case, "method": method, "model_seed": seed,
            "base": base, "skills": skills, "required_files": files, "download_bundles": bundles,
            **({"actor": actor} if args.env == "cube" and args.case == "compose" else {})}


def save_tensor_trace(path, trace):
    import numpy as np
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() if hasattr(v, "detach") else v for k,v in trace.items()})


def run_tworoom(args, spec, out, device):
    import torch
    if args.case == "base":
        from tworoom_gan.evaluate import load_checkpoint, evaluate
        generator, encoder, _ = load_checkpoint(spec["base"], device)
        summary, trace = evaluate(generator, encoder, "artifacts/data_v2/test.npz", args.count,
                                  args.codes, args.seed, device, independent_latents=True)
    elif args.case in ["keepout", "waypoint"]:
        from tworoom_addons.evaluate import load, contexts, evaluate
        model, encoder, _ = load(spec["skills"][args.case], device)
        bank = contexts(f"addon_artifacts/data/{args.case}/test_contexts.npz", device, args.count)
        summary, trace = evaluate(model, encoder, bank, args.case, args.codes, args.seed, gaussian=True)
    else:
        from tworoom_composition.models import FrozenBank
        from tworoom_composition.search import SearchConfig
        from tworoom_composition.evaluate import load_cases, summary as summarize
        from tworoom_composition.rollout import rollout
        name = "tworoom_task_aware" if spec["method"] == "task-aware" else "tworoom_agreement"
        config = json.loads((ROOT/f"configs/{name}.json").read_text())
        # CUDA graphs are only a throughput optimization; CPU uses eager mode.
        config["cuda_graph"] = device == "cuda" and config.get("cuda_graph", False)
        bank = FrozenBank(spec["skills"]["keepout"], spec["skills"]["waypoint"], device=device)
        priors = json.loads((ROOT/f"composition_artifacts/priors/seed{spec['model_seed']}.json").read_text())
        if priors["frozen_files"] != bank.files:
            raise ValueError("Composition prior/model fingerprints differ")
        cases = load_cases(f"composition_artifacts/tasks/test_{args.stratum}.npz", device, args.count)
        rows, trace, planning = rollout(bank, cases, SearchConfig(**config), priors["skills"], args.seed)
        summary = summarize(rows, planning)
        (out/"records.json").write_text(json.dumps(rows, indent=2)+"\n")
        (out/"planning.json").write_text(json.dumps(planning, indent=2)+"\n")
    save_tensor_trace(out/"trajectories.npz", trace)
    return summary


def run_cube_base(args, spec, out, device):
    import numpy as np
    from cube.cube_gan.env import make_env, reset_context, restore, features, completion, close_env, goal_position
    from cube.cube_gan.policy import CubePolicy
    from cube_addons.evaluate import load_contexts
    policy = CubePolicy(spec["base"], device)
    cases = load_contexts("cube/artifacts/data/test_contexts.npz", args.count)
    env = make_env()
    rng = np.random.default_rng(args.seed)
    rows = []
    try:
        for i, case in enumerate(cases):
            for j in range(args.codes):
                reset_context(env, int(case["seed"]), int(case["task_id"]))
                ob, info = restore(env, case["initial_state"])
                policy.reset(rng=rng)
                previous = None; stable = 0; actions = []; poses = []; native_seen = False
                for t in range(200):
                    state = features(env, ob, info, case["initial_cube"], previous)
                    action = policy.act(state)
                    poses.append(np.r_[info["proprio/effector_pos"], info["privileged/block_0_pos"]])
                    previous = info["proprio/effector_pos"].copy()
                    ob, _, _, _, info = env.step(action)
                    native, released = completion(env, info)
                    native_seen |= native
                    stable = stable+1 if released else 0
                    actions.append(action)
                    if stable >= 10: break
                poses.append(np.r_[info["proprio/effector_pos"], info["privileged/block_0_pos"]])
                row = {"context": i, "code_index": j, "steps": len(actions), "native_success": bool(native_seen),
                       "success": stable >= 10, "released_success": stable >= 10,
                       "final_goal_error": float(np.linalg.norm(info["privileged/block_0_pos"]-goal_position(env)))}
                rows.append(row)
                np.savez_compressed(out/f"rollout_{i:04d}_{j:02d}.npz", actions=actions, poses=poses,
                                    initial_state=case["initial_state"], base_z=policy.z)
    finally:
        close_env(env)
    (out/"records.json").write_text(json.dumps(rows, indent=2)+"\n")
    return {"episodes": len(rows), "success": float(np.mean([r["success"] for r in rows])),
            "native_success": float(np.mean([r["native_success"] for r in rows])),
            "success_definition": "Released and settled original-goal placement; no added skill."}


def run_cube(args, spec, out, device):
    if args.case == "base":
        return run_cube_base(args, spec, out, device)
    if args.case in ["keepout", "waypoint"]:
        from cube_addons.evaluate import load_adapter, load_contexts, evaluate
        model, encoder, _ = load_adapter(spec["skills"][args.case], spec["base"], device)
        cases = load_contexts(f"cube_addon_artifacts/data_force/{args.case}/test_contexts.npz", args.count)
        summary, _ = evaluate(model, encoder, cases, args.case, args.codes, args.seed, out=out, force_contacts=True)
        return summary
    tasks = f"cube_agreement_artifacts/tasks_v1/test_{args.stratum}.npz"
    if spec["method"] == "agreement":
        if device != "cuda":
            raise ValueError("Raw Cube action agreement uses the CUDA batch evaluator. Use --device cuda or --method route --device cpu.")
        command = [sys.executable, "-m", "cube_agreement.batch_evaluate", "--tasks", tasks, "--out", str(out),
                   "--model-seed", str(spec["model_seed"]), "--seed", str(args.seed), "--count", str(args.count),
                   "--batch", str(min(args.count,64)), "--steps", "80", "--starts", "12",
                   "--action-steps", "20", "--action-starts", "1", "--action-interval", "1"]
    else:
        command = [sys.executable, "-m", "cube_agreement.actor_evaluate", "--tasks", tasks, "--out", str(out),
                   "--keepout", spec["skills"]["keepout"], "--waypoint", spec["skills"]["waypoint"],
                   "--actor", spec["actor"], "--method", "agreement", "--steps", "120", "--starts", "12",
                   "--execute", "1", "--seed", str(args.seed), "--count", str(args.count), "--device", device]
    subprocess.run(command, check=True, cwd=ROOT)
    return json.loads((out/"summary.json").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["tworoom", "cube"], required=True)
    parser.add_argument("--case", choices=["base", "keepout", "waypoint", "compose"], required=True)
    parser.add_argument("--method", choices=["agreement", "task-aware", "route"])
    parser.add_argument("--count", type=int, default=4, help="Number of held-out contexts, not a full benchmark by default")
    parser.add_argument("--codes", type=int, default=2, help="Codes per context for base/individual skills")
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--seed", type=int, help="Code/search RNG; default 41 for TwoRoom composition, 42 otherwise")
    parser.add_argument("--stratum", choices=["supported_easy", "supported_conflict", "coupled_ood"], default="supported_conflict")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Show exact assets/model choices without importing ML libraries")
    args = parser.parse_args(argv)
    try:
        spec = specification(args)
        if args.dry_run:
            print(json.dumps(spec, indent=2)); return
        missing = [p for p in spec["required_files"] if not (ROOT/p).is_file()]
        if missing:
            raise ValueError("Missing assets. Run: python scripts/download_assets.py --bundle "+spec["download_bundles"][0])
        import torch
        torch.set_num_threads(2)
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select a GPU runtime or use --device cpu for supported examples.")
        if args.env == "cube" and args.case == "compose" and spec["method"] == "agreement" and device != "cuda":
            raise ValueError("Cube raw action agreement requires CUDA; use a GPU runtime or --method route.")
        args.seed = args.seed if args.seed is not None else (41 if args.env == "tworoom" and args.case == "compose" else 42)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = (args.out or ROOT/f"outputs/{args.env}_{args.case}_{spec['method'] or 'policy'}_{stamp}").resolve()
        if out.exists() and any(out.iterdir()):
            raise ValueError("Use a fresh --out folder; existing results are not overwritten.")
        out.mkdir(parents=True, exist_ok=True)
        os.chdir(ROOT)  # Preserve the original loaders' relative checkpoint paths.
        # The batch evaluator owns config.json; keep wrapper metadata separate.
        (out/"example.json").write_text(json.dumps({**spec, "device":device, "count":args.count,
                                                   "codes":args.codes, "search_seed":args.seed}, indent=2)+"\n")
        summary = run_tworoom(args, spec, out, device) if args.env == "tworoom" else run_cube(args, spec, out, device)
        (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2))
        print(f"Saved to {out}", flush=True)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Example failed: {error}\n")


if __name__ == "__main__":
    main()
