"""Frozen base/adapter inference and native composition demos.

Keep a LoadedPolicy instance alive across replans. The CLI accepts --previous
to carry the selected inputs between observation files from the same episode.
Synthetic --check probes are interface checks, never benchmark measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


def configuration(environment, method="single-gaussian", steps=20):
    """The frozen study settings; only the declared initialization/budget vary."""
    if environment not in ("tworoom", "cube"):
        raise ValueError(environment)
    if method not in ("grid", "single-gaussian"):
        raise ValueError(method)
    if steps not in (0, 5, 10, 20, 40):
        raise ValueError("Adam steps must be 0, 5, 10, 20 or 40")
    if method == "grid" and steps != 20:
        raise ValueError("The original reference uses grid initialization and 20 updates")
    c = json.loads((ROOT / "configs/composition" / environment / "grid.json").read_text())
    c.update(steps=steps, starts=4 if method == "grid" else 1)
    return c


class LoadedPolicy:
    def __init__(self, environment, method="single-gaussian", steps=20, seed=241, device="cpu"):
        import torch
        from independent_composition.regularization_models import load_bank
        from independent_composition.regularized_latent_composition import RegularizedConfig, RegularizedComposer
        from single_gaussian_study.planning import SingleGaussianComposer

        self.environment, self.method = environment, method
        self.seed, self.device = seed, device
        self.episode_base_code = None
        config = RegularizedConfig(**configuration(environment, method, steps))
        manifest = json.loads((ROOT / "configs/research_assets.json").read_text())
        entries = {e["path"]: e for e in manifest["bundles"]["composition-models"]["files"]}
        models = {}
        for role in ("base", "keepout", "waypoint"):
            relative = f"research_assets/composition/{environment}/{role}.pt"
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError("Download models first: python scripts/download_assets.py "
                                        "--manifest configs/research_assets.json --bundle composition")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entries[relative]["sha256"]:
                raise ValueError(f"Frozen checkpoint checksum mismatch: {relative}")
            models[role] = str(path)
        torch.set_num_threads(2)
        self.bank = load_bank(environment, **models, device=device)
        self.config = config
        cls = SingleGaussianComposer if method == "single-gaussian" else RegularizedComposer
        self.planner = cls(self.bank, config, seed, record=False) if method == "single-gaussian" else cls(self.bank, config, seed)

    def predict(self, observation, case="compose", private_code=-.8):
        import numpy as np
        import torch
        from independent_composition.inference_guard import tripwires

        state_dim, context_dim = (10, 12) if self.environment == "tworoom" else (47, 20)

        def tensor(key, width):
            x = np.atleast_2d(np.asarray(observation[key], dtype=np.float32))
            if x.ndim != 2 or x.shape[1] != width or not np.isfinite(x).all():
                raise ValueError(f"{key} must be a finite [batch,{width}] array")
            return torch.as_tensor(x, device=self.device)

        s, b = tensor("state", state_dim), tensor("base_z", 2)
        if len(s) != len(b):
            raise ValueError("Match state/base_z batch sizes")
        if case == "base":
            with torch.no_grad(), tripwires():
                return self.bank.base(s, b)
        names = ("keepout_context", "waypoint_context") if case == "compose" else (case + "_context",)
        contexts = tuple(tensor(k, context_dim) for k in names)
        if any(len(c) != len(s) for c in contexts):
            raise ValueError("Context batch sizes must match state")
        with tripwires():
            if case == "compose":
                if bool((b[:, 0] == 0).any()):
                    raise ValueError("Composition requires nonzero first base-code coordinates")
                if self.episode_base_code is None:
                    self.episode_base_code = b.detach().clone()
                elif not torch.equal(self.episode_base_code, b):
                    raise ValueError("Keep the original base code/batch fixed; load a new policy for a new episode")
                # Input gradients are required even though every model is frozen.
                return self.planner.sample(s, contexts, b)
            if case not in ("keepout", "waypoint"):
                raise ValueError(case)
            actor = self.bank.k if case == "keepout" else self.bank.w
            with torch.no_grad():
                return actor(s, b, s.new_full((len(s), 1), private_code), contexts[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=["tworoom", "cube"], required=True)
    p.add_argument("--case", choices=["base", "keepout", "waypoint", "compose"], default="compose")
    p.add_argument("--method", choices=["grid", "single-gaussian"], default="single-gaussian")
    p.add_argument("--steps", type=int, choices=[0, 5, 10, 20, 40], default=20)
    p.add_argument("--seed", type=int, default=241)
    p.add_argument("--device", default="cpu")
    p.add_argument("--private-code", type=float, default=-.8)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--check", action="store_true", help="Load weights and probe synthetic interfaces; no task score")
    source.add_argument("--input", type=Path, help="NPZ containing current measured features")
    source.add_argument("--tasks", type=Path, help="Native joint-task NPZ bank, generated/restored locally")
    p.add_argument("--previous", type=Path, help="Previous prediction file from this same episode")
    p.add_argument("--count", type=int, default=4, help="Native demo contexts, not the full study")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.previous and (not a.input or a.case != "compose"):
        p.error("--previous requires --input and --case compose")
    if a.tasks and a.case != "compose":
        p.error("--tasks uses the joint composition monitor; individual skills use examples/run.py")
    if a.count < 1:
        p.error("--count must be positive")
    if not a.check and (a.out is None or a.out.exists()):
        p.error("Provide a fresh --out file (prediction) or directory (native evaluation)")
    if a.input and a.out.suffix != ".npz":
        p.error("Prediction --out must end in .npz")
    policy = LoadedPolicy(a.env, a.method, a.steps, a.seed, a.device)
    import numpy as np
    import torch

    if a.tasks:
        from independent_composition.evaluate_diffusion import evaluate_tworoom, evaluate_cube, summarize
        from independent_composition.inference_guard import GuardedPolicy
        from independent_composition.regularized_latent_composition import RegularizedComposer
        from single_gaussian_study.planning import SingleGaussianComposer

        def factory(options, unused, base, seed):
            cls = SingleGaussianComposer if a.method == "single-gaussian" else RegularizedComposer
            actor = cls(policy.bank, policy.config, seed, record=False) if a.method == "single-gaussian" else cls(policy.bank, policy.config, seed)
            return GuardedPolicy(actor)

        a.out.mkdir(parents=True)
        options = SimpleNamespace(tasks=str(a.tasks), count=a.count, device=a.device, seed=a.seed, execute=8)
        evaluator = evaluate_tworoom if a.env == "tworoom" else evaluate_cube
        rows = evaluator(options, [], policy.bank.k.base, a.out, factory)
        (a.out / "records.json").write_text(json.dumps(rows, indent=2) + "\n")
        if a.env == "tworoom":
            from tworoom_gan.env import make_reference
            from tworoom_composition.audit import replay
            env = make_reference()
            try:
                replay_result = replay(a.out, env)
            finally:
                env.close()
            assert replay_result["all_match"] and replay_result["episodes"] == len(rows)
        summary = {**summarize(rows), "config": vars(policy.config), "environment": a.env,
                   "method": a.method, "seed": a.seed, "scope": "Native closed-loop demo; full study uses 64 contexts/category and all three seeds."}
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
        return
    if a.check:
        width, context = (10, 12) if a.env == "tworoom" else (47, 20)
        state = np.zeros((1, width), np.float32)
        if a.env == "cube":
            state = policy.bank.k.base.obs_mean.detach().cpu().numpy().reshape(1, width)
        observed = {"state": state, "base_z": np.array([[1., .2]], np.float32),
                    "keepout_context": np.zeros((1, context), np.float32),
                    "waypoint_context": np.zeros((1, context), np.float32)}
    else:
        with np.load(a.input, allow_pickle=False) as z:
            observed = {k: z[k] for k in z.files}
    if a.previous:
        with np.load(a.previous, allow_pickle=False) as z:
            if not {"selected_inputs", "environment", "method", "seed", "steps", "base_z"}.issubset(z.files):
                p.error("--previous must be a composition prediction from this entrypoint")
            if str(z["environment"]) != a.env or str(z["method"]) != a.method or int(z["seed"]) != a.seed:
                p.error("Previous prediction environment, method and seed must match")
            if int(z["steps"]) != a.steps:
                p.error("Keep the Adam budget fixed across replans")
            if not np.array_equal(np.atleast_2d(observed["base_z"]), z["base_z"]):
                p.error("Keep the original base code fixed throughout an episode")
            previous = torch.as_tensor(z["selected_inputs"], dtype=torch.float32, device=a.device)
            if previous.shape != (len(np.atleast_2d(observed["state"])), 4):
                p.error("Previous selected inputs have the wrong batch shape")
            policy.planner.previous = previous
    actions = policy.predict(observed, a.case, a.private_code).detach().cpu().numpy()
    assert np.isfinite(actions).all()
    policy.bank.assert_frozen()
    if a.check:
        print(json.dumps({"passed": True, "environment": a.env, "case": a.case, "method": a.method,
                          "action_shape": list(actions.shape), "models_frozen": True,
                          "synthetic_interface_probe_only": True, "benchmark_score": None}))
        return
    issued = actions[:, :8].copy()
    if a.env == "cube":
        issued[..., 4] = np.where(issued[..., 4] >= 0, 1., -1.)
    arrays = {"actions": actions, "execute_actions": issued, "base_z": np.atleast_2d(observed["base_z"]),
              "environment": a.env, "method": a.method, "seed": a.seed, "steps": a.steps}
    if a.case == "compose":
        arrays["selected_inputs"] = policy.planner.previous.detach().cpu().numpy()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **arrays)
    print(json.dumps({"saved": str(a.out), "predicted": list(actions.shape), "execute": list(issued.shape)}))


if __name__ == "__main__":
    main()
