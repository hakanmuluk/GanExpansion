"""Audit actual deliverables and numerical results against the baseline objective."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageStat


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    checks = {}
    final = Path("artifacts/final")
    manifest = json.loads((final / "manifest.json").read_text())
    assert sha256(final / "base_policy.pt") == manifest["checkpoint_sha256"]
    for file, digest in manifest["source_sha256"].items():
        assert sha256(file) == digest, f"Source hash mismatch: {file}"
    checks["checkpoint_and_source_hashes"] = True
    quality = json.loads(Path("artifacts/data_v2/quality.json").read_text())
    assert quality["reference_parity"]["random_transitions"] >= 3000
    assert quality["reference_parity"]["max_position_error_pixels"] < 3e-5
    sets, counts = {}, {}
    for split, required in [("train", 8192), ("val", 512), ("test", 1024)]:
        d = np.load(f"artifacts/data_v2/{split}.npz")
        assert len(d["lengths"]) == required
        assert np.isfinite(d["states"]).all() and np.isfinite(d["actions"]).all()
        assert np.abs(d["actions"]).max() <= 1.00001
        assert np.all(d["mode"].reshape(-1, 4) == [0, 0, 1, 1])
        endpoints = (d["states"][np.arange(required), d["lengths"], :2] + 1) * 112
        assert np.all(np.linalg.norm(endpoints - d["goal"], axis=-1) < 16.0001)
        sets[split] = set(map(tuple, np.concatenate([d["start"][::4], d["goal"][::4]], -1)))
        counts[split] = required
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    checks["successful_complete_paired_data_and_disjoint_splits"] = counts
    seed_summaries = {}
    for seed, run_name in [(42, "04_auxiliary_style"), (123, "05_auxiliary_seed123"), (2026, "06_auxiliary_seed2026")]:
        run = Path("artifacts/runs") / run_name
        done = json.loads((run / "DONE.json").read_text())
        assert done["step"] == 5000
        selected = json.loads((run / "selection.json").read_text())
        assert selected["validation"]["step"] >= 2500 and not selected["test_used"]
        m = json.loads(Path(f"artifacts/evaluation/test_seed{seed}/metrics.json").read_text())
        i = m["independent_gaussian_latents"]
        assert m["rollouts"] == 8192 and i["rollouts"] == 8192
        assert m["success_rate"] >= 0.95 and i["success_rate"] >= 0.95
        assert m["both_modes_covered_fraction"] >= 0.95
        assert m["same_latent_route_agreement_across_states"] >= 0.95
        assert i["rollout_latent_rmse_all"] < 0.35
        assert all(x is not None and x > 0.9 for x in i["rollout_latent_correlations"])
        assert i["intended_route_accuracy_successful"] >= 0.99
        assert i["crossing_offset_rmse_pixels_successful"] < 3
        assert i["mean_speed_mae_successful"] < 0.08
        assert m["encoded_demonstration_rollouts"]["episodes"] == 1024
        assert m["encoded_demonstration_rollouts"]["success_rate"] >= 0.95
        assert m["encoded_demonstration_rollouts"]["route_match_fraction_all"] >= 0.95
        assert m["constant_zero_latent"]["both_modes_covered_fraction"] == 0
        assert m["reference_audit"]["rollouts"] >= 128
        assert m["reference_audit"]["success_and_route_parity"]
        assert m["reference_audit"]["max_position_error_pixels"] < 0.002
        assert (run / "train.jsonl").stat().st_size > 0
        for row in [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines()]:
            assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
        seed_summaries[str(seed)] = {"selected_step": m["checkpoint_step"], "test_independent_success": i["success_rate"], "test_both_routes": m["both_modes_covered_fraction"], "test_latent_rmse": i["rollout_latent_rmse_all"]}
    checks["three_completed_seeds_and_closed_loop_tests"] = seed_summaries
    resume = torch.load(final / "training_resume.pt", map_location="cpu", weights_only=False)
    d_steps = int(next(iter(resume["optimizer_d"]["state"].values()))["step"])
    assert resume["step"] == 5000 and d_steps == 4000
    assert resume["train_config"]["adv_weight"] > 0
    assert all(key in resume for key in ["generator", "generator_ema", "encoder", "discriminator", "optimizer_ge", "optimizer_d"])
    checks["full_g_e_d_and_optimizers_with_actual_adversarial_updates"] = {"training_steps": 5000, "discriminator_updates": d_steps}
    for device, count in [("cuda", 64), ("cpu", 8)]:
        api = json.loads((final / f"policy_api_{device}.json").read_text())
        assert api["episodes"] == count and api["original_environment_success_rate"] >= 0.95
        assert api["expert_disabled_during_inference"]
    checks["public_policy_api_original_environment_cpu_and_gpu"] = True
    images = ["dataset_quality.png", "training_curves.png", "gan_stability.png", "generated_trajectories.png", "latent_control_and_recovery.png", "style_control_accuracy.png", "policy_rollout_keyframe.png"]
    for name in images:
        with Image.open(final / "figures" / name) as im:
            assert min(im.size) >= 300
            assert max(ImageStat.Stat(im.convert("RGB")).var) > 1
    with Image.open(final / "figures/policy_rollouts.gif") as animation:
        assert animation.n_frames >= 20
    anim = np.load(final / "figures/policy_rollouts.npz")
    assert anim["success"].all() and len(set(anim["mode"].tolist())) == 2
    assert np.all(np.linalg.norm(anim["positions"][:, -1] - anim["positions"][:, 0], axis=-1) > 40)
    checks["nonempty_figures_and_successful_two_route_animation"] = images
    assert Path("RESULTS.md").exists() and Path("README.md").exists() and Path("EXPERIMENTS.md").exists()
    assert Path("artifacts/comparison/validation_comparison.csv").exists()
    for name in ["01_bicyclegan", "02_moments_latent", "03_physical_cycle", "07_label_free_prior"]:
        assert (Path("artifacts/runs") / name / "config.json").exists()
    checks["attempts_reproduction_and_results_documented"] = True
    report = {"passed": True, "checks": checks,
              "scope": "Fixed-layout state-based TwoRoom base policy; auxiliary synthetic expert-style supervision is explicit. No skill expansion/composition implemented."}
    (final / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
