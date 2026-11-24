"""Evaluate retained recipes on the SAME validation starts and Gaussian codes."""

import argparse
import csv
import json
from pathlib import Path
import torch

from .evaluate import load_checkpoint, evaluate, offline_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/comparison")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(817)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("01_basic", "01_bicyclegan/latest.pt", "data"),
        ("02_moments", "02_moments_latent/best.pt", "data"),
        ("03_physical", "03_physical_cycle/latest.pt", "data_v2"),
        ("04_aux_seed42", "04_auxiliary_style/selected.pt", "data_v2"),
        ("05_aux_seed123", "05_auxiliary_seed123/selected.pt", "data_v2"),
        ("06_aux_seed2026", "06_auxiliary_seed2026/selected.pt", "data_v2"),
        ("07_label_free", "07_label_free_prior/selected.pt", "data_v2"),
    ]
    reports, flat = {}, []
    for name, relative, data in specs:
        checkpoint = Path("artifacts/runs") / relative
        if not checkpoint.exists():
            checkpoint = Path("artifacts/retained_models") / name / "base_policy.pt"
        if not checkpoint.exists():
            print("SKIP", name, "selected checkpoint unavailable", flush=True)
            continue
        g, e, metadata = load_checkpoint(checkpoint)
        data_path = f"artifacts/{data}/val.npz"
        shared, _ = evaluate(g, e, data_path, pairs=128, latent_count=32, seed=7001)
        independent, _ = evaluate(g, e, data_path, pairs=128, latent_count=16, seed=7074, independent_latents=True)
        offline = offline_metrics(g, e, data_path)
        reports[name] = {"checkpoint": str(checkpoint), "step": metadata["step"], "shared": shared, "independent": independent, "offline": offline}
        row = {"configuration": name, "step": metadata["step"],
               "shared_success": shared["success_rate"], "independent_success": independent["success_rate"],
               "both_routes_shared": shared["both_modes_covered_fraction"],
               "same_z_route_agreement": shared["same_latent_route_agreement_across_states"],
               "actual_latent_rmse": independent["rollout_latent_rmse_all"],
               "action_reconstruction_l1": offline["action_reconstruction_l1"],
               "strict_wall_overlap": independent["strict_body_wall_overlap_episode_fraction"]}
        flat.append(row)
        print(json.dumps(row), flush=True)
        del g, e, metadata
    (out / "validation_comparison.json").write_text(json.dumps(reports, indent=2))
    if flat:
        with (out / "validation_comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


if __name__ == "__main__":
    main()
