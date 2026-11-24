"""Select an inference snapshot using validation only, after GAN warmup."""

import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--min-step", type=int, default=2500)
    args = p.parse_args()
    run = Path(args.run)
    rows = [json.loads(line) for line in (run / "validation.jsonl").read_text().splitlines() if line]
    candidates = [r for r in rows if r["step"] >= args.min_step and (run / "snapshots" / f"step_{r['step']:06d}.pt").exists()]
    if not candidates:
        raise RuntimeError("No eligible post-warmup snapshots exist")

    def rank(r):
        return (r["success_rate"], r["both_modes_covered_fraction"], r["same_latent_route_agreement_across_states"],
                -r.get("strict_body_wall_overlap_episode_fraction", 1.0),
                -r.get("rollout_latent_rmse_all", 100.0), -r["action_reconstruction_l1"])

    chosen = max(candidates, key=rank)
    source = run / "snapshots" / f"step_{chosen['step']:06d}.pt"
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    selection = {"source": str(source), "min_step": args.min_step,
                 "criteria_lexicographic": ["success_rate", "both_modes_covered_fraction", "same_latent_route_agreement_across_states", "negative_strict_wall_overlap", "negative_actual_rollout_latent_rmse", "negative_action_reconstruction_l1"],
                 "validation": chosen, "test_used": False}
    ckpt["selection"] = selection
    torch.save(ckpt, run / "selected.pt")
    (run / "selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
