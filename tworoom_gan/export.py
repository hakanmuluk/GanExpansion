"""Export the selected G/E inference checkpoint and provenance manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="artifacts/final")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    export = {k: v for k, v in ckpt.items() if k in ["step", "model_config", "train_config", "task", "normalization", "execute_actions", "generator_ema", "encoder", "selection", "validation"]}
    # Keep a conventional generator key too for code that requests raw weights;
    # exported policy intentionally uses the selected EMA generator in either case.
    export["generator"] = ckpt.get("generator_ema", ckpt["generator"])
    export["model_config"] = dict(ckpt["model_config"])
    export["model_config"].setdefault("semantic_style", ckpt.get("train_config", {}).get("style_weight", 0) > 0 or ckpt.get("train_config", {}).get("prior_rec_weight", 0) > 0)
    path = out / "base_policy.pt"
    torch.save(export, path)
    source_files = sorted(p for p in Path("tworoom_gan").glob("*.py") if not p.name.startswith(".")) + [Path("requirements.txt"), Path("pyproject.toml")]
    manifest = {"checkpoint": str(path), "checkpoint_sha256": digest(path), "source_checkpoint": str(args.checkpoint),
                "source_checkpoint_sha256": digest(args.checkpoint), "step": ckpt["step"],
                "model_config": export["model_config"], "task": ckpt.get("task"),
                "normalization": "pixel / 112 - 1", "predict_actions": 16, "execute_actions": 8,
                "latent_sampling": "N(0,I), once per episode", "inference_expert_used": False,
                "source_sha256": {str(p): digest(p) for p in source_files},
                "reference_commit": "21446f1ede6d5284e981bd7b47f432b994e6d812"}
    if Path("artifacts/runtime.json").exists():
        manifest["runtime"] = json.loads(Path("artifacts/runtime.json").read_text())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
