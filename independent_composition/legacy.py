"""Read-only loading of the original independently trained action models."""
from pathlib import Path
import hashlib


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_skill(environment, skill, device="cuda", seed=42):
    if environment == "tworoom":
        from tworoom_addons.evaluate import load
        path = (f"addon_artifacts/final/{skill}.pt" if seed == 42 else
                f"addon_artifacts/runs/{skill}_s{seed}/best.pt")
        actor, encoder, metadata = load(path, device)
    elif environment == "cube":
        from cube_addons.evaluate import load_adapter
        path = f"cube_addon_artifacts/final/{skill}_s{seed}/adapter.pt"
        actor, encoder, metadata = load_adapter(path, device=device)
    else:
        raise ValueError(environment)
    actor.eval().requires_grad_(False)
    encoder.eval().requires_grad_(False)
    return actor, encoder, {"checkpoint": path, "sha256": digest_file(path)}
