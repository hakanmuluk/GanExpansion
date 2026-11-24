"""Install dependencies for the included tasks without replacing PyTorch."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SWM = "stable-worldmodel @ git+https://github.com/galilai-group/stable-worldmodel.git@21446f1ede6d5284e981bd7b47f432b994e6d812"


def pip(*arguments):
    subprocess.run([sys.executable, "-m", "pip", "install", *arguments], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=["tworoom", "all"], default="all")
    args = p.parse_args()
    if sys.version_info < (3, 10):
        p.error("Use Python 3.10 or newer; Python 3.11 is recommended for a fresh environment.")
    try:
        import torch
    except ImportError:
        p.error("Install the desired CPU/CUDA PyTorch build first: https://pytorch.org/get-started/locally/")
    print("Keeping PyTorch", torch.__version__, "CUDA", torch.cuda.is_available(), flush=True)
    pip("-r", str(ROOT/"requirements.txt"))
    # Older Torch wheels may not support NumPy 2. Test the actual tensor bridge
    # rather than replacing the user's selected CPU/CUDA Torch installation.
    bridge = "import numpy as np,torch; torch.from_numpy(np.zeros(1)).numpy()"
    checked = subprocess.run([sys.executable, "-c", bridge])
    if checked.returncode and sys.version_info < (3, 13):
        pip("numpy>=1.24,<2")
        subprocess.run([sys.executable, "-c", bridge], check=True)
    elif checked.returncode:
        p.error("The installed PyTorch/NumPy bridge is incompatible; install a PyTorch build supporting this Python version.")
    if args.env == "all":
        pip("mujoco==3.11.0", "dm-env==1.6", "glfw==2.11.0", "absl-py", "dm-tree",
            "lxml", "PyOpenGL", "protobuf", "pyparsing", "requests", "etils[epath]")
        # These Cube tasks do not need labmaze; avoid its Python 3.13 build chain.
        pip("--no-deps", "dm_control==1.0.44", "ogbench==1.2.2")
    # Only environment APIs are used; do not install SWM's world-model training,
    # dataset, arcade or robotics extras, or replace Torch/torchvision.
    pip("--no-deps", SWM)
    pip("--no-deps", "-e", str(ROOT))
    check = "from tworoom_gan.env import make_reference; e=make_reference(); e.close(); print('TwoRoom import/reset API OK')"
    subprocess.run([sys.executable, "-c", check], cwd=ROOT, check=True)
    if args.env == "all":
        env = os.environ.copy()
        if sys.platform.startswith("linux"):
            env.setdefault("MUJOCO_GL", "egl")
        check = "from cube_addons.env import make_env,reset_case,close_env; e=make_env(); reset_case(e,100001); close_env(e); print('Native Cube reset OK')"
        subprocess.run([sys.executable, "-c", check], cwd=ROOT, env=env, check=True)
    print("Environment ready. Download assets with: python scripts/download_assets.py --bundle all")


if __name__ == "__main__":
    main()
