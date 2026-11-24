"""Animate learned closed-loop rollouts using the original TwoRoom renderer."""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from .env import make_reference, reset_reference
from .evaluate import load_checkpoint, rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="artifacts/figures/policy_rollouts.gif")
    args = p.parse_args()
    torch.set_num_threads(4)
    g, _, _ = load_checkpoint(args.checkpoint)
    z = torch.tensor([[-1.0, -0.8], [1.0, -0.8], [-0.4, 0.8], [1.8, 0.8]], device="cuda")
    starts = torch.tensor([[50.0, 112.0]], device="cuda").repeat(4, 1)
    goals = torch.tensor([[175.0, 112.0]], device="cuda").repeat(4, 1)
    r = rollout(g, starts, goals, z)
    pos = r["positions"].cpu().numpy()
    lengths = r["lengths"].cpu().numpy()
    success = r["success"].cpu().numpy()
    envs = [make_reference() for _ in range(4)]
    for env in envs:
        reset_reference(env, [50, 112], [175, 112])
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
    small = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 11)
    frames = []
    colors = [(40, 100, 180), (224, 132, 39)]
    stop = min(int(lengths.max()) + 12, pos.shape[1])
    for t in range(stop):
        canvas = Image.new("RGB", (472, 590), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), "Same start/goal; fixed latent throughout each rollout", font=font, fill="#202428")
        draw.text((10, 27), f"Learned generator only | 16 predict / 8 execute | step {t}", font=small, fill="#50565b")
        for j, env in enumerate(envs):
            env._set_state(pos[j, t])
            image = Image.fromarray(env.render())
            trace = ImageDraw.Draw(image)
            mode = int(r["mode"][j])
            if t > 0:
                trace.line([tuple(p) for p in pos[j, :t+1]], fill=colors[max(mode, 0)], width=2)
            x, y = 8 + (j % 2) * 232, 84 + (j // 2) * 268
            canvas.paste(image, (x, y))
            status = "success" if success[j] and t >= lengths[j] else "moving"
            draw.text((x + 6, y - 20), f"z=[{float(z[j,0]):+.1f},{float(z[j,1]):+.1f}]  {status}", font=small, fill="#202428")
        frames.append(canvas)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    frames[min(30, len(frames) - 1)].save(out.with_name("policy_rollout_keyframe.png"))
    np.savez_compressed(out.with_suffix(".npz"), **{k: v.cpu().numpy() for k, v in r.items()})
    for env in envs:
        env.close()
    print({"animation": str(out), "success": success.tolist(), "steps": lengths.tolist()})


if __name__ == "__main__":
    main()
