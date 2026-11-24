"""Standalone publication-quality PNG figures (no browser dependency)."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

from .env import TASK

COLORS = ["#2864b4", "#e08427"]


def room(ax):
    intervals = [(14, 35), (77, 147), (189, 210)]
    for lo, hi in intervals:
        ax.add_patch(Rectangle((107, lo), 10, hi - lo, color="#343a40", zorder=4))
    ax.add_patch(Rectangle((14, 14), 196, 196, fill=False, linewidth=2, edgecolor="#343a40"))
    ax.set(xlim=(10, 214), ylim=(214, 10), aspect="equal")
    ax.set_xticks([28, 112, 196])
    ax.set_yticks([28, 112, 196])
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dataset_figure(data_dir, out):
    data = np.load(Path(data_dir) / "train.npz")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    room(axes[0])
    axes[0].scatter(*data["start"][::16].T, s=4, alpha=0.3, color="#257c8e", label="Start")
    axes[0].scatter(*data["goal"][::16].T, s=4, alpha=0.3, color="#b64a77", label="Goal")
    axes[0].set_title("Diverse starts and goals", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower center")
    room(axes[1])
    for i in range(4):
        length = int(data["lengths"][i])
        pos = (data["states"][i, :length + 1, :2] + 1) * 112
        axes[1].plot(*pos.T, color=COLORS[int(data["mode"][i])], linewidth=2, linestyle="-" if i % 2 else "--", alpha=0.8)
    axes[1].scatter(*data["start"][0], color="#257c8e", s=40, zorder=7)
    axes[1].scatter(*data["goal"][0], color="#2b8c53", marker="*", s=110, zorder=7)
    axes[1].set_title("Same task: two routes, two speeds", fontsize=11)
    for mode in [0, 1]:
        axes[2].hist(data["lengths"][data["mode"] == mode], bins=np.arange(0, 121, 5), alpha=0.6, color=COLORS[mode], label=["Upper door", "Lower door"][mode])
    axes[2].set(xlabel="Episode length (steps)", ylabel="Successful demonstrations", title="Balanced route distribution")
    axes[2].legend(fontsize=8)
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Training data: {len(data['lengths']):,} successful complete episodes", fontsize=14)
    save(fig, out / "dataset_quality.png")


def trajectory_figure(rollout_path, expert_path, out, latent_count=None):
    r, expert = np.load(rollout_path), np.load(expert_path)
    starts = r["start"]
    if latent_count is None:
        latent_count = int(np.argmax(np.any(starts != starts[0], axis=1))) or len(starts)
    pairs = len(starts) // latent_count
    chosen = np.linspace(0, pairs - 1, min(6, pairs), dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), constrained_layout=True)
    for ax, pair in zip(axes.flat, chosen):
        room(ax)
        sl = slice(pair * latent_count, (pair + 1) * latent_count)
        for i in range(sl.start, sl.stop):
            length = int(r["lengths"][i])
            color = COLORS[int(r["mode"][i])] if r["success"][i] and r["mode"][i] >= 0 else "#acb2b8"
            ax.plot(*r["positions"][i, :length + 1].T, color=color, alpha=0.62, linewidth=1.2)
        for j in range(pair * 4, pair * 4 + 4):
            length = int(expert["lengths"][j])
            pos = (expert["states"][j, :length + 1, :2] + 1) * 112
            ax.plot(*pos.T, color="#202428", linestyle=":", linewidth=1, alpha=0.6)
        ax.scatter(*r["start"][sl.start], color="#257c8e", s=35, edgecolor="white", zorder=8)
        ax.scatter(*r["goal"][sl.start], color="#26934f", marker="*", s=110, edgecolor="white", zorder=8)
        ax.add_patch(Circle(r["goal"][sl.start], 16, facecolor="#26934f", edgecolor="#26934f", alpha=0.12, zorder=1))
        succeeded = int(r["success"][sl].sum())
        covered = len(set(r["mode"][sl][r["success"][sl]]) - {-1})
        ax.set_title(f"Unseen task {pair + 1}: {succeeded}/{latent_count} success, {covered}/2 routes", fontsize=10)
    handles = [Line2D([0], [0], color=COLORS[0], label="Generated: upper door"),
               Line2D([0], [0], color=COLORS[1], label="Generated: lower door"),
               Line2D([0], [0], color="#202428", linestyle=":", label="Expert demonstrations"),
               Line2D([0], [0], color="#26934f", marker="*", linestyle="None", label="Goal (shaded radius: 16px)")]
    if not r["success"].all():
        handles.append(Line2D([0], [0], color="#acb2b8", label="Failed rollout"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("Closed-loop generated trajectories: same start/goal, different fixed Gaussian latents", fontsize=13)
    save(fig, out / "generated_trajectories.png")


def latent_figure(rollout_path, out):
    r = np.load(rollout_path)
    z, recovered, mode, success = r["z"], r["recovered_z"], r["mode"], r["success"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for m in [0, 1]:
        valid = success & (mode == m)
        axes[0].scatter(z[valid, 0], z[valid, 1], color=COLORS[m], s=12, alpha=0.35, label=["Upper door", "Lower door"][m])
    axes[0].set(xlabel="Episode latent z[0]", ylabel="Episode latent z[1]", title="Latent-dependent route selection")
    axes[0].legend(fontsize=8)
    for j in [0, 1]:
        ax = axes[j + 1]
        ax.scatter(z[success, j], recovered[success, j], c=np.array(COLORS)[mode[success]], s=5, alpha=0.2)
        lo, hi = min(z[:, j].min(), recovered[:, j].min()), max(z[:, j].max(), recovered[:, j].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set(xlabel=f"Sampled z[{j}]", ylabel=f"Recovered E(rollout)[{j}]", title=f"Actual-rollout latent recovery: axis {j}")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "latent_control_and_recovery.png")


def training_figure(runs_dir, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)
    gan_fig, gan_axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    found = False
    for run in sorted(Path(runs_dir).iterdir()):
        if not (run / "validation.jsonl").exists() or run.name.startswith("smoke"):
            continue
        found = True
        vals = [json.loads(x) for x in (run / "validation.jsonl").read_text().splitlines() if x]
        train = [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines() if x]
        steps = [x["step"] for x in train]
        label = {"01_bicyclegan": "01 basic", "02_moments_latent": "02 speed moments", "03_physical_cycle": "03 physical cycle", "04_auxiliary_style": "04 aux (seed 42)", "05_auxiliary_seed123": "05 aux (seed 123)", "06_auxiliary_seed2026": "06 aux (seed 2026)", "07_label_free_prior": "07 prior match"}.get(run.name, run.name)
        axes[0, 0].plot(steps, [x["rec"] for x in train], label=label)
        axes[0, 1].plot(steps, [x["latent"] for x in train], label=label)
        axes[0, 2].plot([x["step"] for x in vals], [x["rollout_latent_rmse_all"] for x in vals], marker=".", label=label)
        axes[1, 0].plot([x["step"] for x in vals], [x["success_rate"] for x in vals], marker=".", label=label)
        axes[1, 1].plot([x["step"] for x in vals], [x["both_modes_covered_fraction"] for x in vals], marker=".", label=label)
        axes[1, 2].plot([x["step"] for x in vals], [x["same_latent_route_agreement_across_states"] for x in vals], marker=".", label=label)
        adversarial = [x for x in train if x.get("gan_weight", 0) > 0]
        gan_axes[0].plot([x["step"] for x in adversarial], [x["d_loss"] for x in adversarial], label=label)
        gan_axes[1].plot([x["step"] for x in adversarial], [x["adv"] for x in adversarial], label=label)
        gan_axes[2].plot(steps, [x["kl"] for x in train], label=label)
    for ax, title in zip(axes.flat, ["Action reconstruction (training L1)", "Teacher-state latent cycle (training L1)", "Actual-rollout latent recovery (val RMSE)", "Closed-loop success (validation)", "Tasks covering both routes (validation)", "Same-latent route agreement (validation)"]):
        ax.set(title=title, xlabel="Optimizer steps")
        ax.spines[["top", "right"]].set_visible(False)
        if found:
            ax.legend(fontsize=7)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_ylim(0, 1.03)
    axes[1, 1].set_ylim(0, 1.03)
    axes[1, 2].set_ylim(0, 1.03)
    save(fig, out / "training_curves.png")
    for ax, title in zip(gan_axes, ["Discriminator hinge loss + lazy R1", "Generator adversarial loss", "Posterior KL regularizer"]):
        ax.set(title=title, xlabel="Optimizer steps")
        ax.spines[["top", "right"]].set_visible(False)
        if found:
            ax.legend(fontsize=7)
    save(gan_fig, out / "gan_stability.png")


def style_accuracy_figure(rollout_path, out):
    r = np.load(rollout_path)
    if "intended_offset" not in r.files:
        return
    ok = r["success"] & (r["mode"] >= 0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    c = np.array(COLORS)[r["mode"][ok]]
    axes[0].scatter(r["intended_offset"][ok], r["observed_crossing_offset"][ok], c=c, s=5, alpha=0.25)
    axes[0].plot([-12, 12], [-12, 12], "k--", linewidth=1)
    axes[0].set(xlabel="Latent-requested crossing offset (px)", ylabel="Actual crossing offset (px)", title="Geometric variation within each route")
    axes[1].scatter(r["intended_speed"][ok], r["mean_speed"][ok], c=c, s=5, alpha=0.25)
    axes[1].plot([0.55, 1], [0.55, 1], "k--", linewidth=1)
    axes[1].set(xlabel="Latent-requested action magnitude", ylabel="Mean executed action magnitude", title="Episode-level speed control")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "style_control_accuracy.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/data")
    p.add_argument("--evaluation")
    p.add_argument("--split", default="test")
    p.add_argument("--runs", default="artifacts/runs")
    p.add_argument("--out", default="artifacts/figures")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "medium"})
    dataset_figure(args.data, out)
    if Path(args.runs).exists():
        training_figure(args.runs, out)
    if args.evaluation:
        trajectory_figure(Path(args.evaluation) / "rollouts.npz", Path(args.data) / f"{args.split}.npz", out)
        latent_figure(Path(args.evaluation) / "rollouts.npz", out)
        independent = Path(args.evaluation) / "independent_rollouts.npz"
        style_accuracy_figure(independent if independent.exists() else Path(args.evaluation) / "rollouts.npz", out)


if __name__ == "__main__":
    main()
