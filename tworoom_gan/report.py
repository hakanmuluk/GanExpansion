"""Write the final research report from executed, saved metrics."""

import argparse
import json
from pathlib import Path
import shlex


def pct(x):
    return f"{100*x:.2f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="RESULTS.md")
    args = p.parse_args()
    root = Path("artifacts")
    quality = json.loads((root / "data_v2/quality.json").read_text())
    runtime = json.loads((root / "runtime.json").read_text())
    manifest = json.loads((root / "final/manifest.json").read_text())
    tests = {seed: json.loads((root / f"evaluation/test_seed{seed}/metrics.json").read_text()) for seed in [42, 123, 2026]}
    comparison = json.loads((root / "comparison/validation_comparison.json").read_text())
    main_metrics = tests[42]
    independent = main_metrics["independent_gaussian_latents"]
    reconstruction = main_metrics["encoded_demonstration_rollouts"]
    cuda_api = json.loads((root / "final/policy_api_cuda.json").read_text())
    cpu_api = json.loads((root / "final/policy_api_cpu.json").read_text())
    lines = [
        "# TwoRoom baseline: executed results", "",
        f"The selected style-modulated trajectory policy achieves **{pct(independent['success_rate'])} held-out task success** with independent Gaussian latents. It was actually trained and evaluated on the Colab {runtime['gpu'].split(',')[0]}. No new-skill adapters or composition method are implemented.", "",
        "**Important scope of the result:** the strongest recipe uses auxiliary supervision from the synthetic expert's known route, crossing offset, and speed. It retains BicycleGAN reconstruction/cycle objectives, a learned whole-episode encoder, KL, and conditional adversarial training. The original objectives alone did not produce equally reliable latent-controlled behavior in the attempted configurations. This is a stabilized baseline, not evidence that vanilla BicycleGAN alone solves the task.", "",
        "## Held-out results", "",
        "Each seed uses 256 unseen start/goal pairs, 32 fixed Gaussian codes shared across pairs, and a separate set of 32 independently sampled codes per pair. Checkpoints and the recipe were selected using validation only; the test split was not used for training or checkpoint selection. The 160-step timeout, 16-pixel goal tolerance, and original SWM collision behavior define task success.", "",
        "| Training seed | Selected step | Shared-code success | Independent-code success | Both routes per task (shared) | Same-code route agreement | Actual latent RMSE (independent) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, m in tests.items():
        i = m["independent_gaussian_latents"]
        lines.append(f"| {seed} | {m['checkpoint_step']} | {pct(m['success_rate'])} | {pct(i['success_rate'])} | {pct(m['both_modes_covered_fraction'])} | {pct(m['same_latent_route_agreement_across_states'])} | {i['rollout_latent_rmse_all']:.4f} |")
    lines += ["",
        f"For the primary seed, the independent panel contains {independent['rollouts']:,} executed episodes. Its descriptive Wilson 95% interval is {pct(independent['success_95ci_wilson'][0])} to {pct(independent['success_95ci_wilson'][1])}; grouping-aware bootstrap intervals are also saved. Both panels reuse start/goal pairs, and the shared panel additionally reuses latent codes. These are not IID draws over initial configurations; the shared panel primarily measures control consistency and coverage.", "",
        f"Actual-rollout latent correlations (primary independent panel): {', '.join(f'{v:.4f}' for v in independent['rollout_latent_correlations'])}. Route requests match successful executed routes in {pct(independent['intended_route_accuracy_successful'])} of cases. Crossing-offset RMSE is {independent['crossing_offset_rmse_pixels_successful']:.3f} pixels, and mean executed-speed MAE is {independent['mean_speed_mae_successful']:.4f}. Within-task speed standard deviation across successful codes is {independent['mean_within_task_speed_std']:.4f}, demonstrating variation beyond route identity.", "",
        f"The original SWM replay audit covers {main_metrics['reference_audit']['rollouts']} stratified primary test rollouts, with {main_metrics['reference_audit']['max_position_error_pixels']:.6f} pixels maximum discrepancy and exact agreement in success and crossed door. Public policy API checks, replanning from original SWM observations with the training expert disabled, succeed in {pct(cuda_api['original_environment_success_rate'])} of {cuda_api['episodes']} CUDA episodes and {pct(cpu_api['original_environment_success_rate'])} of {cpu_api['episodes']} CPU episodes.", "",
        "## Reconstruction, latent consistency, and ablations", "",
        f"All {reconstruction['episodes']:,} held-out demonstrations are encoded and reconstructed through actual policy rollouts: success {pct(reconstruction['success_rate'])}, demonstrated-route matching {pct(reconstruction['route_match_fraction_all'])}, time-aligned position RMSE {reconstruction['time_aligned_position_rmse_pixels']:.3f} pixels. Offline chunk reconstruction L1 is {main_metrics['action_reconstruction_l1']:.5f} in normalized action units.", "",
        f"With a zero constant latent, success is {pct(main_metrics['constant_zero_latent']['success_rate'])} and both-route coverage for the same task is {pct(main_metrics['constant_zero_latent']['both_modes_covered_fraction'])}. Resampling the latent every replan changes success to {pct(main_metrics['resample_latent_every_replan']['success_rate'])}. Keeping z fixed therefore matters for coherent behavior.", "",
        f"Teacher-state cycle RMSE is {main_metrics['teacher_state_latent_rmse']:.4f}; actual-rollout RMSE is reported separately above. Teacher-state sequences use a demonstration's states with newly generated actions and can be physically inconsistent. They must not be confused with actual generated trajectories. Experiments 02-03 demonstrate why optimizing latent recovery alone can still produce unsuccessful control.", "",
        f"Under the original simulator, the primary independent panel has collision episodes in {pct(independent['collision_episode_fraction'])} of trials. The stricter circular-body rectangle-overlap check reports {pct(independent['strict_body_wall_overlap_episode_fraction'])}. The latter differs from upstream's permissive doorway margin; it is included explicitly rather than conflated with simulator success.", "",
        "## Common validation comparison", "",
        "All retained models below use the same 128 validation start/goal pairs and the same shared/independent Gaussian codes. Different data versions retain identical start/goal pairs; v2 additionally varies crossing position. These are recipe comparisons with multiple changes, not isolated causal ablations of a single loss.", "",
        "| Retained configuration | Step | Independent success | Both routes (shared) | Actual latent RMSE | Offline action L1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, d in comparison.items():
        lines.append(f"| {name} | {d['step']} | {pct(d['independent']['success_rate'])} | {pct(d['shared']['both_modes_covered_fraction'])} | {d['independent']['rollout_latent_rmse_all']:.4f} | {d['offline']['action_reconstruction_l1']:.5f} |")
    lines += ["", "## Data and architecture", "",
        f"Dataset v2 contains {quality['splits']['train']['episodes']:,} successful training episodes, {quality['splits']['val']['episodes']:,} validation episodes, and {quality['splits']['test']['episodes']:,} test episodes. Each start/goal pair has two routes at two sampled speeds; crossing offset varies independently within +/-12 pixels. Splitting is by complete start/goal pair, preventing alternate-route leakage. Full states/actions, lengths, style metadata, starts, and goals are retained. Route labels are checked against actual wall crossings. The GPU simulator matches {quality['reference_parity']['random_transitions']:,} original SWM transitions with {quality['reference_parity']['max_position_error_pixels']} pixels maximum error.", "",
        "G has a 2D Gaussian latent, 128D mapping/style network, four 256-wide residual blocks with per-layer scale/shift modulation, and a 16x2 action output. E uses masked temporal convolutions over complete episodes, with pooled speed statistics and a diagonal Gaussian posterior. D is spectral-normalized and conditioned on state, action chunk, and valid-step mask. No BatchNorm or mutable action-normalization statistics are used. State normalization is pixel/112-1. One z is sampled per attempt, 16 actions are predicted, and 8 are executed before replanning.", "",
        "The auxiliary recipe adds exact reflection symmetry, a derived route-sign feature, encoder style targets, and prior-sample expert targets. The Gaussian coordinate transform respects the expert's sampling distribution. Physical latent cycles and expert feedback at visited training states supplement the original objectives. The deployed generator calls no expert, planner, CDF action transform, or safety filter. See `EXPERIMENTS.md` for the failures and improvements leading to this recipe.", "",
        "## Artifacts and reproduction", "",
        "- [Reusable setup/training/evaluation instructions](README.md)",
        "- [Detailed experiment record](EXPERIMENTS.md)",
        "- [Selected base G/E checkpoint](artifacts/final/base_policy.pt)",
        "- [Full final optimizer/G/E/D checkpoint for resuming](artifacts/final/training_resume.pt)",
        "- [Source/runtime/checkpoint SHA-256 manifest](artifacts/final/manifest.json)",
        "- [Primary held-out numerical metrics](artifacts/evaluation/test_seed42/metrics.json)",
        "- [Common validation comparison](artifacts/comparison/validation_comparison.csv)",
        "- [Dataset quality and reference parity](artifacts/data_v2/quality.json)",
        "- [Final verification](artifacts/final/verification.json)", "",
        "![Same-task generated and expert trajectories](artifacts/final/figures/generated_trajectories.png)", "",
        "![Losses and closed-loop validation across configurations](artifacts/final/figures/training_curves.png)", "",
        "![Actual latent reconstruction](artifacts/final/figures/latent_control_and_recovery.png)", "",
        "![Geometric and speed control](artifacts/final/figures/style_control_accuracy.png)", "",
        "[Animated original-renderer rollouts](artifacts/final/figures/policy_rollouts.gif)", "",
        "## Practical scope and limitations", "",
        "These results cover a fixed two-door geometry and state observations, with unseen continuous starts/goals and both travel directions. They do not establish generalization to new layouts, pixel observations, real robots, or new tasks. The strongest recipe uses known synthetic expert style metadata; performance without this extra supervision is explicitly reported rather than assumed. No continual-learning, expansion, LoRA, modifier, or composition claim is tested in this stage. The reference simulator's doorway collision convention and success tolerance are preserved. Finite-seed results and confidence intervals do not prove universal reliability.", "",
        "All experiments, data generation, training, evaluation, and plotting ran on Colab. Runtime versions are recorded in `artifacts/runtime.json`. The local workspace contains persistent copies because Drive was not mounted. G can be frozen for the next stage; this stage implements only the baseline.", "",
        "## Executed commands", "",
    ]
    for run in sorted((root / "runs").iterdir()):
        if run.name.startswith("smoke") or not (run / "process.json").exists():
            continue
        command = json.loads((run / "process.json").read_text())["command"]
        lines += [f"### {run.name}", "", "```bash", shlex.join(command), "```", ""]
        if (run / "STOPPED_EARLY.json").exists():
            stopped = json.loads((run / "STOPPED_EARLY.json").read_text())
            lines += [f"Intentionally stopped after saving step {stopped['last_checkpoint_step']}: {stopped['reason']}", ""]
    lines += [f"Reference environment: [stable-worldmodel commit {manifest['reference_commit'][:12]}](https://github.com/galilai-group/stable-worldmodel/tree/{manifest['reference_commit']}/stable_worldmodel/envs/two_room).", ""]
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
