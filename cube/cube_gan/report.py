"""Build the results document and grouped-context uncertainty from saved evidence."""
import csv
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def percent(value):return f"{100*value:.2f}%" if value is not None else "n/a"


def context_intervals(records):
    contexts=sorted({r["context"] for r in records})
    success=[];coverage=[]
    for c in contexts:
        rs=[r for r in records if r["context"]==c]
        success.append(np.mean([r["released_success"] for r in rs]))
        routes={r["route"] for r in rs if r["released_success"]}
        coverage.append(-1 in routes and 1 in routes)
    ids=np.random.default_rng(1977).integers(len(contexts),size=(2000,len(contexts)))
    return {"released_success_95":np.percentile(np.array(success)[ids].mean(1),[2.5,97.5]).tolist(),
            "both_routes_95":np.percentile(np.array(coverage)[ids].mean(1),[2.5,97.5]).tolist(),
            "unit":"full initial simulator context; its latent attempts stay grouped","bootstrap_replicates":2000}


def main():
    root=Path("artifacts")
    assert (root/"finish_completed.json").exists(),"Finalize evaluations first"
    names=["test_seed42","test_seed43","test_seed44","baseline","shared_seed42","resample_seed42","zero_seed42","reconstruction_seed42"]
    panels={name:read(root/"evaluation"/name/"summary.json") for name in names}
    records=read(root/"evaluation/test_seed42/records.json")
    intervals=context_intervals(records)
    manifests={str(s):read(root/("final/manifest.json" if s==42 else f"final/seed{s}/manifest.json")) for s in [42,43,44]}
    validation={name:read(root/"validation"/name/"summary.json") for name in ["bicycle_seed42","aux_seed42","aux_seed43","aux_seed44"]}
    quality=read(root/"data/quality.json");audit=read(root/"data/audit.json");verification=read(root/"final/verification.json")
    report={"panels":panels,"primary_context_bootstrap":intervals,"manifests":manifests,"validation":validation,
            "data_quality":quality,"data_audit":audit,"verification":verification,"recipe_frozen":read(root/"recipe_frozen.json")}
    (root/"final/metrics.json").write_text(json.dumps(report,indent=2))
    with (root/"final/comparison.csv").open("w",newline="") as f:
        keys=["panel","attempts","native_success","released_success","both_routes_per_context","route_matches_latent_given_success","closed_loop_latent_rmse","within_context_speed_spearman","within_context_side_amplitude_spearman"]
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for name,d in panels.items():writer.writerow({"panel":name,**{k:d.get(k) for k in keys[1:]}})
    lines=["# OGBenchCube baseline: executed results","",
           "The learned policy was trained and evaluated on the Colab A100 using the actual single-cube `swm/OGBCube-v0` MuJoCo environment. It predicts 16 actions, executes eight, and keeps one 2D Gaussian latent fixed per episode. Only the base generative policy is implemented.","",
           "The selected recipe uses explicit auxiliary expert-style supervision. The original BicycleGAN objectives alone produced successful placements but largely collapsed latent control. Joint recovery refinement reduced success and was rejected. Encoder-only calibration subsequently improved physical latent recovery with generator hashes verified unchanged.","",
           "## Held-out evaluation","",
           "Each seed uses all 150 held-out simulator contexts: 20 for each of the five native goals and 50 continuous randomized transfers. Eight independent Gaussian codes per context give 1,200 actual episodes per seed. Native success is entering the 4cm goal radius; released-and-settled success additionally requires release, low contact and cube speed, table placement, and end-effector clearance for ten consecutive steps.","",
           "| Training seed | G step | Native goal | Released and settled | Both routes per context | Requested route match, given success | Actual latent RMSE (z₀, z₁) |",
           "|---|---:|---:|---:|---:|---:|---|" ]
    for seed in [42,43,44]:
        d=panels[f"test_seed{seed}"];rmse=d["closed_loop_latent_rmse"]
        lines.append(f"| {seed} | {manifests[str(seed)]['step']} | {percent(d['native_success'])} | {percent(d['released_success'])} | {percent(d['both_routes_per_context'])} | {percent(d['route_matches_latent_given_success'])} | {rmse[0]:.3f}, {rmse[1]:.3f} |")
    primary=panels["test_seed42"];ci=intervals["released_success_95"]
    lines += ["",f"For the primary seed, the context-bootstrap 95% interval for released success is {percent(ci[0])}–{percent(ci[1])}. Attempts within a context are correlated; these are grouped intervals, not IID episode intervals. All seeds share data and evaluation draws. Three seeds do not establish universal reliability.","",
              "## Primary results by task","","| Task | Attempts | Native goal | Released and settled |","|---|---:|---:|---:|"]
    for key,d in primary["per_task"].items():lines.append(f"| {'Random transfer' if key=='0' else 'Native goal '+key} | {d['n']} | {percent(d['native_success'])} | {percent(d['released_success'])} |")
    lines += ["",f"Released placement within a stricter 2cm radius occurs in {percent(primary['released_within_2cm'])} of all primary attempts. Mean final cube-goal distance among released successes is {100*primary['mean_final_distance_success']:.2f}cm. The native 4cm benchmark tolerance is retained; these results do not imply millimeter placement accuracy.","",
              "## Latent control, reconstruction, and ablations","",
              f"Within-context speed/episode-length Spearman correlation is {primary['within_context_speed_spearman']:.3f}; larger z₁ requests faster execution. Within the same context and carry side, |z₀| versus physical carry-amplitude correlation is {primary['within_context_side_amplitude_spearman']:.3f}. These measures expose continuous variation beyond route identity.","",
              "| Panel | Actual attempts | Released and settled | Both routes per context |","|---|---:|---:|---:|"]
    for name in ["baseline","shared_seed42","resample_seed42","zero_seed42","reconstruction_seed42"]:
        d=panels[name];lines.append(f"| {name} | {d['attempts']} | {percent(d['released_success'])} | {percent(d['both_routes_per_context'])} |")
    reconstruction=panels["reconstruction_seed42"]
    lines += ["",f"Reconstruction encodes all {reconstruction['attempts']} retained demonstrations from the paired held-out set, then executes G at the inferred mean code from the corresponding saved initial state. A demonstration context is retained only when all four expert styles succeed. Demonstrated-route matching is {percent(reconstruction['demonstrated_route_match'])}; mean time-aligned cube-position RMSE is {100*reconstruction['mean_time_aligned_cube_rmse_m']:.2f}cm. All contexts remain in the ordinary test panels, including those with expert failures.","",
              f"Primary offline masked reconstruction loss is {primary['offline']['reconstruction_loss']:.4f} (Cartesian/yaw L1 + 0.15 gripper BCE). Teacher-state cycle RMSE is {primary['offline']['teacher_state_cycle_rmse']}; actual closed-loop recovery is reported separately above. Teacher-state actions at recorded states can be physically inconsistent. The resampling ablation has no single episode latent, so single-code recovery is intentionally undefined there.","",
              "Carry direction is fitted from the lifted cube at 20–80% task progress, using signed lateral amplitude and a 3mm dead zone. Both-route coverage counts only released successes. Eight fair random route draws have a maximum expected sampled coverage of 99.22% even before control errors; the metric is sensitive to route failures and near-straight short transfers.","",
              "## Common full-validation comparison","",
              "All rows below use the same 100 contexts and eight independent Gaussian codes. These are recipe comparisons, not isolated causal ablations.","",
              "| Configuration | Released and settled | Both routes | Physical latent RMSE (z₀, z₁) |","|---|---:|---:|---|"]
    for name,d in validation.items():
        rm=d["closed_loop_latent_rmse"];lines.append(f"| {name} | {percent(d['released_success'])} | {percent(d['both_routes_per_context'])} | {rm[0]:.3f}, {rm[1]:.3f} |")
    lines += ["", "Encoder calibration selects weights using actual frozen-G validation rollouts. Generator weights, normalizers, and its outputs remain unchanged. Test results are not used to train or select checkpoints. All training and failed refinements are recorded in `EXPERIMENTS.md` and `artifacts/runs/`.","",
              "## Artifacts and reproduction","",
              "- [Setup, environment details, and reproduction commands](README.md)",
              "- [Experiment record, including failed refinement](EXPERIMENTS.md)",
              "- [Portable selected G/E weights](artifacts/final/base_policy.pt)",
              "- [Full selected G/E/D and optimizer checkpoint](artifacts/final/training_resume.pt)",
              "- [All numerical panels and grouped uncertainty](artifacts/final/metrics.json)",
              "- [Checkpoint provenance and hashes](artifacts/final/manifest.json)",
              "- [Public policy API verification and runtime](artifacts/final/verification.json)",
              "- [Dataset quality](artifacts/data/quality.json) and [split/diversity audit](artifacts/data/audit.json)","",
              "![Actual expert routes](artifacts/final/figures/expert_routes.png)","",
              "![Held-out learned routes](artifacts/final/figures/learned_routes.png)","",
              "![Lift, carry, and release](artifacts/final/figures/carry_height.png)","",
              "![Latent control and recovery](artifacts/final/figures/latent_control.png)","",
              "![Training history](artifacts/final/figures/training.png)","",
              "[Original MuJoCo animation](artifacts/final/figures/policy_rollouts.gif)","",
              "## Scope","",
              "The dataset has 3,732/364/544 complete successful train/validation/test demonstrations, split by full simulator context. All 1,000/100/150 contexts are retained separately. State hashes have zero overlap between splits, and actual expert carry directions agree with labels in 100% of retained demonstrations. The native goal arrangements occur in training; randomized transfers test new positions from the same distribution. This does not test new objects, layouts, pixel observations, real robots, skill expansion, adapters, or composition.","",
              f"The exported CPU policy API was exercised with the training expert disabled: {verification['max_api_action_error']:.1e} maximum action discrepancy from direct G, fixed episode latents verified. Task success for that small API check is reported separately in its JSON. Formal held-out metrics come from the larger panels above.","",
              "Source, data, checkpoints, logs, complete evaluated trajectories, figures, and numerical results are persisted locally. Delivery hashes and the independent local receipt document the copied files. The existing TwoRoom delivery is preserved.",""]
    Path("RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"report":"RESULTS.md","primary_released_success":primary["released_success"],"interval":ci}))


if __name__=="__main__":main()
