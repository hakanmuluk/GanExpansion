"""Build machine-readable comparisons and a concise evidence-based report."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def kind(name,primary):
    if name in primary:return "primary"
    if name.endswith("_noz_single"):return "deterministic_single_strategy"
    for needle,label in [("_d64","data_efficiency"),("_d256","data_efficiency"),("_d1024","data_efficiency"),("_keep0","preservation"),("_keep2","preservation"),("_noz","no_new_latent"),("_fullft","full_finetuning"),("_vanilla","vanilla")]:
        if name.endswith(needle):return label
    return "ablation"


def pct(x):return f"{100*x:.2f}%"


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default="addon_artifacts");p.add_argument("--out",default="ADDONS_RESULTS.md");args=p.parse_args()
    root=Path(args.root);selection=json.loads((root/"selection.json").read_text())
    primary=sum(selection["replicate_runs"].values(),[]);rows=[];raw={}
    for f in sorted((root/"evaluation").glob("*/metrics.json")):
        m=json.loads(f.read_text());name=f.parent.name;raw[name]=m;q=m["quantiles"];g=m["gaussian"];a=m["args"]
        info=json.loads((root/"runs"/name/"config.json").read_text())
        rows.append({"run":name,"skill":m["skill"],"kind":kind(name,primary),"seed":a["seed"],"demonstrations":info.get("training_demonstrations_used",a["episodes"]),"step":m["step"],
                     "rank":m["adapter_config"]["rank"],"actor_trainable":info["counts"]["actor_trainable"],"keep_weight":a["keep_weight"],
                     "success":q["success"],"gaussian_success":g["success"],"both_strategies":q["both_successful_strategies_per_context"],
                     "style_preserving_success":q["style_preserving_success"],"door_agreement":q["base_door_agreement"],
                     "latent_rmse":q.get("closed_loop_latent_rmse"),"latent_correlation":q.get("closed_loop_latent_correlation"),
                     "action_l2_near":q["matched_state_action_l2_near"],"action_l2_far":q["matched_state_action_l2_far"],
                     "disabled_goal_success":m["preservation"]["disabled_goal_success"],"disabled_output_max_error":m["preservation"]["prediction_max_abs_difference"],
                     "native_collision_fraction":q["native_collision_fraction"],"strict_wall_overlap":q["strict_body_wall_overlap_fraction"]})
    aggregates={}
    for skill,names in selection["replicate_runs"].items():
        r=[next(row for row in rows if row["run"]==name) for name in names]
        stats={"training_seeds":[x["seed"] for x in r],"runs":names}
        for key in ["success","gaussian_success","both_strategies","latent_rmse","latent_correlation","action_l2_far"]:
            values=np.array([x[key] for x in r],dtype=float);stats[key+"_mean"]=float(values.mean());stats[key+"_std"]=float(values.std(ddof=1)) if len(values)>1 else 0.
        aggregates[skill]=stats
    summary={"primary_aggregate":aggregates,"runs":rows,"selection":selection,
             "evaluation":"256 held-out contexts, 8 modifier codes per context; Gaussian and shared quantiles; fixed independent base and modifier latents.",
             "statistics":"Seed standard deviations are descriptive. Per-model success confidence intervals resample entire contexts (2000 bootstrap replicates)."}
    (root/"summary.json").write_text(json.dumps(summary,indent=2))
    with (root/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=["# Independent TwoRoom skill addition","","Two separately trained, switchable latent-conditioned adapters modify the pretrained TwoRoom generator. The original generator, mapping, encoder, fixed normalization, and original checkpoint remain unchanged. Composition is not implemented.","", "## Primary held-out results", "", "Each skill uses 2,048 complete demonstrations from 512 task contexts, with four paired styles per context. Each selected model is tested on 256 new contexts with eight Gaussian modifier codes (2,048 rollouts), and separately on eight shared Gaussian quantiles. Checkpoints are selected on validation data; training seeds are 42, 123, and 2026.","", "| Skill | Gaussian success, mean ± seed SD | Both successful strategies per context (quantiles) |", "|---|---:|---:|"]
    for skill,s in aggregates.items():text.append(f"| {skill.title()} | {100*s['gaussian_success_mean']:.2f}% ± {100*s['gaussian_success_std']:.2f}% | {pct(s['both_strategies_mean'])} |")
    text += ["", "The base generator achieves 100% original-goal success and 0% new-skill success on these deliberately nontrivial constraint contexts. This task distribution places the keep-out disk on the original route and places the waypoint away from that route. Results do not establish generalization to arbitrary obstacle layouts.", "", "Every seed's checkpoint is retained under `addon_artifacts/runs/<run>/best.pt`. The convenient `final/*.pt` exports and trajectory figures use the prespecified seed-42 models; they are not selected by test performance. The aggregate above averages all three seeds, while the table below shows each model's performance.", "", "| Skill / training seed | Selected step | Gaussian success | Quantile success | Both strategies | Latent RMSE (actual rollout) |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["kind"]=="primary":text.append(f"| {row['skill']} / {row['seed']} | {row['step']} | {pct(row['gaussian_success'])} | {pct(row['success'])} | {pct(row['both_strategies'])} | {row['latent_rmse']:.3f} |")
    text += ["", "## Iterations and diagnosed failures", "", "| Run | Selected validation step | Validation success | Change / finding |", "|---|---:|---:|---|"]
    descriptions={"keepout_pilot":"Rank-four corrections already learned both detour sides; later selection also considered actual latent recovery.",
                  "waypoint_pilot":"Initial waypoint model stalled or missed the visit; original demonstration handoff was at x=132.",
                  "waypoint_feedback":"Corrected handoff x=124 and stored-path feedback improved execution, but a post-visit progress assumption still caused misses.",
                  "waypoint_history":"Adding the previous observed action did not improve this training recipe and was rejected.",
                  "waypoint_capacity":"Rank eight, larger modulation, output scale/shift, position tracking, and phase-aware finite feedback improved successful execution and latent recovery."}
    for name,description in descriptions.items():
        path=root/"runs"/name/"validation.jsonl"
        if not path.exists():continue
        history=[json.loads(line) for line in path.read_text().splitlines()]
        eligible=[r for r in history if r.get("eligible",True)];best=max(eligible,key=lambda r:r["score"])
        text.append(f"| {name} | {best['step']} | {pct(best['success'])} | {description} |")
    text += ["", "The first full waypoint collection exposed rare failed demonstrations: its handoff could reactivate the base before a wall-side approach was finished. Moving that handoff from x=132 to the cleared wall boundary x=124 produced 100% successful demonstrations in the final collection. A separate feedback bug advanced past the stored waypoint visit even when the actual policy had not visited; the corrected tracker waits for an observed swept visit. A regression test covers this failure. These combined engineering iterations do not isolate the causal contribution of every architectural change.", "", "## Preservation, diversity, and intervention", ""]
    for skill,name in selection["primary_runs"].items():
        m=raw[name];q=m["quantiles"];pr=m["preservation"]
        text += [f"**{skill.title()}, seed 42:** disabled output maximum difference {pr['prediction_max_abs_difference']:.1f}, disabled trajectory maximum difference {pr['trajectory_max_abs_difference']:.1f}, and original/disabled success {pct(pr['original_goal_success'])}/{pct(pr['disabled_goal_success'])}. The original task uses its original 160-step horizon. Active new-skill rollouts use 240 steps.","",f"The new code matches the geometrically measured strategy on {pct(q['strategy_matches_code'])} of style-preserving successes. Actual closed-loop encoder correlation is {q['closed_loop_latent_correlation']:.3f}; teacher-state cycle RMSE is {m['offline'].get('teacher_state_cycle_latent_rmse',float('nan')):.3f}, versus actual rollout RMSE {q['closed_loop_latent_rmse']:.3f}. The latter measures recovery from actually executed trajectories. Codes stay exactly fixed during execution (maximum code change {q['fixed_code_max_change']:.1f}). Resampling codes every eight steps changes skill success to {pct(m['resampled']['success'])}.","",f"At matched replanning states, mean unsquared action L2 modification is {q['matched_state_action_l2_near']:.3f} near the constraint and {q['matched_state_action_l2_far']:.3f} outside its intervention region. Outside-region speed correlation with the base style is {q['outside_intervention_speed_style_correlation']:.3f}. Local intervention is measured empirically; gate-on behavior is not guaranteed to match the base away from a constraint.",""]
        text += [f"Large versus small absolute modifier codes change the measured clearance/approach excursion by {q['within_context_large_minus_small_code_amount_mean']:.2f} pixels on average within the same context and strategy. This continuous control is weaker than the discrete strategy distinction when the paired effect is small. Native collision fraction is {pct(q['native_collision_fraction'])}; strict body/wall overlap fraction is {pct(q['strict_body_wall_overlap_fraction'])}. The strict check accounts for the native environment's permissive doorway margin.",""]
        style_path=root/"audits"/f"{skill}_base_styles.json"
        if style_path.exists():
            style=json.loads(style_path.read_text())
            text += [f"The continuous base doorway style is also retained approximately: active versus original wall-crossing height has mean absolute error {style['base_crossing_y_mae_pixels']:.2f} px (95th percentile {style['base_crossing_y_p95_error_pixels']:.2f} px). Its correlation with the original latent's target doorway offset is {style['original_offset_target_correlation']:.3f}. These gate-on differences are distinct from exact gate-off preservation.",""]
        if skill=="waypoint":text += [f"Waypoint success means an observed visit followed by final goal completion. Earlier traversal of the goal region is allowed. Under the stricter first-goal-entry convention, success is {pct(q['strict_first_goal_entry_skill_success'])}; {pct(q['goal_region_entered_before_waypoint_fraction'])} enter the goal region before visiting. These are different termination definitions, and the stricter result should be used when early goal entry must invalidate an episode.",""]
    text += ["## Independent execution checks", "", "| Skill | Dense disabled-state checks | Disabled max error | Native replay episodes | Native max position error |", "|---|---:|---:|---:|---:|"]
    for skill in selection["primary_runs"]:
        pr=json.loads((root/"audits"/f"{skill}_preservation.json").read_text())
        native=json.loads((root/"audits"/f"{skill}_native.json").read_text())
        deployed=json.loads((root/"audits"/f"{skill}_deployment.json").read_text())
        text.append(f"| {skill} | {pr['randomized_states']:,} | {pr['randomized_max_abs_difference']:.1f} | {native['reference_episodes']} | {native['max_position_error_pixels']:.1f} px |")
        assert native["numpy_constraint_and_goal_agreement"] and deployed["max_action_error"]==0
    text += ["", "Recorded actions are replayed in the actual pinned stable-worldmodel TwoRoom environment, with separate NumPy swept-geometry checks. Failure cases are deliberately included where available; audit-subset success rates are not population estimates. The standalone deployment wrapper also matches the evaluated execution protocol with zero action and position difference in eight enabled/disabled episodes per skill. Frozen generator/encoder tensor fingerprints match before and after training.", "", "## Comparisons and finite-data efficiency", "", "All rows below use the same eight quantiles and held-out contexts. Full fine-tuning updates a cloned base, preserving the original checkpoint file; its disabled path exposes changes to the cloned backbone. Its backbone learning rate is 0.0001 and new-module rate is 0.0005; the keep-loss target remains the immutable original generator. A bank retaining the original generator alongside complete fine-tuned per-skill copies could also restore the original exactly; the adapter approach reduces per-skill actor storage. Changes in FT outputs should not be equated with a loss of original-goal success when that success remains high. No-new-latent adapters receive a constant code and cannot express multiple strategies at fixed task and base latent. Vanilla removes synthetic code-label and known-code reconstruction supervision, using inferred demonstration codes for feedback. Main runs use that auxiliary supervision in addition to BicycleGAN objectives.","", "| Run | Demos | Trainable actor | Success | Both strategies | Far action L2 | Disabled output error |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["kind"]!="primary":text.append(f"| {r['run']} | {r['demonstrations']} | {r['actor_trainable']:,} | {pct(r['success'])} | {pct(r['both_strategies'])} | {r['action_l2_far']:.3f} | {r['disabled_output_max_error']:.3g} |")
    deterministic=next((r for r in rows if r["skill"]=="waypoint" and r["kind"]=="no_new_latent"),None)
    if deterministic:
        text += ["",f"The all-mode waypoint no-latent control reaches {pct(deterministic['success'])} task success with {pct(deterministic['both_strategies'])} within-context two-strategy coverage. Task completion alone therefore does not show that a modifier latent is useful or used. The latent-conditioned models add independent, selectable approaches while retaining the original code; this benefit is established by geometric coverage and actual rollout recovery rather than by assuming they must outperform deterministic controls on goal completion."]
    sweep=sorted([r for r in rows if r["skill"]=="keepout" and (r["kind"]=="preservation" or r["run"]==selection["primary_runs"]["keepout"])],key=lambda r:r["keep_weight"])
    if len(sweep)>=2:
        low,high=sweep[0],sweep[-1]
        text += ["",f"In the keep-out sweep, increasing the keep weight from {low['keep_weight']:g} to {high['keep_weight']:g} changes success from {pct(low['success'])} to {pct(high['success'])}, and outside-region action L2 from {low['action_l2_far']:.4f} to {high['action_l2_far']:.4f}. The reduction in unnecessary motion is modest over this range, while the stronger penalty also reduces clearance and successful avoidance. This is a measured tradeoff, not a guarantee of locality."]
    text += ["", "The additional `noz_single` controls keep the same 512 contexts but train on their positive-side demonstrations only: 1,024 unique demonstrations from the available 2,048-episode prefix. They distinguish learning one valid deterministic strategy from expressing multiple strategies at a fixed task and base latent. Poor no-latent performance on the contradictory paired dataset does not establish that a deterministic policy cannot learn a single strategy.", "", "Training budgets are finite nested prefixes: 64, 256, 1,024, and 2,048 demonstrations. Feedback uses only those stored paths; it does not query the expert for additional actions. These are demonstration-efficiency comparisons with known simulator dynamics and rollout regularization; they do not measure total interaction sample complexity. Data-efficiency comparisons use one training seed and are exploratory. The keep-loss weights test whether preserving the base too strongly prevents a necessary modifier. Other architectural/loss changes and unsuccessful attempts are retained in run configurations and validation logs.","", "## Artifacts and reproduction", "", "- [Methods, commands, and deployment API](ADDONS_README.md)", "- [Machine-readable summary](addon_artifacts/summary.json) and [comparison CSV](addon_artifacts/comparison.csv)", "- [Same-task trajectory overlays](addon_artifacts/figures/trajectories.png)", "- [Training, latent, and intervention diagnostics](addon_artifacts/figures/diagnostics.png)", "- [Data-efficiency and comparison plots](addon_artifacts/figures/comparisons.png)", "- [Keep-out animation](addon_artifacts/figures/keepout.gif) and [waypoint animation](addon_artifacts/figures/waypoint.gif)", "- [Keep-out checkpoint](addon_artifacts/final/keepout.pt) and [waypoint checkpoint](addon_artifacts/final/waypoint.pt)", "", "Each evaluation directory includes complete metrics, the frozen pre-test checkpoint fingerprint, and the exact evaluation command. Selected models also have native-environment replay audits, dense exact-preservation checks, and saved trajectories. The delivery receipt verifies local files against SHA-256 hashes and confirms the earlier TwoRoom and Cube deliveries remain intact."]
    Path(args.out).write_text("\n".join(text)+"\n")


if __name__=="__main__":main()
