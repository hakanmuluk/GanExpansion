"""Build the final scientific tables/figure from saved, complete measurements."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyze import bounded_mean_interval


ROOT = Path('cube_composition_artifacts')
KINDS = ['supported_easy', 'supported_conflict', 'coupled_ood']
LABELS = {
    'base': 'Base', 'keepout_only': 'Keep-out only', 'waypoint_only': 'Waypoint only',
    'naive_both': 'Simultaneous adapters', 'action_average': 'Whole-action average',
    'agreement_adam_keepout_fixed': 'Adam, execute K, fixed',
    'agreement_adam_waypoint_fixed': 'Adam, execute W, fixed',
    'agreement_adam_waypoint_warm': 'Adam, execute W, warm',
    'agreement_random_waypoint_warm': 'Random, execute W, warm',
    'agreement_cem_waypoint_warm': 'CEM, execute W, warm',
    'agreement_hybrid_waypoint_warm': 'Hybrid, execute W, warm',
    'native_select_full4': 'Native full-horizon family selection',
    'native_latent_256': 'Native task latent W, 256',
    'native_trajectory_256': 'Native pose agreement W, 256',
    'projected_waypoint_selected': 'Projected waypoint, fixed',
    'common_pair_selected': 'Common pair, fixed',
    'grip_safe_blend': 'Gripper-preserving blend',
    'common_latent_selected': 'Common pair, Adam warm',
    'common_native_256': 'Common pair, native task search 256',
    'direct_action_256': 'Direct action CEM, 256',
    'direct_action_CEM272': 'Direct action CEM, 272',
    'waypoint_encoder_ablation': 'W: encoder bank, Adam 8',
    'waypoint_encoder_prefix_ablation': 'W: causal prefix, Adam 8',
    'common_encoder_ablation': 'Common: encoder bank, Adam 8',
    'common_encoder_prefix_ablation': 'Common: causal prefix, Adam 8',
}
CORE = list(LABELS)[:11]
EXTENSIONS = list(LABELS)[11:21]


def load(name):
    return json.loads((ROOT / name).read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]]) + '\n'


def main():
    main_data = load('analyses/evidence_main/metrics.json')['groups']
    data = {(a['kind'], a['method']): a for a in main_data}
    ablations = load('analyses/evidence_ablation/metrics.json')['groups']
    abl = {(a['kind'], a['method']): a for a in ablations}
    validation = {(a['kind'], a['method']): a for a in load('analyses/evidence_validation/metrics.json')['groups']}
    text = ['## Quantitative results\n',
            'Joint success is a percentage, followed by the standard deviation across the three model/search runs in percentage points. '
            'Each main cell uses 96 native episodes on 32 shared contexts; the base has 32 episodes from one model. '
            'Encoder/prior cells use 48 episodes on 16 contexts. These counts describe executions, not independent task replicates.\n']

    def success(a, sd=True):
        value = f"{100*a['success']:.1f}"
        return value + (f" ± {100*a['success_run_sd']:.1f}" if sd and a['success_run_sd'] is not None else '')

    for title, methods in [('Frozen-policy and agreement baselines', CORE), ('Task-aware search and common-action extensions', EXTENSIONS)]:
        text += ['### ' + title + '\n', table(['Method', 'Easy', 'Conflict', 'Coupled OOD'],
                   [[LABELS[m], *[success(data[k, m]) for k in KINDS]] for m in methods])]
    text += ['Conservative context-level 95% intervals and all per-run counts are saved in '
             '`analyses/evidence_main/metrics.json`; the plot uses those intervals. In particular, an observed 0/96 on 32 contexts '
             'has interval [0.0%, 10.9%], and 96/96 has [89.1%, 100.0%], conditional on these tested models.\n',
             '![Held-out joint success with conservative context intervals](cube_composition_artifacts/analyses/final/main_comparison.png)\n']
    selected = ['waypoint_only', 'agreement_adam_waypoint_warm', 'naive_both', 'action_average',
                'projected_waypoint_selected', 'common_pair_selected', 'common_native_256', 'direct_action_CEM272']
    rows = []
    for k in KINDS:
        for m in selected:
            a = data[k, m]
            rows.append([k.replace('supported_', ''), LABELS[m],
                         *[f"{100*a[key]:.1f}" for key in ['released_success', 'keepout_satisfied', 'waypoint_satisfied', 'dropped']],
                         f"{a['grasp_losses']:.2f}",
                         f"{100*a['supported_transport_path_fraction']:.1f}", f"{100*a['robot_table_contact_in_transport']:.1f}"])
    text += ['### Physical outcomes\n',
             table(['Stratum', 'Method', 'Released goal %', 'K %', 'W %', 'Drop %', 'Grasp losses/episode', 'Supported path %', 'Robot/table %'], rows),
             'K/W satisfaction includes the inherited force-supported transport predicates. Supported-path percentage is the '
             'mean path-length fraction, rather than the fraction of successful episodes. Drop, contact and grasp-loss metrics '
             'use the explicit thresholds above; they are not exhaustive robot safety measurements. A sampled support loss does '
             'not automatically fail an episode if the cube is retained, support remains above the required path fraction, '
             'and every other criterion passes.\n']
    names = [('Gaussian init, no prior', 'waypoint_init_gaussian_prior_none'),
             ('Gaussian init, Gaussian prior', 'waypoint_init_gaussian_prior_gaussian'),
             ('Gaussian init, empirical prior', 'waypoint_init_gaussian_prior_empirical'),
             ('Encoder bank, Gaussian prior', 'waypoint_init_encoder_prior_gaussian'),
             ('Encoder bank, empirical prior', 'waypoint_encoder_ablation'),
             ('Causal prefix, empirical prior', 'waypoint_encoder_prefix_ablation')]
    text += ['### Matched initialization and regularization\n',
             'All rows below execute W with eight Adam steps, four sign-sector starts and warm replanning. '
             'No full-trajectory encoder input from the future is supplied.\n',
             table(['Initialization / prior', 'Easy', 'Conflict', 'Coupled OOD'],
                   [[label, *[success(abl[k, m]) for k in KINDS]] for label, m in names]),
             table(['Common-action causal encoder control', 'Easy', 'Conflict', 'Coupled OOD'],
                   [[LABELS[m], *[success(abl[k, m]) for k in KINDS]]
                    for m in ['common_encoder_ablation', 'common_encoder_prefix_ablation']])]
    behavior = load('analyses/behavior_main/episodes.json')
    plausibility = []
    for k in KINDS:
        for m in ['waypoint_only', 'agreement_adam_waypoint_warm', 'waypoint_encoder_prefix_ablation']:
            episodes = [a for a in behavior if a['kind'] == k and a['method'] == m]
            plausibility.append([k.replace('supported_', ''), LABELS[m], len(episodes),
                                 f"{np.mean([a['empirical_nll_pair'] for a in episodes]):.3f}",
                                 f"{100*np.mean([a['waypoint_code_outside_training_1_99_quantiles'] for a in episodes]):.1f}"])
    text += ['The same original encoder density is used to score every method below, regardless of its optimization penalty. '
             'Plausibility improves without guaranteeing a valid joint trajectory.\n',
             table(['Stratum', 'Method', 'Episodes', 'Mean empirical pair NLL', 'W codes outside TRAIN 1–99% interval %'], plausibility)]
    temporal = ['common_adam_fixed', 'common_adam_cold', 'common_adam_warm', 'common_adam_tail']
    text += ['### Temporal and optimizer ablations\n',
             'The following development/validation comparison uses model 42 and 12 tasks per supported stratum. '
             'It selected the warm common-latent protocol before test unblinding; these are not extra held-out test results.\n',
             table(['Common-action replanning', 'Easy joint successes', 'Conflict joint successes'],
                   [[m.removeprefix('common_adam_'), *[f"{validation[k,m]['success_count']}/{validation[k,m]['episodes']}" for k in KINDS[:2]]] for m in temporal]),
             'The complete validation tables also isolate 0/3/15 optimizer steps, H=8 versus H=16 agreement, '
             'Gaussian/encoder/structured initialization, and prior removal. On the eight conflicting validation scenes, '
             'all tested pure-action-agreement step/initialization/prior variants have zero joint successes.\n']
    combined = {(a['kind'], a['method']): a for a in load('analyses/evidence_combined/metrics.json')['groups']}
    text += ['### Additional combined-objective validation\n',
             'This controlled check was specified after primary test unblinding and uses validation only. '
             'The native task cost is augmented by action-agreement weight 0, 1 or 10; other CEM settings and physics budgets match. '
             'Each cell has 24 episodes on eight contexts; 32 weight-zero/model-42 records reuse earlier validation executions.\n',
             table(['Executed family / agreement weight', 'Easy %', 'Conflict %'],
                   [[f'{family}, {weight}', *[success(combined[k, f'task_agreement_{family}_w{weight}']) for k in KINDS[:2]]]
                    for family in ['waypoint', 'common'] for weight in [0, 1, 10]])]
    robust = []
    robust_methods = ['agreement_adam_waypoint_warm', 'agreement_hybrid_waypoint_warm',
                      'waypoint_encoder_prefix_ablation', 'direct_action_CEM272']
    for m in robust_methods:
        for k in KINDS:
            tree = 'main_core' if m.startswith('agreement_') else 'main_extensions'
            baseline = load(f'{tree}/seed42/test_{k}/{m}/records.json')[:8]
            values = [sum(a['success'] for a in baseline)]
            for seed in [73, 101]:
                rows = load(f'robustness_test/search{seed}/test_{k}/{m}/records.json')
                assert [a['seed'] for a in rows] == [a['seed'] for a in baseline]
                values.append(sum(a['success'] for a in rows))
            robust.append([LABELS[m], k.replace('supported_', ''), *[f'{v}/8' for v in values]])
    text += ['### Fixed-model optimizer randomness\n',
             'Actor pair 42 is held fixed; only the search RNG changes. The baseline reuses its first eight primary-test scenes. '
             'Each column uses identical scenes. This small audit is separate from the three-model comparison.\n',
             table(['Method', 'Stratum', 'Search seed 41', 'Search seed 73', 'Search seed 101'], robust)]
    phases = load('analyses/behavior_main/phases.json')['groups']
    phase_rows = []
    for m in ['waypoint_only', 'agreement_adam_waypoint_fixed', 'agreement_adam_waypoint_warm']:
        for phase in ['reach_grasp_lift', 'carry_to_waypoint', 'carry_to_goal', 'placement_release']:
            a = next(v for v in phases if v['kind'] == 'supported_conflict' and v['method'] == m and v['phase'] == phase)
            phase_rows.append([LABELS[m], phase, a['episode_count'], f"{a['raw_action_disagreement_mean']:.5f}",
                               f"{a['same_phase_code_jump_mean']:.2f}" if a['same_phase_code_jump_mean'] is not None else '—',
                               '/'.join(f'{v:.2f}' for v in a['same_phase_code_sign_flips_mean'])])
    text += ['### Phase consistency and compatibility\n',
             'Conflict-stratum phase summaries condition on reaching that phase. MSE compares raw independent generator chunks; '
             'it is measured before any geometric correction. Code sign flips are representation changes, not automatically changes '
             'in the physical route. Code jumps average consecutive replans within the same phase.\n',
             table(['Method', 'Phase', 'Episodes reaching phase', 'Raw action MSE', 'Code jump L2', 'K/W sign flips per phase'], phase_rows)]
    cap_rows = []
    for kind in KINDS[:2]:
        rows = load(f'capability_dev/{kind}/records.json')
        cap_rows.append([kind.replace('supported_', ''), len(rows),
                         f"{np.mean([r['minimum_initial_action_disagreement'] for r in rows]):.5f}",
                         f"{np.mean([r['minimum_closed_loop_trajectory_disagreement'] for r in rows]):.4f}",
                         sum(r['initial_agreement_minimizer_joint_success_W'] for r in rows),
                         sum(r['trajectory_minimizer_joint_success_if_execute_W'] for r in rows),
                         sum(any(r['any_joint_success_by_family'].values()) for r in rows)])
    text += [table(['Finite-grid stratum', 'Contexts', 'Mean minimum initial action MSE', 'Mean minimum native pose MSE',
                    'Initial-agreement W joint successes', 'Pose-agreement W joint successes', 'Any family/code joint successes'], cap_rows),
             'Pose MSE compares end-effector/cube trajectories over the fixed H=240 window, scaled by 10 cm, with completed '
             'candidate states padded after placement. These minima are over eight codes per individual family; summed activation '
             'adds 64 pairs. This is a finite fixed-code probe, not a proof about the full learned distribution.\n']
    geo = load('geometry_audit_full/summary.json')
    geometry = defaultdict(list)
    for a in geo['runs']:
        geometry[a['kind'], a['method']].append(a)
    methods = ['waypoint_only', 'agreement_adam_waypoint_warm', 'agreement_random_waypoint_warm',
               'native_latent_256', 'projected_waypoint_selected', 'common_pair_selected', 'common_native_256', 'direct_action_CEM272']
    rows = []
    for m in methods:
        cells = []
        for k in KINDS:
            a = geometry[k, m]
            cells.append(' → '.join(f'{100*np.mean([x[key] for x in a]):.1f}'
                                   for key in ['primary_joint_success', 'secondary_sampled_box_joint_success']))
        rows.append([LABELS[m], *cells])
    text += ['### Native geometry sensitivity\n',
             f"All {geo['episodes']:,} planned source trajectories/futures were replayed: "
             f"{geo['native_controls']:,} controls and {geo['geometry_samples']:,} native geometry samples. "
             f"Maximum pose reproduction error was {geo['max_replay_pose_error_m']:.3g} m.\n",
             'Each cell is conservative-primary joint success → sampled actual-box joint success (%). '
             'All goal, contact, support and drop criteria remain the same.\n',
             table(['Method', 'Easy', 'Conflict', 'Coupled OOD'], rows)]
    text.append('![Same native trajectories, different body-clearance geometry](cube_composition_artifacts/analyses/final/geometry_sensitivity.png)\n')
    capability = []
    for bank in geo['capability']:
        for case in bank['cases']:
            capability.append(case)
    text.append('Finite fixed-code probe: ' + ', '.join(
        f"{sum(any(c[key].values()) for c in capability)}/{len(capability)} contexts have at least one successful candidate under {name} geometry"
        for key, name in [('any_primary_success_by_family', 'primary'), ('any_secondary_box_success_by_family', 'actual-box')]) + '.\n')
    text += ['### Dedicated inference latency\n',
             'Warm columns pool the ten repeated-state queries across three phase states in each of three example episodes (90 calls/method). '
             'First-solve is the median of nine first queries. Worker startup is reported separately. '
             'Fixed baseline timings include pair-agreement diagnostics and are not minimal standalone generator timings.\n']
    timing = defaultdict(list)
    for k in KINDS:
        for a in load('benchmarks/' + k + '.json')['records']:
            timing[a['method']].append(a)
    rows = []
    aliases = load('protocols/benchmark_sources.json')['aliases']
    for m in LABELS:
        measured_name = aliases.get(m, m)
        if measured_name not in timing:
            continue
        records = timing[measured_name]
        values = [c['seconds'] for a in records for c in a['calls'][1:]]
        rows.append([LABELS[m], f"{1000*np.median([a['initial_seconds'] for a in records]):.1f}",
                     f'{1000*np.median(values):.1f}', f'{1000*np.quantile(values,.95):.1f}', f'{100*np.mean(np.array(values)>.4):.1f}'])
    text.append(table(['Method', 'First solve ms', 'Warm p50 ms', 'Warm p95 ms', 'Warm >400 ms %'], rows))
    startups = [load('benchmarks/' + k + '.json')['native_worker_warmup_seconds'] for k in KINDS]
    text.append('Four-worker native startup/warmup per source: ' + ', '.join(f'{x:.2f} s' for x in startups) +
                '. Closed-loop planning times, CPU forecast counts, neural candidate equivalents, batched calls and backwards '
                'are retained separately in `analyses/evidence_main/metrics.json`. Identical configuration aliases reuse the same '
                'timing measurements, as recorded in `protocols/benchmark_sources.json`.\n')
    output = ROOT / 'analyses/final'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'quantitative_tables.md').write_text('\n'.join(text) + '\n')
    chosen = ['waypoint_only', 'agreement_adam_waypoint_warm', 'agreement_random_waypoint_warm',
              'naive_both', 'native_trajectory_256', 'native_latent_256', 'grip_safe_blend',
              'common_latent_selected', 'common_pair_selected', 'projected_waypoint_selected',
              'direct_action_CEM272', 'common_native_256']
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8), sharey=True)
    for ax, k in zip(axes, KINDS):
        records = [data[k, m] for m in chosen]
        means = np.array([a['success'] for a in records]) * 100
        low = np.array([a['success_context_bounded_95ci'][0] for a in records]) * 100
        high = np.array([a['success_context_bounded_95ci'][1] for a in records]) * 100
        y = np.arange(len(chosen))
        ax.barh(y, means, color=['#54799b'] * 6 + ['#bd8244'] * 6)
        ax.errorbar(means, y, xerr=np.maximum(0, np.stack([means-low, high-means])), fmt='none', color='#293746', capsize=2)
        ax.set(yticks=y, yticklabels=[LABELS[m] for m in chosen], xlim=(0, 103), xlabel='Native joint success (%)', title=k.replace('_', ' '))
        ax.grid(axis='x', alpha=.2)
    axes[0].tick_params(axis='y', labelsize=9)
    fig.suptitle('Frozen Cube skills | 32 shared test scenes per stratum, three model/search runs\nPrimary conservative cube keep-out; 95% bounded context intervals', fontsize=12)
    fig.tight_layout()
    fig.savefig(output / 'main_comparison.png', dpi=180)
    fig.savefig(output / 'main_comparison.pdf')
    plt.close(fig)
    raw_geometry = load('geometry_audit_full/records.json')
    chosen_geometry = ['waypoint_only', 'agreement_adam_waypoint_warm', 'native_latent_256',
                       'projected_waypoint_selected', 'direct_action_CEM272', 'common_native_256']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for offset, field, color, label in [(-.18, 'primary_joint_success', '#a9afb5', 'Primary conservative envelope'),
                                             (.18, 'secondary_sampled_box_joint_success', '#397c9b', 'Sampled actual oriented box')]:
            means, lower, upper = [], [], []
            for method in chosen_geometry:
                records = [a for a in raw_geometry if a['candidate_index'] is None and a['kind'] == kind
                           and Path(a['source']).parent.name == method]
                by_context = defaultdict(list)
                for record in records:
                    by_context[record['seed']].append(record[field])
                context_rates = [np.mean(v) for v in by_context.values()]
                value = float(np.mean(context_rates))
                lo, hi = bounded_mean_interval(value, len(context_rates))
                means.append(100*value)
                lower.append(100*lo)
                upper.append(100*hi)
            means = np.array(means)
            y = np.arange(len(chosen_geometry)) + offset
            ax.barh(y, means, height=.32, color=color, label=label)
            ax.errorbar(means, y, xerr=np.maximum(0, np.stack([means-np.array(lower), np.array(upper)-means])),
                        fmt='none', color=color, capsize=2)
        ax.set(yticks=np.arange(len(chosen_geometry)), yticklabels=[LABELS[m] for m in chosen_geometry],
               xlim=(0, 103), xlabel='Native joint success (%)', title=kind.replace('_', ' '))
        ax.grid(axis='x', alpha=.15)
    axes[0].tick_params(axis='y', labelsize=9)
    fig.suptitle('Geometry sensitivity on the same executed trajectories\nAll other grasp, support, drop and goal criteria retained; 95% bounded context intervals', fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / 'geometry_sensitivity.png', dpi=180)
    fig.savefig(output / 'geometry_sensitivity.pdf')
    plt.close(fig)
    print('Wrote final quantitative tables and comparison figure')


if __name__ == '__main__':
    main()
