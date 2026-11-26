"""Presentation figure for the fully evaluated two-seed TwoRoom rank subset.

This reads aggregate evidence only. It does not change any scientific protocol
and does not label the unfinished Cube/modulation study as complete.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    source = Path(args.summary)
    data = json.loads(source.read_text())
    rows = [r for r in data['adapters']
            if r['environment'] == 'tworoom' and not r['lora_only']]
    expected = {(skill, rank, seed) for skill in ['keepout', 'waypoint']
                for rank in [1, 2, 4, 8, 16] for seed in [123, 2026]}
    keys = [(r['skill'], r['rank'], r['seed']) for r in rows]
    assert len(keys) == len(set(keys)) == 20 and set(keys) == expected
    assert all(r['episodes'] == 2048 and r['contexts'] == 256 for r in rows)
    assert all(r['disabled_prediction_bitwise_equal'] for r in rows)
    for row in rows:
        for field in ['success', 'both_successful_strategies']:
            assert 0 <= row[field] <= 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    ranks = [1, 2, 4, 8, 16]
    fig = plt.figure(figsize=(10.8, 6.7), layout=None)
    grid = fig.add_gridspec(2, 3, left=.065, right=.98, top=.85,
                           bottom=.155, hspace=.47, height_ratios=[1.6, 1])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    metrics = [('success', 'Joint skill success (%)', 100),
               ('both_successful_strategies', 'Both successful strategies (%)', 100),
               ('adapter_parameters', 'Action-adapter parameters (k)', .001)]
    colors = {'keepout': '#2874a6', 'waypoint': '#dc7b28'}
    means = []
    for rank in ranks:
        entry = {'rank': rank}
        for skill in ['keepout', 'waypoint']:
            subset = [r for r in rows if r['skill'] == skill and r['rank'] == rank]
            for metric, _, _ in metrics:
                values = [r[metric] for r in subset]
                for name, value in [('mean', statistics.mean(values)),
                                    ('min', min(values)), ('max', max(values))]:
                    entry[f'{skill}_{metric}_{name}'] = value
        means.append(entry)
    for ax, (metric, label, scale) in zip(axes, metrics):
        for skill, name in [('keepout', 'Keep-out'), ('waypoint', 'Waypoint')]:
            y = np.array([r[f'{skill}_{metric}_mean'] for r in means]) * scale
            low = np.array([r[f'{skill}_{metric}_min'] for r in means]) * scale
            high = np.array([r[f'{skill}_{metric}_max'] for r in means]) * scale
            ax.errorbar(ranks, y, yerr=[y-low, high-y], marker='o', markersize=4,
                        capsize=3, color=colors[skill], linewidth=1.5, label=name)
        ax.set_xscale('log', base=2)
        ax.set_xticks(ranks, [str(r) for r in ranks])
        ax.set_xlabel('LoRA rank')
        ax.set_ylabel(label)
        ax.grid(alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
        if scale == 100:
            ax.set_ylim(80, 101)
            ax.set_yticks([80, 85, 90, 95, 100])
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')

    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis('off')
    cells = []
    for r in means:
        cells.append([r['rank']] + [
            f"{r[f'{skill}_{metric}_mean']*scale:.1f}"
            for skill, metric, scale in [
                ('keepout', 'success', 100), ('keepout', 'both_successful_strategies', 100),
                ('waypoint', 'success', 100), ('waypoint', 'both_successful_strategies', 100),
                ('keepout', 'adapter_parameters', .001), ('waypoint', 'adapter_parameters', .001)]])
    table = table_ax.table(cellText=cells,
                          colLabels=['Rank', 'K success (%)', 'K both (%)',
                                     'W success (%)', 'W both (%)', 'K params (k)', 'W params (k)'],
                          loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('#d5dce0')
        cell.set_linewidth(.45)
        if row == 0:
            cell.set_facecolor('#eaf0f4')
            cell.set_text_props(weight='bold', color='#17354a')

    fig.suptitle('TwoRoom: completed two-seed LoRA rank study', fontsize=15,
                 color='#17354a', y=.98, fontweight='bold')
    fig.text(.5, .915,
             'Full adapters with the original style branch; seeds 123 and 2026',
             ha='center', fontsize=10)
    fig.text(.065, .077,
             'Points/table: two-seed means. Bars: seed minimum to maximum, not confidence intervals.\n'
             'Each model: 256 held-out contexts x 8 Gaussian private codes; original joint goal-plus-skill rules.\n'
             'Both: contexts with both successful strategies and original doorway style. Parameters exclude base, E and D.\n'
             'Only rank changes. The fixed style branch remains; Cube and pure branch-removal replication are separate.',
             fontsize=8.5, va='center', linespacing=1.45)
    for extension in ['pdf', 'png']:
        fig.savefig(out/f'tworoom_two_seed_rank.{extension}', dpi=160)
    plt.close(fig)
    for name, records in [('per_seed_results.csv', rows), ('mean_and_seed_ranges.csv', means)]:
        with (out/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    receipt = {'scope': 'Completed TwoRoom full-adapter rank subset only',
               'whole_study_complete': False, 'models': 20,
               'training_seeds': [123, 2026], 'closed_loop_episodes': sum(r['episodes'] for r in rows),
               'source_summary_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'all_disabled_action_checks_exact': True,
               'scientific_protocol_changed': False,
               'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'COMPLETED_SUBSET.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'out': str(out), 'models': 20,
                      'closed_loop_episodes': receipt['closed_loop_episodes'], 'whole_study_complete': False}))


if __name__ == '__main__':
    main()
