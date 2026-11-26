"""Offline aggregate code-switch diagnostics; no inference or model changes.

Read completed action-agreement evaluations on the collection host only. This
measures association with failure, not whether switching causes failure. Never
exports native observations/actions or individual latent traces.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); out = Path(a.out); assert not out.exists()
    result = {'scope': __doc__, 'benchmarks': {}}
    for folder in sorted(Path(a.evaluations).glob('*/*')):
        if not (folder / 'DONE.json').exists():
            continue
        plans = json.loads((folder / 'planning.json').read_text())
        records = json.loads((folder / 'records.json').read_text())
        groups = {'success': [], 'failure': []}
        for i, (episode, record) in enumerate(zip(plans, records)):
            with np.load(folder / ('rollout_%04d.npz' % i)) as x:
                context = x['contexts']
                selected = [v for v in episode if context[v['step'], 0, 14] > .5
                            and context[v['step'], 0, 13] > .5
                            and .1 < context[v['step'], 0, 10] < .9]
            if len(selected) < 2:
                continue
            codes = np.asarray([v['selected_existing_inputs'][0] for v in selected])
            assert codes.shape[1] == 4 and np.isfinite(codes).all()
            delta = np.diff(codes, axis=0)
            groups['success' if record['success'] else 'failure'].append({
                'carry_replans': len(selected),
                'private_sign_flip_fraction': np.mean(np.sign(codes[1:, :2]) != np.sign(codes[:-1, :2]), axis=0),
                'input_step_rms': np.sqrt(np.mean(delta ** 2, axis=0)),
            })
        aggregates = {}
        for name, rows in groups.items():
            aggregates[name] = {'episodes': len(rows)}
            for key in ['carry_replans', 'private_sign_flip_fraction', 'input_step_rms']:
                if rows:
                    aggregates[name][key] = np.mean([v[key] for v in rows], axis=0).tolist()
        result['benchmarks'][str(folder.relative_to(a.evaluations))] = aggregates
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
