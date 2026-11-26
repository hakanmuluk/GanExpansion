"""Concatenate own-skill training corpora with separate context identities."""
import argparse
import json
from pathlib import Path
import numpy as np

from .legacy import digest_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Fresh merged own-skill data directory required')
    out.mkdir(parents=True)
    report = {'sources': args.sources, 'skills': {},
              'scope': 'Own-skill demonstration concatenation only. No joint bank or test access.'}
    keys = ['states', 'contexts', 'actions', 'lengths', 'base_z', 'new_z']
    for skill in ['keepout', 'waypoint']:
        (out / skill).mkdir(); report['skills'][skill] = {}
        for split in ['train', 'val']:
            records = []; fingerprints = []; identities = []
            for source_index, source in enumerate(args.sources):
                path = Path(source) / skill / (split + '.npz')
                with np.load(path) as d:
                    records.append({k: d[k] for k in keys})
                    ids = d['seed'] if 'seed' in d else d['pair_id']
                    _, inverse = np.unique(ids, return_inverse=True)
                    identities.append(inverse + source_index * 10000000)
                fingerprints.append({'path': str(path), 'sha256': digest_file(path),
                                     'episodes': len(records[-1]['lengths'])})
            # Some collections have longer padded state storage. Trim only
            # padding; lengths/masks retain the exact valid demonstration data.
            for key in ['states', 'contexts', 'actions']:
                size = min(r[key].shape[1] for r in records)
                assert all((r['lengths'] <= size).all() for r in records)
                for record in records:
                    record[key] = record[key][:, :size]
            assert len({r['new_z'].shape[-1] for r in records}) == 1, 'Incompatible style labels'
            merged = {k: np.concatenate([r[k] for r in records]) for k in keys}
            merged['seed'] = np.concatenate(identities)
            np.savez_compressed(out / skill / (split + '.npz'), **merged)
            report['skills'][skill][split] = {'sources': fingerprints, 'episodes': len(merged['lengths']),
                                            'contexts': len(np.unique(merged['seed']))}
    (out / 'DONE.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
