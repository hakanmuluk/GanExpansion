"""Summarize repeated model pairs without counting scenes repeatedly."""
import argparse
import json
from pathlib import Path
import numpy as np
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();manifest=json.loads(Path(args.manifest).read_text());rng=np.random.default_rng(719231)
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    groups={};provenance=[]
    for entry in manifest['runs']:
        root=Path(entry['path']);assert (root/'DONE.json').exists()
        config=json.loads((root/'config.json').read_text());rows=json.loads((root/'records.json').read_text())
        assert [row['context'] for row in rows]==list(range(len(rows)))
        group=groups.setdefault((entry['stratum'],entry['method']),[])
        group.append((entry['model_seed'],rows,config['task_sha256']))
        provenance.append({**entry,'records_sha256':digest_file(root/'records.json'),'config_sha256':digest_file(root/'config.json')})
    summaries=[]
    for (stratum,method),runs in sorted(groups.items()):
        assert len({seed for seed,_,_ in runs})==len(runs)
        assert len({h for _,_,h in runs})==1,'Repeated seeds must use the same scene bank'
        assert len({len(rows) for _,rows,_ in runs})==1
        matrix=np.asarray([[row['success'] for row in rows] for _,rows,_ in runs],dtype=float)
        n=matrix.shape[1];clustermean=matrix.mean(0)
        boots=clustermean[rng.integers(0,n,(10000,n))].mean(1)
        metrics={key:float(np.mean([[row[key] for row in rows] for _,rows,_ in runs]))
                 for key in ['goal_success','keepout_satisfied','waypoint_satisfied',
                             'keepout_full_success','waypoint_full_success','steps','planning_seconds']}
        summaries.append({'stratum':stratum,'method':method,'scene_identities':n,'trained_pairs':len(runs),
            'native_episodes':int(matrix.size),'strict_joint_success':float(matrix.mean()),
            'scene_clustered_bootstrap_95_interval':np.quantile(boots,[.025,.975]).tolist(),
            'per_training_seed':[{'seed':seed,'successes':int(sum(row['success'] for row in rows)),
                                 'episodes':len(rows),'success_rate':float(np.mean([row['success'] for row in rows]))}
                                 for seed,rows,_ in runs],**metrics})
    result={'scope':manifest['scope'],'runs':provenance,'groups':summaries,
            'uncertainty':'Percentile bootstrap over scene identities, averaging training pairs within each resampled scene. The interval describes these trained pairs and this benchmark distribution; three training seeds do not establish broad architecture robustness.',
            'manifest_sha256':digest_file(args.manifest)}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'scope':result['scope'],'groups':summaries},indent=2))


if __name__=='__main__':main()
