"""Recompute geometry metrics from persisted trajectories, without rerunning policy."""
import argparse
import json
from pathlib import Path
import numpy as np
from .metrics import carry_geometry
from .evaluate import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument("paths",nargs="+");args=p.parse_args()
    for name in args.paths:
        root=Path(name)
        records=json.loads((root/"records.json").read_text())
        with np.load(root/"rollouts.npz") as archive:
            d={k:archive[k] for k in ["poses","lengths","initial_cube","goal"]}
            for i,r in enumerate(records):r.update(carry_geometry(d["poses"][i],int(d["lengths"][i]),d["initial_cube"][i],d["goal"][i]))
        report=json.loads((root/"summary.json").read_text());report.update(summarize(records));report["route_metric_version"]=2
        (root/"records.json").write_text(json.dumps(records,indent=2));(root/"summary.json").write_text(json.dumps(report,indent=2))
        print(str(root),report["released_success"],report["both_routes_per_context"])


if __name__=="__main__":main()
