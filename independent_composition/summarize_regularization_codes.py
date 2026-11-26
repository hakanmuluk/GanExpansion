"""Aggregate actual selected codes, excluding already completed batch episodes."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(folder):
    config=json.loads((folder/'config.json').read_text())['composition']
    rows=json.loads((folder/'records.json').read_text());plans=json.loads((folder/'planning.json').read_text())
    private=[];cycle=[];amplitude=[]
    if plans and isinstance(plans[0],dict):
        lengths=np.asarray([r['steps'] for r in rows])
        grouped=[(p,np.asarray(lengths)>p['step']) for p in plans]
    else:grouped=[(p,np.ones(1,dtype=bool)) for episode in plans for p in episode]
    for p,alive in grouped:
        z=np.atleast_2d(np.asarray(p['selected_existing_inputs'],dtype=np.float64))
        private.extend(z[alive,:2].tolist())
        amplitude.extend(np.asarray(p['base_amplitude_tether'])[alive].tolist())
        if config['cycle_weight']:
            cycle.extend(np.asarray(p['private_cycle_mae'])[alive].tolist())
    private=np.asarray(private)
    return {'alive_replan_conditions':len(private),'mean_absolute_private_code':float(np.abs(private).mean()),
        'mean_private_gaussian_nll':float((.5*private**2).sum(1).mean()),
        'median_absolute_private_code':float(np.median(np.abs(private))),
        'private_absolute_code_p95':float(np.quantile(np.abs(private),.95)),
        'mean_base_amplitude_change_squared':float(np.mean(amplitude)),
        'cycle_mae_when_enabled':float(np.mean(cycle)) if cycle else None,
        'private_prior_weight':config['private_prior_weight'],'cycle_weight':config['cycle_weight'],
        'cycle_target':config['cycle_target'],
        'interpretation':'Actual alive replans, not a common-state causal comparison. Disabled cycle scores are null, not zero.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluations',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();report={}
    for folder in sorted(Path(a.evaluations).glob('*/*')):
        if (folder/'DONE.json').exists():report[str(folder.relative_to(a.evaluations))]=summarize(folder)
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'cells':len(report),'out':a.out}))


if __name__=='__main__':main()
