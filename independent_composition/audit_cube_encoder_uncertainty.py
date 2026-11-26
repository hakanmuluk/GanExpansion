"""Audit existing posterior uncertainty on actual Cube prefixes, without fitting."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .legacy import load_skill,digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--runs',default='independent_composition_artifacts/adapter_only_original_diagnostic_v15_cube/results')
    p.add_argument('--count',type=int,default=8);args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
    models={};records=[];sources={};started=time.time()
    for path in sorted(Path(args.runs).glob('*/config.json')):
        a=json.loads(path.read_text())['args'];skill=a['skill']
        if skill not in ['keepout','waypoint'] or not Path(a['tasks']).stem.endswith(('supported_easy','supported_conflict')):continue
        assert (path.parent/'DONE.json').exists()
        if skill not in models:
            actor,encoder,source=load_skill('cube',skill);models[skill]=(actor,encoder);sources[skill]=source
        encoder=models[skill][1];j=0 if skill=='keepout' else 1
        for rollout in sorted(path.parent.glob('rollout_*.npz'))[:args.count]:
            with np.load(rollout) as data:
                inputs=[torch.as_tensor(data[name],device='cuda')[None] for name in ['states','actions']]
                contexts=torch.as_tensor(data['contexts'][:,j],device='cuda')[None]
                bz=torch.as_tensor(data['base_z'],device='cuda')[None]
            for prefix in [8,16,32,64,0]:
                n=min(len(inputs[1][0]),prefix or len(inputs[1][0]))
                with torch.no_grad():
                    mu,lv=encoder(inputs[0][:,:n],inputs[1][:,:n],contexts[:,:n],bz,torch.tensor([n],device='cuda'))
                    std=(.5*lv).exp();error=mu-float(a['code']);standardized=error/std
                    kl=.5*(mu.square()+lv.exp()-1-lv)
                    nll=.5*(np.log(2*np.pi)+lv+standardized.square())
                records.append({'skill':skill,'prefix':str(prefix) if prefix else 'full',
                    'std':float(std.item()),'squared_error':float(error.square().item()),
                    'squared_standardized_error':float(standardized.square().item()),
                    'covered95':bool(standardized.abs().item()<=1.96),
                    'kl_to_unit':float(kl.item()),'nll':float(nll.item())})
    metrics=[]
    for skill in ['keepout','waypoint']:
        for prefix in ['8','16','32','64','full']:
            rows=[r for r in records if r['skill']==skill and r['prefix']==prefix]
            assert len(rows)==32
            metrics.append({'skill':skill,'prefix':prefix,'episodes':len(rows),
                'mean_predicted_std':float(np.mean([r['std'] for r in rows])),
                'actual_rmse':float(np.sqrt(np.mean([r['squared_error'] for r in rows]))),
                'standardized_rmse':float(np.sqrt(np.mean([r['squared_standardized_error'] for r in rows]))),
                'coverage95':float(np.mean([r['covered95'] for r in rows])),
                'kl_to_unit_mean':float(np.mean([r['kl_to_unit'] for r in rows])),
                'gaussian_nll_mean':float(np.mean([r['nll'] for r in rows]))})
    assert all(digest_file(s['checkpoint'])==s['sha256'] for s in sources.values())
    result={'scope':__doc__,'metrics':metrics,'models':sources,'seconds':time.time()-started,
        'original_weights_unchanged':True,'source_sha256':digest_file(__file__),
        'calibration_note':'Balanced +/-0.8 original skill-code pilot; observed prefix posterior only. No new training, code proposal or trajectory export.'}
    (out/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':64,'prefix_posteriors':len(records),'original_weights_unchanged':True})+'\n')
    print(json.dumps({'out':str(out),'metrics':metrics,'seconds':result['seconds']}))


if __name__=='__main__':main()
