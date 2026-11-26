"""Offline causal-prefix audit of ORIGINAL Cube skill encoders.

Uses recorded states/contexts only. Counterfactual action tokens do not imply
counterfactual physics trajectories. No new encoder or model is trained.
Only aggregate metrics are exported; raw trajectories stay in the runtime.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .legacy import load_skill,digest_file


def generated_tokens(actor,states,contexts,bz,code,execute):
    t=states.shape[1];observed=states[:,::execute];own=contexts[:,::execute]
    n=observed.shape[1]
    plans=actor(observed.flatten(0,1),bz[:,None].expand(-1,n,-1).flatten(0,1),
                code[:,None].expand(-1,n,-1).flatten(0,1),own.flatten(0,1))
    controls=plans[:,:execute].reshape(len(states),-1,5)[:,:t]
    # The audit of the encoder uses the controls actually sent to the robot.
    return torch.cat([controls[...,:4],torch.where(controls[...,4:]>=0,1.,-1.)],-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',default='independent_composition_artifacts/adapter_only_original_diagnostic_v15_cube/results')
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=8)
    p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out);assert not out.exists(),out
    out.mkdir(parents=True);started=time.time();cache={};samples=[];inputs=[]
    for cfgpath in sorted(Path(args.runs).glob('*/config.json')):
        directory=cfgpath.parent;config=json.loads(cfgpath.read_text());a=config['args']
        if a['skill'] not in ['keepout','waypoint'] or not Path(a['tasks']).stem.endswith(('supported_easy','supported_conflict')):continue
        assert (directory/'DONE.json').exists()
        skill=a['skill'];index=0 if skill=='keepout' else 1
        if skill not in cache:cache[skill]=load_skill('cube',skill,args.device)
        actor,encoder,source=cache[skill]
        rows=json.loads((directory/'records.json').read_text())[:args.count]
        inputs.append({'run':str(directory),'record_sha256':digest_file(directory/'records.json'),'episodes':len(rows),'skill':skill,'code':a['code']})
        for row in rows:
            with np.load(directory/f"rollout_{row['context']:04d}.npz") as data:
                states=torch.as_tensor(data['states'],device=args.device)[None]
                actions=torch.as_tensor(data['actions'],device=args.device)[None]
                context=torch.as_tensor(data['contexts'][:,index],device=args.device)[None]
                bz=torch.as_tensor(data['base_z'],device=args.device)[None]
            total=actions.shape[1]
            for requested in [8,16,32,64,0]:
                t=min(total,requested or total);s=states[:,:t];act=actions[:,:t];c=context[:,:t]
                lengths=torch.tensor([t],device=args.device);code=s.new_tensor([[float(a['code'])]])
                with torch.no_grad():
                    actual_mu,_=encoder(s,act,c,bz,lengths)
                    zero_mu,_=encoder(s,torch.zeros_like(act),c,bz,lengths)
                    own_actions=generated_tokens(actor,s,c,bz,code,a['execute'])
                    opposite_actions=generated_tokens(actor,s,c,bz,-code,a['execute'])
                    own_mu,_=encoder(s,own_actions,c,bz,lengths)
                    opposite_mu,_=encoder(s,opposite_actions,c,bz,lengths)
                z=code.detach().clone().requires_grad_(True)
                differentiable=generated_tokens(actor,s,c,bz,z,a['execute'])
                recovered,_=encoder(s,differentiable,c,bz,lengths)
                sensitivity,=torch.autograd.grad(recovered.sum(),z)
                samples.append({'skill':skill,'group':Path(a['tasks']).stem.removeprefix('val_'),
                    'prefix':'full' if not requested else str(requested),'length':t,'code':float(code.item()),
                    'actual':float(actual_mu.item()),'zero_actions':float(zero_mu.item()),
                    'generated':float(own_mu.item()),'flipped_generated':float(opposite_mu.item()),
                    'derivative':float(sensitivity.item()),
                    'replay_action_mse':float((own_actions-act).square().mean()),
                    'changed_code_action_mse':float((own_actions-opposite_actions).square().mean()),
                    'own_success':bool(row[skill+'_full_success'])})
    assert len(inputs)==8,inputs
    metrics=[]
    for skill in ['keepout','waypoint']:
        for group in ['supported_easy','supported_conflict']:
            for prefix in ['8','16','32','64','full']:
                rows=[r for r in samples if (r['skill'],r['group'],r['prefix'])==(skill,group,prefix)]
                code=np.array([r['code'] for r in rows]);assert set(np.sign(code))=={-1,1}
                metric={'skill':skill,'group':group,'prefix':prefix,'episodes':len(rows),
                    'mean_observed_length':float(np.mean([r['length'] for r in rows])),
                    'own_success_count':sum(r['own_success'] for r in rows)}
                for name in ['actual','zero_actions','generated','flipped_generated']:
                    values=np.array([r[name] for r in rows]);target=-code if name=='flipped_generated' else code
                    metric[name+'_rmse']=float(np.sqrt(np.mean((values-target)**2)))
                    metric[name+'_sign_accuracy']=float(np.mean(np.sign(values)==np.sign(target)))
                metric.update(action_code_sensitivity_mean=float(np.mean([abs(r['derivative']) for r in rows])),
                    sign_reversal_from_changed_generated_actions=float(np.mean([np.sign(r['generated'])!=np.sign(r['flipped_generated']) for r in rows])),
                    replay_action_mse=float(np.mean([r['replay_action_mse'] for r in rows])),
                    generated_code_flip_action_mse=float(np.mean([r['changed_code_action_mse'] for r in rows])))
                metrics.append(metric)
    for actor,encoder,source in cache.values():
        assert digest_file(source['checkpoint'])==source['sha256']
        assert all(not p.requires_grad and p.grad is None for module in [actor,encoder] for p in module.parameters())
    result={'scope':__doc__,'inputs':inputs,'metrics':metrics,'seconds':time.time()-started,
        'models':{skill:source for skill,(actor,encoder,source) in cache.items()},
        'encoder_geometry_only':{skill:encoder.geometry_only for skill,(actor,encoder,source) in cache.items()},
        'source_sha256':digest_file(__file__),'original_weights_unchanged':True,
        'warning':'Changing actions at fixed observed states is a sensitivity audit, not a physically valid alternate rollout or proof of skill satisfaction.'}
    (out/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':sum(i['episodes'] for i in inputs),'prefix_measurements':len(samples),'original_weights_unchanged':True})+'\n')
    print(json.dumps({'out':str(out),'episodes':sum(i['episodes'] for i in inputs),'prefix_measurements':len(samples),'seconds':result['seconds']}))


if __name__=='__main__':main()
