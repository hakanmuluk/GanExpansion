"""Native evaluation kept outside the tensor-only composition policy.

Only validation/development banks may be used during method selection. Fresh
final test banks require a recorded model/configuration freeze before opening.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from .legacy import digest_file
from .window import load_encoder
from .latent_search import LatentComposer,LatentConfig


class TwoRoomBridge:
    def __init__(self, composer):self.composer=composer

    def plan(self,pos,case,visited,clock,previous_action):
        from tworoom_gan.env import observation
        from tworoom_composition.geometry import contexts
        ctx=contexts(pos,case,visited,clock,previous_action,self.composer.bank.dimensions)
        return self.composer.optimize(observation(pos,case['goal']),case['base_z'],ctx)


class CubeBridge:
    def __init__(self, composer):self.composer=composer

    def plan(self,state,contexts,base_z,runtime=None):
        # Runtime may contain the environment for monitoring. Never inspect it.
        d=self.composer.device
        s=torch.as_tensor(state,dtype=torch.float32,device=d)[None]
        c=tuple(torch.as_tensor(x,dtype=torch.float32,device=d)[None] for x in contexts)
        bz=torch.as_tensor(base_z,dtype=torch.float32,device=d)[None]
        action,z,meta=self.composer.optimize(s,bz,c)
        meta={k:float(v[0]) if torch.is_tensor(v) else v for k,v in meta.items()}
        meta['native_controls']=0;meta['native_forks']=0
        return action[0].cpu().numpy(),z[0].cpu().numpy(),meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--env',choices=['tworoom','cube'],required=True);p.add_argument('--tasks',required=True)
    p.add_argument('--out',required=True);p.add_argument('--config',required=True)
    p.add_argument('--encoders',nargs=2);p.add_argument('--count',type=int,default=16)
    p.add_argument('--adapter-only-consistency',action='store_true')
    p.add_argument('--execution-weight',type=float,default=1.)
    p.add_argument('--model-seed',type=int,default=42);p.add_argument('--seed',type=int,default=41)
    p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(2)
    if '/test' in args.tasks or Path(args.tasks).name.startswith('test'):
        raise ValueError('Do not open final tests during exploratory ablation')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    c=LatentConfig(**json.loads(args.config if args.config.startswith('{') else Path(args.config).read_text()))
    encoders=[]
    if args.adapter_only_consistency and args.encoders:
        raise ValueError('Adapter-only composition cannot load auxiliary encoders')
    for path in args.encoders or []:
        encoder,saved=load_encoder(path,args.device)
        assert saved['config']['args']['env']==args.env
        assert saved['config']['args']['model_seed']==args.model_seed
        encoders.append(encoder)
    if args.env=='tworoom':
        from tworoom_composition.models import FrozenBank
        from tworoom_composition.evaluate import load_cases,summary
        from tworoom_composition.rollout import rollout
        k='addon_artifacts/final/keepout.pt' if args.model_seed==42 else f'addon_artifacts/runs/keepout_s{args.model_seed}/best.pt'
        w='addon_artifacts/final/waypoint.pt' if args.model_seed==42 else f'addon_artifacts/runs/waypoint_s{args.model_seed}/best.pt'
        bank=FrozenBank(k,w,device=args.device)
    else:
        from cube_composition.models import FrozenBank
        bank=FrozenBank(args.model_seed,args.device)
    settings={'args':vars(args),'search':asdict(c),'frozen_files':bank.files,
              'task_sha256':digest_file(args.tasks),'encoder_sha256':{x:digest_file(x) for x in args.encoders or []},
              'source_sha256':{p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')},
              'inference':'CURRENT observations only. Zero simulator queries, task costs, or geometry projections.'}
    (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
    def make_composer(seed):
        if args.adapter_only_consistency:
            from .adapter_only import AdapterOnlyComposer
            return AdapterOnlyComposer(bank,c,seed,args.execution_weight)
        return LatentComposer(bank,encoders or None,c,seed)
    if args.env=='tworoom':
        composer=make_composer(args.seed)
        cases=load_cases(args.tasks,args.device,args.count)
        rows,trace,info=rollout(bank,cases,None,None,args.seed,planner_override=TwoRoomBridge(composer))
        measured=summary(rows,info)
        np.savez_compressed(out/'trajectories.npz',**trace,**{k:v.cpu().numpy() for k,v in cases.items()})
        (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
        (out/'planning.json').write_text(json.dumps(info,indent=2)+'\n')
    else:
        from cube_composition.evaluate import load_cases,summarize
        from cube_composition.rollout import rollout
        from cube_addons.env import make_env,close_env
        rows=[];plans=[];env=make_env();started=time.time()
        try:
            for i,case in enumerate(load_cases(args.tasks,args.count)):
                composer=make_composer(args.seed+i*9973)
                row,trace,log=rollout(env,bank,case,planner_override=CubeBridge(composer))
                row['context']=i;rows.append(row);plans.append(log)
                np.savez_compressed(out/f'rollout_{i:04d}.npz',**trace,**case)
                (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
                (out/'planning.json').write_text(json.dumps(plans,indent=2)+'\n')
                print(json.dumps({'episode':i,'success':row['success'],'goal':row['released_success'],'waypoint':row['visited']}),flush=True)
            measured=summarize(rows,plans);measured['total_seconds']=time.time()-started
        finally:close_env(env)
    bank.assert_frozen()
    assert all(p.grad is None and not p.requires_grad for e in encoders for p in e.parameters())
    (out/'summary.json').write_text(json.dumps(measured,indent=2)+'\n')
    proof={'complete':True,'inference_simulator_queries':0,'networks_frozen':True}
    if args.adapter_only_consistency:
        proof.update(new_auxiliary_networks=0,inference_tripwires_enabled=True,
                     training_encoder_forwards_forbidden=True,execution='original_base_with_simultaneous_adapter_updates')
    (out/'DONE.json').write_text(json.dumps(proof)+'\n')
    print(json.dumps(measured,indent=2),flush=True)


if __name__=='__main__':main()
