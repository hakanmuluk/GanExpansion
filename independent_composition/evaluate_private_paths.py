"""Native Cube validation of completely private route/action modules."""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from .path_policy import load_route,PrivatePathPolicy
from .private_paths import load_controller
from .evaluate_diffusion import evaluate_cube,summarize
from .train_diffusion import load_base
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--routes',nargs=2,required=True)
    p.add_argument('--env',choices=['cube','tworoom'],default='cube')
    p.add_argument('--controllers',nargs=2,required=True);p.add_argument('--tasks',required=True)
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=16)
    p.add_argument('--method',choices=['base','keepout','waypoint','fixed','compose','agreement'],default='compose')
    p.add_argument('--executor',type=int,choices=[0,1],default=0)
    p.add_argument('--iterations',type=int,default=120);p.add_argument('--starts',type=int,default=8)
    p.add_argument('--execute',type=int,default=1);p.add_argument('--prior',type=float,default=.0001)
    p.add_argument('--lr',type=float,default=.12);p.add_argument('--seed',type=int,default=41)
    p.add_argument('--agreement',choices=['pointwise','chamfer','hybrid'],default='pointwise')
    p.add_argument('--encoders',nargs=2);p.add_argument('--cycle-mode',choices=['none','self','executed','projection'],default='none')
    p.add_argument('--cycle-weight',type=float,default=0.);p.add_argument('--code-bound',type=float,default=3.)
    p.add_argument('--supports',nargs=2);p.add_argument('--support-weight',type=float,default=0.)
    p.add_argument('--device',default='cuda');p.add_argument('--guard-inference',action='store_true')
    p.add_argument('--final-protocol');p.add_argument('--final-bank-manifest')
    args=p.parse_args();torch.set_num_threads(2)
    if args.env=='tworoom' and args.execute!=1:
        raise ValueError('TwoRoom private followers supervise one control; execute must be1')
    if args.env=='cube' and args.agreement!='pointwise':
        raise ValueError('Frozen Cube recipe uses pointwise agreement')
    if args.env=='cube' and args.method=='agreement':
        raise ValueError('Named domain-disabled ablation is TwoRoom-only')
    if args.env=='cube' and (args.encoders or args.cycle_mode!='none' or args.code_bound!=3. or args.supports or args.support_weight):
        raise ValueError('New full-route encoder experiments are TwoRoom-only')
    is_final=Path(args.tasks).name.startswith(('test','final'))
    if is_final or args.final_protocol or args.final_bank_manifest:
        from .sealed_final import verify_evaluation
        verify_evaluation(args)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh path evaluation output required')
    out.mkdir(parents=True);routes=[];controllers=[];provenance=[]
    base,base_source=load_base(args.env,args.device)
    if args.env=='tworoom':
        from .tworoom_paths import load_controller as controller_loader
        from .tworoom_path_policy import TwoRoomPathPolicy as Policy
        from .evaluate_diffusion import evaluate_tworoom as evaluate
    else:
        controller_loader=load_controller;Policy=PrivatePathPolicy;evaluate=evaluate_cube
    for skill,rpath,cpath in zip(['keepout','waypoint'],args.routes,args.controllers):
        route,rmeta=load_route(rpath,args.device);controller,cmeta=controller_loader(cpath,args.device)
        assert rmeta['skill']==skill and cmeta['skill']==skill
        assert cmeta['base_sha256']==base_source['sha256']
        assert cmeta['config']['args']['skill']==skill
        routes.append(route);controllers.append(controller)
        provenance.append({'skill':skill,'route':{'path':rpath,'sha256':digest_file(rpath),'step':rmeta['step']},
                           'controller':{'path':cpath,'sha256':digest_file(cpath),'step':cmeta['step']}})
    encoders=[];encoder_provenance=[]
    if args.encoders:
        from .path_encoder import load_path_encoder
        for skill,path,rpath in zip(['keepout','waypoint'],args.encoders,args.routes):
            encoder,saved=load_path_encoder(path,args.device)
            assert saved['skill']==skill and saved['config']['route_sha256']==digest_file(rpath)
            encoders.append(encoder);encoder_provenance.append({'path':path,'sha256':digest_file(path)})
    supports=[];support_provenance=[]
    if args.supports:
        from .path_support import load_path_support
        for skill,path,rpath in zip(['keepout','waypoint'],args.supports,args.routes):
            support,saved=load_path_support(path,args.device)
            assert saved['skill']==skill and saved['config']['route_sha256']==digest_file(rpath)
            supports.append(support);support_provenance.append({'path':path,'sha256':digest_file(path)})
    config={'args':vars(args),'models':provenance,'encoders':encoder_provenance,'supports':support_provenance,
            'effective_support_weight':0. if args.method=='agreement' else args.support_weight,
            'base':base_source,'task_sha256':digest_file(args.tasks),
            'scope':'No newly shared learned module: keepout and waypoint own separate trajectory decoders, separate LoRA/FiLM action adapters, optional separate full-route encoders and optional separate learned domain models. Original base remains frozen. Composition optimizes predicted-trajectory agreement, code priors and optional frozen learned cycle/domain terms; never analytic task costs or simulator outcomes.',
            'representation':'A common fixed interpolation prior and physical pose coordinates; these have no learned shared weights. No analytic constraint repair is applied to generated routes.',
            'inference_inputs':'Current numerical state, each own region/context, original behavior code and retained predicted route. No environment handle, future state, branch rollout or witness is passed to the policy.',
            'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');(out/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    from cube_addons.models import state_digest
    initial_digest=state_digest(*routes,*controllers,*encoders,*supports)
    def factory(settings,unused_models,unused_base,seed):
        extra={'agreement':settings.agreement,'encoders':encoders,'cycle_mode':settings.cycle_mode,
               'cycle_weight':settings.cycle_weight,'code_bound':settings.code_bound,
               'supports':supports,'support_weight':0. if settings.method=='agreement' else settings.support_weight} if settings.env=='tworoom' else {}
        method='compose' if settings.method=='agreement' else settings.method
        policy=Policy(routes,controllers,method,settings.executor,
                                 settings.iterations,settings.starts,seed,settings.prior,settings.lr,**extra)
        if settings.guard_inference:
            from .inference_guard import GuardedPolicy
            policy=GuardedPolicy(policy)
        return policy
    started=time.time();rows=evaluate(args,routes,base,out,factory)
    result=summarize(rows);result['total_seconds']=time.time()-started
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    assert digest_file(base_source['path'])==base_source['sha256']
    assert all(p.grad is None and not p.requires_grad for m in [base,*routes,*controllers,*encoders,*supports] for p in m.parameters())
    assert state_digest(*routes,*controllers,*encoders,*supports)==initial_digest
    for item in encoder_provenance:assert digest_file(item['path'])==item['sha256']
    for item in support_provenance:assert digest_file(item['path'])==item['sha256']
    for record in provenance:
        for kind in ['route','controller']:assert digest_file(record[kind]['path'])==record[kind]['sha256']
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'inference_environment_queries':0,
                                           'new_shared_learned_networks':0,'models_unchanged':True,
                                           'inference_tripwires_enabled':args.guard_inference})+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
