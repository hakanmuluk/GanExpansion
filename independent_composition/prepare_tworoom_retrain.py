"""Exact independent retraining manifest for the selected TwoRoom v12 recipe."""
import argparse
import json
from pathlib import Path
import sys
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--capsule',required=True);p.add_argument('--evaluate',action='store_true')
    args=p.parse_args();out=Path(args.out);manifest=Path(args.manifest);cap=Path(args.capsule)
    if out.exists() or manifest.exists():raise FileExistsError('Fresh retraining output and manifest required')
    freeze=json.loads((cap/'freeze.json').read_text());assert freeze['environment']=='tworoom'
    for name,expected in freeze['source_sha256'].items():
        assert digest_file(cap/'independent_composition'/name)==expected
    root=Path('independent_composition_artifacts');jobs=[]

    def add(label,module,arguments,result):
        jobs.append({'id':label,'command':[sys.executable,'-u',str(cap/'run_frozen.py'),module,*arguments],
                     'result':str(result)})

    for skill,pretrain_steps in [('keepout',15000),('waypoint',10000)]:
        routes=root/'data/tworoom_known_routes_v8'/skill
        controls=root/'data/tworoom_diverse_merged_v5'/skill
        for data in [routes,controls]:
            assert (data/'train.npz').exists() and (data/'val.npz').exists()
        common=['--skill',skill,'--seed',str(args.seed)]
        route_data=['--train',str(routes/'train.npz'),'--validation',str(routes/'val.npz')]
        pre=out/'models/routes_pretrain'/skill
        add('pretrain_route_'+skill,'train_tworoom_routes',
            [*common,*route_data,'--out',str(pre),'--steps',str(pretrain_steps)],pre/'DONE.json')
        route=out/'models/routes'/skill
        add('finetune_route_'+skill,'finetune_tworoom_routes',
            [*common,*route_data,'--initial',str(pre/'latest.pt'),'--out',str(route),'--steps','15000'],route/'DONE.json')
        controller=out/'models/controllers'/skill
        add('train_controller_'+skill,'train_tworoom_controller',
            [*common,'--train',str(controls/'train.npz'),'--validation',str(controls/'val.npz'),
             '--out',str(controller),'--steps','10000'],controller/'DONE.json')
        support_pre=out/'models/support_pretrain'/skill
        add('pretrain_support_'+skill,'train_path_support',
            [*common,*route_data,'--route',str(route/'best.pt'),'--out',str(support_pre),'--steps','6000'],support_pre/'DONE.json')
        support=out/'models/supports'/skill
        add('finetune_support_'+skill,'train_path_support_continuous',
            [*common,*route_data,'--route',str(route/'best.pt'),'--initial',str(support_pre/'latest.pt'),
             '--out',str(support),'--steps','15000'],support/'DONE.json')
    if args.evaluate:
        for kind in ['supported_easy','supported_conflict','coupled_ood']:
            target=out/'results'/kind
            add('validate_'+kind,'evaluate_private_paths',
                ['--env','tworoom','--routes',*[str(out/'models/routes'/s/'best.pt') for s in ['keepout','waypoint']],
                 '--controllers',*[str(out/'models/controllers'/s/'best.pt') for s in ['keepout','waypoint']],
                 '--supports',*[str(out/'models/supports'/s/'best.pt') for s in ['keepout','waypoint']],
                 '--tasks',str(root/'validation_inputs/tworoom_extension_v9b'/f'val_{kind}_extension.npz'),
                 '--out',str(target),'--count','112','--method','compose','--agreement','pointwise',
                 '--executor','1','--support-weight','.03','--prior','.0001','--code-bound','3',
                 '--iterations','120','--starts','8','--execute','1','--seed','241','--guard-inference'],target/'summary.json')
    record={'args':vars(args),'source_capsule_sha256':digest_file(cap/'freeze.json'),
        'manifest_source_sha256':digest_file(__file__),'jobs':jobs,
        'scope':'Independent RNG/minibatches and all six private learned modules. Same original pretrained base and same own-only training data. Fixed recipe before replication; no final tests and no joint training labels.',
        'fixed_inference':{'agreement':'pointwise','executor':1,'support_weight':.03,'prior':.0001,
                           'code_bound':3.,'iterations':120,'starts':8,'execute':1},
        'prototype_initialization':'Fixed route pretraining steps K15000/W10000 and support pretraining6000 use latest.pt, matching seed42 exact initialization steps. Subsequent models select best own-only validation checkpoint.'}
    manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'manifest':str(manifest),'jobs':len(jobs),'seed':args.seed}))


if __name__=='__main__':main()
