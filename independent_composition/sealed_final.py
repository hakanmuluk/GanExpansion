"""Receipt checks for fresh final evaluation after freezing method and weights."""
import json
from pathlib import Path
from .legacy import digest_file


def verify_evaluation(args):
    assert args.final_protocol and args.final_bank_manifest,'Final evaluation requires freeze and bank receipts'
    protocol=json.loads(Path(args.final_protocol).read_text())
    bank=json.loads(Path(args.final_bank_manifest).read_text())
    assert protocol['environment']==args.env
    assert bank['protocol_sha256']==digest_file(args.final_protocol)
    assert bank['created']>protocol['created']
    assert bank['selection']=='Geometry and independent native witness only; no learned-policy outcome filtering'
    assert len(bank['strata'])==3
    for name,expected in protocol['source_sha256'].items():
        assert digest_file(Path(__file__).parent/name)==expected,('Frozen source changed',name)
    for name,expected in protocol.get('repository_dependencies',{}).items():
        assert digest_file(name)==expected,('Original dependency changed',name)
    actual=[digest_file(p) for p in [*args.routes,*args.controllers,*(getattr(args,'encoders',None) or []),
                                    *(getattr(args,'supports',None) or [])]]
    assert actual in [pair['ordered_weight_sha256'] for pair in protocol['model_pairs']]
    assert args.guard_inference
    if args.env=='tworoom':
        assert {'agreement','cycle_mode','cycle_weight','code_bound','support_weight'}<=protocol['inference_configuration'].keys()
        assert args.count==protocol['scene_count']
    for key,value in protocol['inference_configuration'].items():
        assert getattr(args,key)==value,('Frozen inference setting changed',key)
    assert args.method in protocol['methods']
    assert args.seed==protocol['evaluation_seed']
    task=Path(args.tasks).resolve()
    entry=next(b for b in bank['strata'].values() if Path(b['path']).resolve()==task)
    assert digest_file(task)==entry['sha256']
    assert args.count==entry['count'] and args.count>=64

