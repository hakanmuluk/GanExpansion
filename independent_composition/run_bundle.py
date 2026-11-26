"""Run a frozen study bundle with its recorded weights and inference settings.

Run from the GanExpansion checkout root after downloading the original assets.
Use a validation bank for a small demo, or provide the sealed bank manifest for
the exact full final evaluation. No raw scene data is bundled with model files.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule',required=True);p.add_argument('--tasks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model-seed',type=int,default=42);p.add_argument('--method',default='compose')
    p.add_argument('--count',type=int);p.add_argument('--bank-manifest');p.add_argument('--device',default='cuda')
    p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if not Path('tworoom_gan').is_dir() or not Path('cube').is_dir():p.error('Run from the GanExpansion checkout root')
    cap=Path(args.capsule).resolve();protocol_path=cap/'protocol.json'
    if not protocol_path.exists():p.error('Pass a completed final-study capsule containing protocol.json')
    protocol=json.loads(protocol_path.read_text());available=[x['training_seed'] for x in protocol['model_pairs']]
    if args.model_seed not in available:p.error(f'Available model seeds: {available}')
    if args.method not in protocol['methods']:p.error(f"Available methods: {protocol['methods']}")
    pair=next(x for x in protocol['model_pairs'] if x['training_seed']==args.model_seed)
    files=[cap/'models'/Path(name).name for name in pair['ordered_weight_paths']]
    assert len(files) in [4,6]
    assert [digest_file(path) for path in files]==pair['ordered_weight_sha256'],'Bundle weights changed'
    asset='cube/artifacts/final/base_policy.pt' if protocol['environment']=='cube' else 'artifacts/final/base_policy.pt'
    if not Path(asset).exists():
        p.error(f"Download original assets first: python scripts/download_assets.py --bundle {protocol['environment']}")
    assert digest_file(asset)==protocol['base_sha256'],'Original base weights changed'
    if not Path(args.tasks).is_file():p.error('Task bank is missing; download validation assets or generate the sealed final bank')
    count=args.count if args.count is not None else (protocol.get('scene_count',64) if args.bank_manifest else 16)
    if args.bank_manifest and count!=protocol.get('scene_count',64):p.error('The sealed final scene count is fixed')
    command=[sys.executable,str(cap/'run_frozen.py'),'evaluate_private_paths']
    if protocol['environment']=='tworoom':command+=['--env','tworoom']
    command+=['--routes',str(files[0]),str(files[1]),'--controllers',str(files[2]),str(files[3])]
    if len(files)==6:command+=['--supports',str(files[4]),str(files[5])]
    command+=['--tasks',str(Path(args.tasks).resolve()),'--out',str(Path(args.out).resolve()),
              '--method',args.method,'--count',str(count),'--seed',str(protocol['evaluation_seed']),'--device',args.device]
    for name,value in protocol['inference_configuration'].items():
        flag='--'+name.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
        else:command.extend([flag,str(value)])
    if args.bank_manifest:command+=['--final-protocol',str(protocol_path),'--final-bank-manifest',str(Path(args.bank_manifest).resolve())]
    if args.dry_run:print(json.dumps({'command':command,'environment':protocol['environment'],'model_seed':args.model_seed},indent=2))
    else:subprocess.run(command,check=True)


if __name__=='__main__':main()
