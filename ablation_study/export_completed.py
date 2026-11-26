"""Export only completed checkpoints, code, figures and aggregate evidence.

Never exports NPZs, per-episode records, observations, actions or simulator
snapshots. Compact checkpoints omit optimizer/RNG state but retain exact model
tensors and original source checksums. Full training checkpoints stay on host.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import torch


SLOTS=['generator','generator_ema','encoder','discriminator','addon','finetuned_base']


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(value):
    h=hashlib.sha256()
    for key,t in sorted(value.items()):
        if torch.is_tensor(t):h.update(key.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--status',default='ablation_results/training_queue_v2/status.json')
    p.add_argument('--out',required=True);p.add_argument('--previous-manifest',action='append',default=[])
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    previous={}
    for path in a.previous_manifest:
        for row in json.loads(Path(path).read_text())['models']:
            previous[row['original']]=row['original_sha256']
    status=json.loads(Path(a.status).read_text());manifest=json.loads(Path('ablation_study/queue_manifest.json').read_text())
    jobs={j['id']:j for j in manifest['jobs']};members=[];models=[];skipped=[]
    candidates={'tworoom_base_full_s42':Path('ablation_results/training/tworoom_base_full_s42')}
    for name,row in status['jobs'].items():
        if row['state']!='complete' or name.endswith('_initial_1000'):continue
        job=jobs.get(name)
        if job is None or job.get('invalidated'):continue
        for config in job.get('configs',[]):
            cfg=json.loads(Path(config).read_text())
            if 'run' in cfg and '/training/' in cfg['run']:candidates[name]=Path(cfg['run'])
            elif name.startswith('calibrate_'):candidates[name]=Path(cfg['out'])
    for name,folder in candidates.items():
        paths=[(kind,folder/(kind+'.pt')) for kind in ['best','latest']]
        # These are the exact files used by the primary Gaussian base tests.
        # Preserve their paths/hashes, rather than substituting latest aliases.
        if name.startswith('tworoom_base_'):
            paths.append(('step_005000',folder/'snapshots/step_005000.pt'))
        for kind,source in paths:
            if not source.exists():continue
            source_hash=file_hash(source)
            if previous.get(str(source))==source_hash:
                skipped.append({'original':str(source),'original_sha256':source_hash});continue
            ck=torch.load(source,map_location='cpu',weights_only=False)
            assert any(k in ck for k in ['generator','addon']),'Not a model checkpoint'
            remove=[k for k in ck if 'optimizer' in k or 'rng' in k.lower()]
            compact={k:v for k,v in ck.items() if k not in remove}
            original={k:state_hash(ck[k]) for k in SLOTS if k in ck}
            compact['export_provenance']={'source_checkpoint':str(source),'source_sha256':source_hash,
                'tensor_sha256':original,'optimizer_and_rng_omitted':True,'scope':'Exact inference tensors; original full resume checkpoint remains on execution host.'}
            dest=out/'checkpoints'/name/(kind+'.pt');dest.parent.mkdir(parents=True,exist_ok=True);torch.save(compact,dest)
            check=torch.load(dest,map_location='cpu',weights_only=False)
            assert all(state_hash(check[k])==h for k,h in original.items())
            models.append({'name':name,'kind':kind,'path':str(dest.relative_to(out)),'sha256':file_hash(dest),
                'original':str(source),'original_sha256':compact['export_provenance']['source_sha256'],'tensor_sha256':original})
            members.append(dest)
        # These are aggregate training/validation logs and model configurations.
        for pattern in ['config.json','controlled_ablation*.json','loss_removal.json','DONE.json','completed.json',
                        'train.jsonl','validation.jsonl','calibration.jsonl','effective_train.py','*.diff']:
            for source in folder.glob(pattern):
                dest=out/'training_evidence'/name/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(source.read_bytes());members.append(dest)
    # Completed diagnostic/benchmark outputs only; never their raw episode files.
    roots=[Path('ablation_results/rq1_tworoom_existing'),Path('ablation_results/rq1_cube_existing')]
    roots.extend(Path('ablation_results/evaluation').glob('*'));roots.extend(Path('ablation_results/figures').glob('*'))
    roots.extend(Path('ablation_results/baseline').glob('*'))
    for folder in roots:
        if not folder.is_dir() or not ((folder/'DONE.json').exists() or (folder/'figure_provenance.json').exists()):continue
        for pattern in ['summary.json','study_summary.json','metrics.json','config.json','figure_provenance.json',
                        'z2_magnitude_aggregate.csv','*.png','*.pdf']:
            for source in folder.glob(pattern):
                dest=out/'results'/folder.name/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(source.read_bytes());members.append(dest)
    for source in Path('ablation_study').rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix in ['.py','.json','.txt','.tex']:
            dest=out/'code'/source;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());members.append(dest)
    for source in Path('ablation_results/aggregate').rglob('*'):
        if not source.is_file() or source.suffix not in ['.json','.csv','.png','.pdf']:continue
        assert not any(s in source.name for s in ['private','records','rollouts'])
        dest=out/'aggregate'/source.relative_to('ablation_results/aggregate')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());members.append(dest)
    proof_files=['ablation_results/common_random_inputs_verified.json',
        'ablation_results/engineering_variants/checks.json','ablation_results/engineering_keepout_variants/checks.json',
        'ablation_results/invalid_rng_attempt_receipt.json']
    proof_files.extend(str(p) for p in Path('ablation_results/initialization').glob('*.json'))
    for name in proof_files:
        source=Path(name)
        if not source.is_file():continue
        dest=out/'controls'/source.relative_to('ablation_results');dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes());members.append(dest)
    # Preserve aggregate reconstruction evidence without exporting any failed
    # context identifiers, episode records or native simulator arrays.
    reconstruction=Path('ablation_recreated/cube_base_recovery_v35_data')
    if (reconstruction/'stage_b_inputs_verified.json').exists():
        summary=json.loads((reconstruction/'summary.json').read_text())
        verification=json.loads((reconstruction/'stage_b_inputs_verified.json').read_text())
        allowed=['scope','goal_contexts','goal_successes','recovery_requested','recovery_successes',
            'training_episodes','maximum_native_prefix_replay_error_m','train_sha256','source_sha256','seconds']
        value={'reconstruction':{k:summary[k] for k in allowed if k in summary},
            'stage_b_verification':{k:v for k,v in verification.items() if k!='stats'}}
        dest=out/'controls/cube_reconstruction_verified.json';dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(value,indent=2)+'\n');members.append(dest)
    rng_receipts=[]
    for source in Path('ablation_results/rng').glob('*.pt'):
        # The live recorder atomically replaces the stable path through this
        # temporary name. It can disappear between globbing and reading.
        if source.name.endswith('.tmp.pt'):continue
        content=source.read_bytes();rows=torch.load(io.BytesIO(content),map_location='cpu',weights_only=False)
        assert all(set(row)=={'cpu','cuda','batch_sha256'} for row in rows.values())
        assert all(torch.is_tensor(row['cpu']) and row['cpu'].dtype==torch.uint8 for row in rows.values())
        dest=out/'reference_rng'/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(content);members.append(dest)
        rng_receipts.append({'path':str(dest.relative_to(out)),'steps':len(rows),'sha256':file_hash(dest),
            'scope':'Random-generator states and SHA256 minibatch fingerprints only; no observation/action tensors.'})
    receipt={'models':models,'previous_manifest_files':a.previous_manifest,'models_omitted_previously_exported':skipped,
        'reference_rng':rng_receipts,'raw_episode_data_exported':False,'queue_status':status,
        'file_sha256':{str(p.relative_to(out)):file_hash(p) for p in sorted(set(members))}}
    (out/'EXPORT_MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n');members.append(out/'EXPORT_MANIFEST.json')
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as bundle:
        for source in sorted(set(members)):bundle.add(source,arcname=str(source.relative_to(out)),recursive=False)
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':file_hash(archive),
        'files':len(set(members)),'model_checkpoints':len(models),'raw_episode_data_exported':False}),flush=True)


if __name__=='__main__':main()
