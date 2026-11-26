"""Check immutable inputs, one-factor configurations and finished-run receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def differences(a,b,ignore):
    return {k:[a.get(k),b.get(k)] for k in set(a)|set(b) if k not in ignore and a.get(k)!=b.get(k)}


def audit_auxiliary(name,cfg,failures):
    """Audit original Cube collection/calibration without exporting episodes."""
    folder=Path(cfg['out'])
    proof=json.loads((folder/'original_procedure_verified.json').read_text())
    expected_module='cube_addons.feedback' if name.startswith('physical_') else 'cube_addons.calibrate'
    if proof.get('module')!=expected_module or not proof.get('variant_loading_only') or not proof.get('original_objective_and_sampling_unchanged'):
        failures.append({'invalid_original_auxiliary_procedure':name})
    if name.startswith('physical_'):
        receipt=json.loads((folder/'summary.json').read_text())
        delta={k:[v,receipt['args'].get(k)] for k,v in cfg.items() if receipt['args'].get(k)!=v}
        valid=(not delta and receipt['actual_learner_rollouts']==cfg['count']*cfg['codes']
               and receipt['expert_action_queries']==0 and receipt['feedback_chunks']==0
               and receipt['generator_checkpoint_sha256']==digest(cfg['checkpoint']))
        result={'job':name,'verified_rollouts':receipt['actual_learner_rollouts'],'valid':valid}
    else:
        receipt=json.loads((folder/'config.json').read_text())
        done=json.loads((folder/'DONE.json').read_text())
        delta={k:[v,receipt['args'].get(k)] for k,v in cfg.items() if receipt['args'].get(k)!=v}
        log=[json.loads(line) for line in (folder/'calibration.jsonl').read_text().splitlines() if line.strip()]
        physical=json.loads((Path(cfg['physical_train'])/'summary.json').read_text())
        validation=json.loads((folder/'validation_rollouts'/'summary.json').read_text())
        actor_sha=digest(cfg['checkpoint'])
        valid=(not delta and [r['step'] for r in log]==list(range(cfg['eval_every'],cfg['steps']+1,cfg['eval_every']))
               and done.get('generator_unchanged') is True and done.get('normalization_unchanged') is True
               and receipt['physical_training_rollouts']==physical['actual_learner_rollouts']==512
               and validation['episodes']==cfg['val_count']*cfg['val_codes']
               and receipt['extra_expert_queries_in_calibration']==0
               and receipt['generator_checkpoint_sha256']==physical['generator_checkpoint_sha256']==actor_sha)
        result={'job':name,'verified_updates':log[-1]['step'] if log else None,
                'physical_training_rollouts':receipt['physical_training_rollouts'],
                'physical_validation_rollouts':validation['episodes'],'valid':valid}
    if not valid:failures.append({'invalid_auxiliary_receipt':name,'configuration_differences':delta,'receipt_check':result})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--require-complete',action='store_true');p.add_argument('--inputs-only',action='store_true')
    args=p.parse_args();out=Path(args.out);manifest=json.loads(Path('ablation_study/queue_manifest.json').read_text())
    original=json.loads(Path('ablation_study/bootstrap_manifest.json').read_text())['files']
    original.update(json.loads(Path('ablation_study/adapter_input_manifest.json').read_text()))
    failures=[];checked=0
    for file,expected in original.items():
        if file.startswith('ablation_study/'):continue
        checked+=1
        if not Path(file).is_file() or digest(file)!=expected:failures.append({'immutable_input':file})
    if args.inputs_only:
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps({'immutable_input_files_checked':checked,'failures':failures,'verified':not failures},indent=2)+'\n')
        print(json.dumps({'immutable_input_files_checked':checked,'failures':failures}),flush=True)
        if failures:raise SystemExit(1)
        return
    configurations={p.stem:json.loads(p.read_text()) for p in Path('ablation_study/configs').glob('*.json')}
    config_checks=[]
    for name,cfg in configurations.items():
        match=re.fullmatch(r'(tworoom|cube)_(keepout|waypoint)_(r\d+(?:_lora_only)?)_s(\d+)(_initial_1000)?',name)
        if not match:continue
        env,skill,variant,seed,phase=match.groups();ref=4 if env=='tworoom' and skill=='keepout' else 8
        reference=f'{env}_{skill}_r{ref}_s{seed}'+(phase or '')
        assert reference in configurations,(name,reference)
        delta=differences(cfg,configurations[reference],{'run','rank','resume'})
        if delta:failures.append({'configuration':name,'unintended_differences':delta})
        config_checks.append({'name':name,'reference':reference,'only_rank_and_destination_differ':not delta})
    for env in ['tworoom','cube_original','cube_latest']:
        for seed in ([42,123] if env=='tworoom' else [42,43]):
            a=f'{env}_base_full_s{seed}';b=f'{env}_base_adversarial_s{seed}'
            delta=differences(configurations[a],configurations[b],{'run','resume'})
            if delta:failures.append({'base_configurations':[a,b],'unintended_differences':delta})
            config_checks.append({'name':b,'reference':a,'same_original_hyperparameters':not delta})
    status=json.loads(Path('ablation_results/training_queue_v2/status.json').read_text())
    active_ids={j['id'] for j in manifest['jobs']}
    history={name:row for name,row in status['jobs'].items() if name not in active_ids}
    correction=Path('ablation_study/cube_known_rec_correction.json')
    allowed_history=set(json.loads(correction.read_text())['invalidated_jobs']) if correction.exists() else set()
    for name in history:
        if name not in allowed_history:failures.append({'unknown_orphaned_job':name})
    states={state:[name for name,row in status['jobs'].items() if name in active_ids and row['state']==state] for state in ['complete','running','failed']}
    pending=[j['id'] for j in manifest['jobs'] if j['id'] not in status['jobs']]
    failures.extend({'failed_job':name} for name in states['failed'])
    receipts=[];auxiliary_receipts=[]
    for name in states['complete']:
        row=status['jobs'][name]
        if not row.get('configuration_unchanged',False):failures.append({'configuration_changed_during_training':name})
        job=next(j for j in manifest['jobs'] if j['id']==name)
        if job.get('invalidated'):continue
        for path in job.get('configs',[]):
            cfg=json.loads(Path(path).read_text())
            if name.startswith(('physical_cube_','calibrate_cube_')):
                auxiliary_receipts.append(audit_auxiliary(name,cfg,failures))
                continue
            if 'run' not in cfg:continue
            folder=Path(cfg['run']);receipt=folder/'controlled_ablation.json'
            # Keepout's first and second phases share a destination. Their
            # immutable per-phase receipts, when present, avoid overwriting.
            first=1001 if cfg.get('resume') and name.startswith('cube_keepout_') and not name.endswith('_initial_1000') else 1
            phase=folder/f'controlled_ablation_steps_{first}_{cfg["steps"]}.json'
            if phase.exists():receipt=phase
            if not receipt.exists():failures.append({'missing_training_receipt':name});continue
            r=json.loads(receipt.read_text())
            expected=cfg['steps']-first+1
            count=r.get('sampled_batches_verified',r.get('sampled_batches_exactly_match_reference'))
            if 'first_step' in r:
                valid=count==expected and r['first_step']==first and r['last_step']==cfg['steps']
            else:
                log=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines() if line.strip()]
                valid=count==expected and log[-1]['step']==cfg['steps']
            if not valid and name.endswith('_initial_1000') and r.get('last_step')==4000:
                # Older wrapper wrote only the final shared receipt. The queue
                # still records the completed immutable1000-step configuration.
                valid=False
            if not valid:failures.append({'invalid_training_budget':name,'receipt':r,'expected_batches':expected})
            receipts.append({'job':name,'verified_batches':count,'valid':valid})
            if 'base_adversarial_' in name:
                removal=json.loads((folder/'loss_removal.json').read_text())['changes']
                if name.startswith('cube_'):
                    if removal.get('known_rec_weight',{}).get('ablation')!=0:failures.append({'known_reconstruction_not_removed':name})
                active=['rec','kl','latent','style','known_rec','feedback','physical_style'] if name.startswith('cube_') else ['rec','kl','latent','rollout_latent','feedback_rec','style_aux','prior_rec','prior_match']
                for line in (folder/'train.jsonl').read_text().splitlines():
                    row=json.loads(line)
                    for key in active:
                        if key in row and row[key]!=0:failures.append({'removed_objective_nonzero':name,'step':row['step'],'term':key,'value':row[key]})
    # The first full TwoRoom reference was launched before the queue. Its
    # terminal receipt and exact saved arguments need the same final checks.
    external_name='tworoom_base_full_s42'
    cfg=configurations[external_name];folder=Path(cfg['run'])
    receipt=json.loads((folder/'controlled_ablation.json').read_text())
    done=json.loads((folder/'DONE.json').read_text())
    saved=json.loads((folder/'config.json').read_text())['args']
    delta={k:[v,saved.get(k)] for k,v in cfg.items() if saved.get(k)!=v}
    valid=(not delta and receipt.get('sampled_batches_verified')==cfg['steps']
           and receipt.get('first_step')==1 and receipt.get('last_step')==cfg['steps']
           and receipt.get('module')=='tworoom_gan.train' and receipt.get('rng_mode')=='record'
           and done.get('step')==cfg['steps'])
    if not valid:failures.append({'invalid_external_reference':external_name,'configuration_differences':delta})
    receipts.append({'job':external_name,'verified_batches':receipt.get('sampled_batches_verified'),'valid':valid,'launched_before_queue':True})
    if args.require_complete and (pending or states['running']):failures.append({'incomplete_jobs':pending+states['running']})
    result={'immutable_input_files_checked':checked,'configuration_checks':config_checks,'training_receipts':receipts,
        'auxiliary_receipts':auxiliary_receipts,
        'job_counts':{k:len(v) for k,v in states.items()},'invalidated_history':history,'pending':pending,'failures':failures,
        'complete':not pending and not states['running'] and not failures,
        'scope':'Controls and execution audit; scientific interpretation still requires the actual held-out results.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checked_inputs':checked,'configuration_checks':len(config_checks),'receipts':len(receipts),
        'auxiliary_receipts':len(auxiliary_receipts),'counts':result['job_counts'],'pending':len(pending),'failures':failures}),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
