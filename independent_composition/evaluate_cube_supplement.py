"""Frozen original-model controls and instruction-paired evaluation.

No tuning or case selection using these outcomes. Parent freeze remains intact;
the supplemental methods and pairing definitions are recorded separately before
construction. All native execution is outside GuardedPolicy.sample.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .amplitude_composition import AmplitudeConfig, AmplitudeComposer
from .cube_checkpoint_bank import CheckpointBank
from .fixed_original_policy import FixedOriginalPolicy
from .evaluate_diffusion import evaluate_cube, summarize
from .inference_guard import GuardedPolicy
from .legacy import digest_file


def verify(a):
    s = json.loads(Path(a.supplement).read_text())
    parent = json.loads(Path(s['parent_protocol']).read_text())
    bank = json.loads(Path(s['parent_bank_manifest']).read_text())
    assert digest_file(s['parent_protocol']) == s['parent_protocol_sha256']
    assert digest_file(s['parent_bank_manifest']) == s['parent_bank_manifest_sha256']
    assert bank['protocol_sha256'] == s['parent_protocol_sha256'] and bank['created'] > parent['created']
    assert a.seed == parent['evaluation_seed'] and a.execute == parent['execute']
    assert digest_file(a.config) == parent['configuration_sha256']
    assert digest_file(a.base) == parent['base_sha256']
    pair = [digest_file(a.keepout), digest_file(a.waypoint)]
    assert pair in list(parent['model_pairs'].values())
    assert a.method in s['methods']
    if a.method != 'composition':
        assert pair == parent['model_pairs']['prefix']
    for name, checksum in parent['source_sha256'].items():
        assert digest_file(Path(__file__).parent / name) == checksum
    for name, checksum in parent['original_dependencies'].items():
        assert digest_file(name) == checksum
    for name, checksum in s['new_source_sha256'].items():
        assert digest_file(Path(__file__).parent / name) == checksum
    manifest = json.loads(Path(a.bank_manifest).read_text())
    if a.paired:
        assert manifest['supplement_sha256'] == digest_file(a.supplement)
        assert manifest['created'] > s['created']
        assert manifest['paired_variable'] == a.paired_variable
        assert manifest['all_constructed_cases_retained']
        assert not manifest['learned_policy_outcomes_used_for_selection']
        definition = s['paired_definitions'][a.paired_variable]
        assert manifest['bank_sha256'] == definition['bank_sha256']
        assert manifest['cases'] == a.count == 2 * definition['worlds']
        assert Path(a.tasks).resolve() == (Path(a.bank_manifest).parent / 'val_paired_keepout.npz').resolve()
        assert digest_file(a.tasks) == manifest['cases_sha256']
    else:
        assert digest_file(a.bank_manifest) == s['parent_bank_manifest_sha256']
        entry = next(v for v in bank['strata'].values() if Path(v['path']).resolve() == Path(a.tasks).resolve())
        assert a.count == entry['count'] == 64 and digest_file(a.tasks) == entry['sha256']
    return {'supplement_sha256': digest_file(a.supplement),
            'parent_freeze_sha256': s['parent_protocol_sha256'], 'frozen_weights_sources_configuration_verified': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ['tasks', 'config', 'out', 'base', 'keepout', 'waypoint', 'supplement', 'bank-manifest']:
        p.add_argument('--' + flag, required=True)
    p.add_argument('--method', required=True,
                   choices=['composition', 'base', 'keepout_mean', 'keepout_prior', 'waypoint_mean', 'waypoint_prior'])
    p.add_argument('--paired', action='store_true')
    p.add_argument('--paired-variable', choices=['keepout', 'waypoint'], default='keepout')
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--execute', type=int, default=8); p.add_argument('--seed', type=int, default=241)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); proof = verify(a); out = Path(a.out); assert not out.exists()
    out.mkdir(parents=True); torch.set_num_threads(2)
    with np.load(a.tasks) as x:
        assert len(x['seed']) == a.count
        pairs = x['paired_id'].astype(int) if a.paired else np.arange(a.count)
        if a.paired:
            assert np.array_equal(pairs[::2], pairs[1::2])
            fixed = 'waypoint' if a.paired_variable == 'keepout' else 'keepout'
            for key in ['initial_state', 'initial_cube', 'goal', 'base_z', fixed]:
                assert np.array_equal(x[key][::2], x[key][1::2]), key
    bank = CheckpointBank(a.keepout, a.waypoint, a.base, a.device)
    cfg = AmplitudeConfig(**json.loads(Path(a.config).read_text())); assert cfg.merge == 'waypoint'
    cursor = iter(pairs)

    def factory(args, unused, base, ignored_seed):
        seed = a.seed + 9973 * int(next(cursor))
        policy = (AmplitudeComposer(bank, cfg, seed) if a.method == 'composition'
                  else FixedOriginalPolicy(bank, a.method, seed))
        return GuardedPolicy(policy)

    (out / 'frozen_proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'config.json').write_text(json.dumps({'args': vars(a), 'models': bank.files,
        'task_sha256': digest_file(a.tasks), 'private_code_scope': 'Fixed zero or one independent standard-normal draw per episode; same draw within each physical pair; no search or outcome selection'}, indent=2) + '\n')
    source = out / 'source'; source.mkdir()
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f, source / f.name)
    started = time.time(); rows = evaluate_cube(a, [], bank.k.base, out, factory)
    metrics = summarize(rows); metrics['total_seconds'] = time.time() - started
    if a.paired:
        metrics['paired_worlds'] = a.count // 2
        metrics['both_variants_successful_worlds'] = sum(rows[i]['success'] and rows[i + 1]['success'] for i in range(0, a.count, 2))
    bank.assert_frozen()
    assert all(digest_file(path) == checksum for path, checksum in bank.files.items())
    (out / 'summary.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (out / 'DONE.json').write_text(json.dumps({'episodes': len(rows), 'weights_unchanged': True,
        'inference_environment_queries': 0, 'inference_tripwires_enabled': True, 'new_learned_modules': 0}) + '\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
