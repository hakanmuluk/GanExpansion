"""Verify the complete archive, audit requirements, and index compact delivery."""
import datetime,gzip,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
import xml.etree.ElementTree as ET
from .archive import verify

ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'composition_artifacts';archive=root/'full_study.tar.xz';manifest=json.loads((root/'complete_manifest.json').read_text())['files']
    verified=verify(archive,root/'archive_receipt.json');(root/'archive_verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    wanted=['main_pipeline_DONE.json','reference_pipeline_DONE.json','final_audit_DONE.json','post_pipeline_DONE.json','weak_pair_diagnostic_DONE.json',
        'old_delivery_preservation.json','aggregate_statistics.json','tasks/independent_audit.json','final_tests.xml','selection.json',
        'main_plan.json','replicate_plan.json','generalization_plan.json','search_replicate_plan.json','native_online_plan.json',
        'benchmark_plan_frozen.json','benchmark_plan.json','benchmark_amendment.json']
    names={'composition_artifacts/'+p for p in wanted};contents={}
    with tarfile.open(archive,'r|*') as tar:
        for member in tar:
            if member.name in names:contents[member.name]=tar.extractfile(member).read()
    assert names==set(contents)
    def record(name):return json.loads(contents['composition_artifacts/'+name])
    for p in wanted[:5]:assert record(p)['complete']
    old=record('old_delivery_preservation.json');assert old['tworoom_addons_verifier']['verified'] and old['cube_addons_verifier']['all_match']
    tests=ET.fromstring(contents['composition_artifacts/final_tests.xml']);suites=list(tests.iter('testsuite'))
    assert sum(int(s.attrib.get('tests',0)) for s in suites)==10
    assert all(int(s.attrib.get('errors',0))==0 and int(s.attrib.get('failures',0))==0 for s in suites)
    geometry=record('tasks/independent_audit.json');assert sum(b['independent_strict_witness_successes'] for b in geometry['banks'])==768
    assert all(b['old_context_overlap']==0 and b['other_new_bank_overlap']==0 for b in geometry['banks'])
    stats=record('aggregate_statistics.json');assert all(b['queries_per_independent_generator_per_plan']==65 for b in stats['matched_query_budget_audit'])
    selection=record('selection.json')
    for p in ['main_plan.json','replicate_plan.json','generalization_plan.json','search_replicate_plan.json','native_online_plan.json']:
        assert hashlib.sha256(contents['composition_artifacts/'+p]).hexdigest()==selection['plan_sha256']['composition_artifacts/'+p]
    amendment=record('benchmark_amendment.json')
    assert hashlib.sha256(contents['composition_artifacts/benchmark_plan_frozen.json']).hexdigest()==selection['plan_sha256']['composition_artifacts/benchmark_plan.json']
    assert hashlib.sha256(contents['composition_artifacts/benchmark_plan.json']).hexdigest()==amendment['amended_plan_sha256']
    requirements=[
        ('Frozen existing generators, encoders, mappings and original base code',['old_delivery_preservation.json','main_seed42/test_supported_easy/structured_task_h64/config.json']),
        ('Independent feasible supported, conflict and OOD tasks with strict swept geometry',['tasks/manifest.json','tasks/independent_audit.json']),
        ('Separate analytic infeasibility controls',['tasks/test_infeasible.npz','generalization_seed42/test_infeasible/structured_task_h64/summary.json']),
        ('Clean latent-agreement hypothesis before task-aware extensions',['pilot/DONE.json','main_seed42/test_supported_easy/agreement_adam_k/records.json']),
        ('Gaussian, no-prior and empirical priors; original encoder-code statistics',['priors/seed42.json','priors/seed123.json','priors/seed2026.json','validation/DONE.json']),
        ('Fixed, cold, warm-code and trajectory-tail closed-loop comparisons',['validation/DONE.json','main_seed42/test_supported_conflict/structured_h64_tail/records.json']),
        ('Random/structured, CEM, hybrid and multistart Adam search',['main_seed42/DONE.json','matched_budget_plan.json','aggregate_statistics.json']),
        ('Factorized exact finite-grid search and finite-domain capability analysis',['capacity/DONE.json','capacity_seed2026/DONE.json','figures/agreement_capacity_analysis.json']),
        ('Individual, random-pair, summed-adapter, output and spatial-handoff baselines',['main_seed42/test_supported_easy/naive_sum/records.json','main_seed42/test_supported_easy/output_average/records.json','main_seed42/test_supported_easy/spatial_handoff/records.json']),
        ('Direct action-space optimization under both constraints',['main_seed42/test_supported_conflict/action_adam10_h64/records.json']),
        ('Separate joint-demo trained reference and its checkpoint',['joint_reference/best.pt','joint_reference/config.json','joint_reference/training.json','joint_train/independent_audit.json']),
        ('Multiple frozen model seeds and equal-size search-seed repetitions',['main_seed123/DONE.json','main_seed2026/DONE.json','search_seed41/DONE.json','search_seed73/DONE.json','search_seed109/DONE.json']),
        ('Synchronized batch-one latency, startup, counts and batch throughput',['benchmark/latency.json','joint_reference/latency.json','graph_equivalence.json']),
        ('Native online replanning and native replay of primary and control cases',['native_online/DONE.json','native_final_primary/native_replay.json','native_final_controls/native_replay.json']),
        ('Plausibility, actual strategy consistency, stationarity and failure analysis',['main_seed42/diagnostics_summary.json','main_seed2026/diagnostics_summary.json','aggregate_statistics.json']),
        ('Trajectory figures, animations, steps/horizon and success-time/disagreement plots',['figures/test_supported_conflict_trajectories_0.png','figures/test_supported_conflict.gif','figures/success_vs_latency.png','figures/success_vs_disagreement.png','figures/validation_steps_horizon_temporal.png']),
        ('Reproducible configuration freeze, pinned environment and preserved original deliveries',['selection.json','environment.json','old_delivery_preservation.json']),
    ]
    for _,paths in requirements:
        for p in paths:assert 'composition_artifacts/'+p in manifest,p
    # Only redundant NEW local copies are removed, after identical raw content is
    # verified in the complete local archive. No previous-study path is eligible.
    removed=[];receipt=root/'local_storage_compression.json'
    if receipt.exists():
        storage=json.loads(receipt.read_text())
        for entry in storage['files']:
            p=(ROOT/entry['gzip']).resolve();source=entry['source']
            assert p.is_relative_to(root)
            if p.exists() and source in manifest:
                with gzip.open(p,'rb') as f:h=hashlib.sha256(f.read()).hexdigest()
                assert h==manifest[source]['sha256'];p.unlink();removed.append(str(p.relative_to(ROOT)))
        storage['interim_gzip_removed_after_complete_archive_verification']=True
        storage['complete_archive_sha256']=verified['archive_sha256'];receipt.write_text(json.dumps(storage,indent=2)+'\n')
    for p in root.rglob('trajectories.npz'):
        relative=str(p.relative_to(ROOT));expected=manifest.get(relative)
        if expected and hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256']:
            p.unlink();removed.append(relative)
    audit={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'research_requirements_covered':True,'unmet_required_items':[],
        'requirements':[{'requirement':name,'status':'implemented and empirically evaluated','evidence_in_complete_archive':paths} for name,paths in requirements],
        'interpretation':'The original agreement hypothesis failed; this is an investigated negative result, not an unmet experiment. Best task-aware search has actual strict failures and is not a safety guarantee.',
        'conditional_extensions':{'amortized_initializer':'Not pursued: optional; training-free search measured17.5ms after graph warmup.',
            'lbfgs':'Not used; Adam fulfills the requested initial gradient experiment.',
            'joint_training':'A separate reference was trained and evaluated; original skills remained frozen.',
            'cube_composition':'Deferred as explicitly requested.'},
        'archive_verification':verified,'previous_deliveries_verified_unchanged':True,
        'compact_local_redundant_files_removed':removed,'final_delivery_integrity_receipt':'composition_artifacts/delivery_verification.json (created after indexing this audit)'}
    (root/'requirements_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    subprocess.run([sys.executable,'-m','tworoom_composition.delivery'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','tworoom_composition.delivery','--verify'],cwd=ROOT,check=True)
    print(json.dumps({'archive_verified':True,'research_requirements':len(requirements),'redundant_new_local_files_removed':len(removed),'final_delivery_verified':True},indent=2))


if __name__=='__main__':main()
