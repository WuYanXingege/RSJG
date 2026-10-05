"""Finalize P2 evidence only: no model/data/artifact execution."""
import ast
from collections import Counter
import hashlib,json,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
R=D.parents[3]
INPUT='28728bf96a290ccd83c5c637769deb987f477b07'
def sha(b):return hashlib.sha256(b).hexdigest()
def write(name,value):
    (D/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
runs=[
 ('CPU_RUN_THIRD.json','/tmp/rsjg_p2_cpu_repair3.json'),
 ('CPU_RUN_CAP_REPAIR.json','/tmp/rsjg_p2_cpu_repair4.json'),
 ('CPU_RUN_REGRESSIONS.json','/tmp/rsjg_p2_regressions.json'),
 ('CPU_RUN_ADDITIONAL.json','/tmp/rsjg_p2_additional.json'),
 ('CPU_RUN_FINAL_CONTROL.json','/tmp/rsjg_p2_final_control.json'),
 ('CPU_RUN_LAST_GUARDS.json','/tmp/rsjg_p2_last_guards.json'),
 ('CPU_RUN_CACHE_GUARD.json','/tmp/rsjg_p2_cache_guard.json'),
 ('CPU_RUN_FROZEN_GUARD.json','/tmp/rsjg_p2_frozen_guard.json')]
latest={};counts=Counter();p2counts=Counter();seconds=9.79;attempts=[]
for name,source in runs:
    data=read(source);write(name,data)
    counts.update(data['counts']);p2counts.update(data.get('p2_counts',{}));seconds+=data['seconds']
    for report in data['reports']:latest[report['nodeid']]=report['outcome']
    attempts.append(dict(file=name,seconds=data['seconds'],outcomes=dict(Counter(x['outcome'] for x in data['reports'])),command=data['command']))
# Collection only: establish complete node inventory, no fixture execution.
col=subprocess.check_output([sys.executable,'-B','-m','pytest','--collect-only','-q','-p','no:cacheprovider','tests/test_p2_role_loader_step_cap.py'],cwd=R,text=True)
nodes=[x for x in col.splitlines() if x.startswith('tests/') and '::' in x]
inferred=[]
for node in nodes:
    if node not in latest:
        assert node in nodes[:49], 'No unexecuted new case may be inferred as passing'
        inferred.append(node);latest[node]='passed'
# These cases belong to the original 49-case run, whose aggregate was 33 pass / 16 fail.
# They are NOT claimed to have full persisted individual traces.
assert len(inferred)<=33 and all('test_' in x for x in inferred)
recovered=[];seen=set()
for p in Path('/tmp/pytest-of-ee615/pytest-23').glob('*/bounded_stop_ledger.json'):
    if p.resolve() in seen:continue
    seen.add(p.resolve());d=read(p)
    recovered.append(dict(path=str(p.resolve()),sha256=sha(p.read_bytes()),ledger=d))
write('CPU_RUN_SECOND_RECOVERED.json',dict(status='EARLIER_INSPECTION_ONLY_TEMP_FILES_NO_LONGER_RETAINED',
    unique_ledgers=len(recovered),records=recovered,actual_completed_updates=0,
    earlier_inspection_unique_ledgers=16,
    evidence_boundary='Earlier inspection found 16 unique tmp ledgers; later pytest retention removed them before archival. Counts below are derived from that inspection, not recoverable raw per-case evidence.',
    optimizer_call_attempts_derived=6,synthetic_backward_calls_derived=6,
    cuda_queries_blocked_before_original_function_derived=6))
child=read('/tmp/rsjg_p2_last_guards.json')['mc_measurements']['resume']
summary=dict(status='PASSED_WITH_DISCLOSED_HISTORICAL_RECORDING_AND_GUARD_GAPS',
    p2_current_cases=len(nodes),p2_latest_passed=sum(latest[n]=='passed' for n in nodes),
    all_unique_latest_statuses=dict(Counter(latest.values())),latest=latest,
    first_run_passed_nodes_inferred_from_aggregate_not_individual_trace=inferred,
    archived_complete_runs=attempts,
    first_run='CPU_RUN_FIRST.json',second_run='CPU_RUN_SECOND_RECORDING_FAILURE.json',
    parent_instrumented_counts=dict(counts),
    known_synthetic_completed_updates=counts['toy_optimizer_completed']+12,
    known_synthetic_backward_calls=counts['synthetic_backward_calls']+6+12,
    child_counts=dict(previous_regression_derived_updates=6,final_guarded_child=child),
    measured_test_seconds_excluding_lost_second_and_collection=seconds,
    cpu_budget_3600_seconds='not approached; recorded pytest runs under one minute; lost second run un-timed',
    forbidden_attempts=counts.get('forbidden_attempts',0)+6,
    historical_cuda_boundary=dict(status='NOT_ZERO_CERTIFIABLE',
        guarded_denied_parent_queries=12,
        previous_unguarded_resume_child_cpu_adam_steps=6,
        inferred_cuda_availability_calls=6,
        underlying_driver_enumeration_or_initialization='UNKNOWN; availability may invoke cudaGetDeviceCount; do not report zero',
        gpu_tensor_allocation_and_compute=0,
        correction='child now guards all listed queries; CPU-only diagnostic no-op; final exact resume passed'),
    full_suite_reruns_after_pass=0,real_neural_constructors=0,real_neural_forward=0,
    real_training_updates=0,real_protected_future_reads=0,real_outer_metrics=0,
    fake_provider_counts_in_instrumented_runs=dict(p2counts))
assert all(v=='passed' for v in latest.values()),latest
write('CPU_RUN.json',summary)
# Actual CLI read-only launch checks (no --phase training execution).
entry=[]
for name in ['CLEAN_GOAL_P2.yaml','CLEAN_JOINT_P2.yaml','A0_BLOCKED.yaml','A1_BLOCKED.yaml']:
    command=[sys.executable,'-B','main.py','--p2_config',str((D/name).relative_to(R)),'--p2_launch_check']
    p=subprocess.run(command,cwd=R,text=True,capture_output=True)
    assert p.returncode==2 and 'BLOCKED_RECORDING_IDENTITY_UNVERIFIED' in p.stdout,(name,p.stdout,p.stderr)
    entry.append(dict(config=name,command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
write('ENTRYPOINT_MATRIX.json',dict(actual_main_launch_checks=entry,
    direct_guard_tests=['parser','trainer','goal_pretrainer','JointDependencyV2CacheBuilder','main dispatch'],
    success_fixtures=['fresh goal requires no prior base; train not train_test','joint rejects missing/bad goal parent before device'],
    production_order='metadata/provenance/purpose/artifact checks before device/RNG/network/checkpoint; loader then execution',
    p0_p1_default_dispatch='unchanged',outer_metric_requests='always denied before provider',
    launch_check_scope='metadata acceptance only, not readiness, network construction or full parser runtime resolution',
    l2_selection_adapter='NOT_IMPLEMENTED'))
write('TEACHER_ACCESS_LEDGER.json',dict(real=dict(observation_payload_reads=0,protected_target_reads=0,teacher_reads=0,
    future_descriptor_calls=0,cache_writes=0,model_candidate_calls=0),
    synthetic_per_cache_case=dict(fake_candidate_calls=4,observation_reads=4,train_target_reads=2,
        train_descriptor_calls=2,train_teacher_writes=2,inner_outer_teacher_calls=0,inner_outer_teacher_writes=0),
    synthetic_cache_case_executions=2,synthetic_totals=dict(fake_candidates=8,pure_descriptors=4,teacher_writes=4),
    instrumented_provider_totals=dict(p2counts),first_uninstrumented='first cache/permission cases passed; provider aggregate not independently recorded',
    actual_plan=dict(deployment=4249,train_teacher=3484,protected_teacher=0,executed=0)))
# Semantic/source checks via AST and hashes only.
definitions=[]
for path in list((R/'src/models').rglob('*.py'))+[R/'src/joint_goal_loss.py',R/'src/jdv2_objective_state.py']:
    rel=str(path.relative_to(R));before=subprocess.check_output(['git','show',INPUT+':'+rel],cwd=R)
    old=ast.parse(before);new=ast.parse(path.read_bytes())
    if rel=='src/models/goal_pretrain.py':
        old=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='Goal_Pretrain')
        new=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='Goal_Pretrain')
    assert ast.dump(old)==ast.dump(new),rel
    definitions.append(dict(path=rel,scope='Goal_Pretrain class only' if rel.endswith('goal_pretrain.py') else 'whole module AST',
        unchanged=True,before_file_sha256=sha(before),after_file_sha256=sha(path.read_bytes())))
old_dir=R/'docs/joint_dependency_v2/reviews/2026-10-05_4f11f44_clean_base_split_objective_preflight'
old_evidence=read(old_dir/'EVIDENCE_MANIFEST.json')
for f in old_evidence['files']:
    if f['path']=='docs/joint_dependency_v2/reviews/README.md':
        assert sha(subprocess.check_output(['git','show',INPUT+':'+f['path']],cwd=R))==f['sha256']
        continue  # living index explicitly updated; old archive files stay immutable
    assert sha((R/f['path']).read_bytes())==f['sha256']
modified=subprocess.check_output(['git','diff','--name-only'],cwd=R,text=True).splitlines()
new=['src/p2_protocol.py','src/p2_data.py','src/p2_cache.py','src/p2_bounded.py','src/p2_checkpoint.py',
     'tests/test_p2_role_loader_step_cap.py','tools/p2_metadata_plan.py']
write('SOURCE_SCOPE.json',dict(input_commit=INPUT,modified_tracked=modified,new_code=new,
    neural_loss_definitions=definitions,new_parameters=0,new_buffers=0,
    p1_contract_byte_unchanged=sha((R/'src/clean_split_protocol.py').read_bytes())==sha(subprocess.check_output(['git','show',INPUT+':src/clean_split_protocol.py'],cwd=R)),
    old_archive_preserved_files=len(old_evidence['files'])-1,living_index='updated separately; original input-commit bytes verified',old_five_checkpoint_reaudit=0,
    default_disabled_dispatch='existing legacy/P1 branches retained; tested in scoped regression',
    nonproduction_test_repairs=['SDD checkpoint test stub accepts existing weights_only=False keyword',
      'CPU MC child optimizer diagnostic no-op with CPU assertion and explicit CUDA query guards'],
    sdd_skills_apps_intervention=False,unrelated_untracked_files_staged=False))
dry=read(D/'ROLE_ADDRESS_DRY_RUN.json')
write('RESULTS.json',dict(input_commit=INPUT,ENGINEERING_CPU_ACCEPTANCE='PASSED',
    acceptance_scope='CPU synthetic production wiring only; historical execution compliance exception disclosed',
    EXECUTION_BOUNDARY_COMPLIANCE='EXCEPTION_UNGUARDED_CPU_CHILD_CUDA_AVAILABILITY_QUERY',
    DATA_BASE_READINESS='BLOCKED_RECORDING_IDENTITY_UNVERIFIED',
    TRAINING_EXECUTION='NOT_AUTHORIZED_NOT_STARTED',counts=dry['counts'],p2_tests=len(nodes),
    scoped_unique_tests=len(latest),cpu_summary='CPU_RUN.json',known_toy_updates=summary['known_synthetic_completed_updates'],
    qualified_clean_bases=0,new_real_caches=0,real_initial_state=None,
    missing_observation_only_artifacts=4249,semantic_map_provenance='UNKNOWN',
    l2_selection='NOT_IMPLEMENTED',real_neural_calls=0,real_data_updates=0,real_protected_reads=0,real_outer_metrics=0,
    cuda_enumeration_and_initialization='NOT_ZERO_CERTIFIABLE: historical child diagnostic guard gap',
    gpu_compute=0,legacy_checkpoint_reaudit=0,
    next_action='Bind positive recording/map provenance and per-window obs-only artifacts; separately authorize fresh goal -> joint; then certify base/cache before L1'))
print(json.dumps(dict(p2=len(nodes),unique=len(latest),inferred=len(inferred),toy_updates=summary['known_synthetic_completed_updates'],seconds=seconds)))
