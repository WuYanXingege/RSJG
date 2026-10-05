"""Guarded metadata/readiness aggregation after exhaustive observation readback."""
import contextlib,copy,hashlib,io,json,os,resource,subprocess,sys,time
from pathlib import Path
from collections import Counter,defaultdict
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from src.p2_observation_guard import install
_,guard=install()
from src.p2_protocol import Registry,digest,file_hash
from src.p2_observation import atomic_json,producer_hash
from src.parser import main_parser
D=Path(__file__).resolve().parent
P=R/'docs/joint_dependency_v2/reviews/2026-10-05_28728bf_p2_role_loader_step_cap_cpu'
def failure_hook(kind,value,tb):
    import traceback
    atomic_json(dict(status='FAILED',traceback=''.join(traceback.format_exception(kind,value,tb)),
        guard=guard()),D/('FINALIZATION_FAILURE_'+str(time.time_ns())+'.json'))
    sys.__excepthook__(kind,value,tb)
sys.excepthook=failure_hook
def read(name):return json.loads((D/name).read_text())
def write(name,value):atomic_json(value,D/name)
old_m=json.loads((P/'P2_RUNTIME_MANIFEST.json').read_text())
old=Registry(old_m,old_m['manifest_hash'])
m=read('P2_RUNTIME_MANIFEST.json');reg=Registry(m,m['manifest_hash'])
assert len(reg.records)==4249 and reg.counts==old.counts
for a,b in zip(old.records,reg.records):
    expected=copy.deepcopy(a);expected['artifacts']['observation']=b['artifacts']['observation']
    assert b==expected
assert m['train_order']==old.manifest['train_order']
for k in ('parents','shared_initial_state','cache'):
    assert m.get(k)==old.manifest.get(k)
catalog=read('OBSERVATION_ARTIFACT_CATALOG.json');rows=[]
for s in catalog['record_files']:
    assert file_hash(R/s['path'])==s['sha256']
    rows+=json.loads((R/s['path']).read_text())
assert {r['window_id'] for r in rows}==set(reg.by_id) and len(rows)==4249
by_source=defaultdict(list);by_role=defaultdict(list)
for row in rows:
    by_source[Path(reg.by_id[row['window_id']]['source']['original_source_path']).name].append(row)
    by_role[row['role']].append(row)
def aggregate(values):
    return dict(expected=len(values),completed=len(values),missing=0,failed=0,hash_mismatch=0,
        numerical_unknown=0,provenance_unknown=len(values),agent_window_exposures=sum(r['agent_count'] for r in values),
        window_local_obs_slots=sum(8*r['agent_count'] for r in values),
        bytes=sum(r['bytes']+r['sidecar_bytes'] for r in values),
        max_artifact_bytes=max(r['bytes'] for r in values),
        agent_count_range=[min(r['agent_count'] for r in values),max(r['agent_count'] for r in values)],
        map_hw=sorted({tuple(r['shape'][-2:]) for r in values}),
        out_of_raster_obs_slots=sum(r['out_of_raster_obs_slots'] for r in values),
        boundary_interventions=0,recording_verified=0,map_protocol_permitted=0,
        per_group_runtime_seconds='NOT_INSTRUMENTED; measured at process/phase scope in RUN files',
        per_group_peak_rss_bytes='NOT_INSTRUMENTED; bounded by measured data-process high-water mark',
        real_future_queries=0,real_targets=0,real_teachers=0)
runs=[(p.name,json.loads(p.read_text())) for p in sorted(D.glob('RUN_*.json'))]
assert all(r['status'] in ('PLANNED','PASSED') for _,r in runs)
assert any(r['phase']=='verify' and r['completed']==4249 for _,r in runs)
counters=Counter()
for _,r in runs:counters.update(r['counters'])
assert counters['observation_tensor_windows']==4249 and counters['window_local_observation_slots']==8*37274
canary_ids=set(read('CANARY_PLAN.json')['window_ids'])
source_read_accounting={}
for name,group in by_source.items():
    records=[reg.by_id[x['window_id']] for x in group]
    first=records[0];size=(R/first['source']['original_source_path']).stat().st_size
    subsets=[[r for r in records if (r['window_id'] in canary_ids)==choose] for choose in (True,False)]
    unique=sum(len({(f,a) for r in subset for f in r['source']['frame_ids'][:8] for a in r['source']['agent_ids']}) for subset in subsets)
    passes=1+len(first['aliases'])+4
    source_read_accounting[name]=dict(raw_hash_passes=passes,raw_hash_bytes=passes*size,
        raw_scan_bytes=2*size,unique_raw_obs_xy_decodes_summed_per_scan=unique,
        window_local_observation_slots=sum(8*len(r['source']['agent_ids']) for r in records),
        scope='Per-source decomposition reconstructed from exact authorized routes and completed two scans; global measured counters reconciled, not a separately instrumented timer')
for key in ('raw_hash_passes','raw_hash_bytes','raw_scan_bytes','window_local_observation_slots'):
    assert sum(x[key] for x in source_read_accounting.values())==counters[key],key
assert sum(x['unique_raw_obs_xy_decodes_summed_per_scan'] for x in source_read_accounting.values())==counters['unique_raw_obs_xy_decodes']
write('PER_SOURCE_READ_ACCOUNTING.json',source_read_accounting)
checks=[]
for mode,config,extra in [
    ('goal','CLEAN_GOAL_P2.yaml',[]),('joint','CLEAN_JOINT_P2.yaml',[]),
    ('cache','A0_BLOCKED.yaml',['--p2_mode','cache','--phase','build-jdv2-cache']),
    ('l1_a0','A0_BLOCKED.yaml',[]),('l1_a1','A1_BLOCKED.yaml',[])]:
    argv=['main.py','--p2_config',str((P/config).relative_to(R)),
        '--p2_manifest',str((D/'P2_RUNTIME_MANIFEST.json').relative_to(R)),
        '--p2_manifest_hash',m['manifest_hash'],'--device','cpu','--p2_launch_check',*extra]
    stream=io.StringIO();sys.argv=argv
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        try:main_parser();code=None
        except SystemExit as exc:code=exc.code
    value=json.loads(stream.getvalue())
    assert code==2 and value['reason']=='BLOCKED_RECORDING_IDENTITY_UNVERIFIED'
    checks.append(dict(mode=mode,argv=argv,exit_code=code,result=value))
write('LAUNCH_CHECKS.json',dict(scope='Actual src.parser.main_parser --p2_launch_check branch; not main.py timed wrapper or constructors',
    checks=checks,guard=guard()))
tests=[];latest={}
for p in sorted(D.glob('CPU_TESTS_*.json')):
    value=json.loads(p.read_text());tests.append(dict(path=p.name,exit_code=value.get('exit_code'),
        outcomes=dict(Counter(r['outcome'] for r in value.get('tests',[]))),
        seconds=value.get('seconds'),guard=value.get('guard'),
        setup_failure=value.get('status') if not value.get('tests') else None))
    for r in value.get('tests',[]):
        if r['when']=='call':latest[r['nodeid']]=r['outcome']
assert latest and set(latest.values())=={'passed'}
guarded=[dict(path=p.name,guard=json.loads(p.read_text())['guard']) for p in sorted(D.glob('CPU_TESTS_*.json'))
    if 'guard' in json.loads(p.read_text())]
guarded += [dict(path=name,guard=r['guard']) for name,r in runs]
guarded += [dict(path='METADATA_PROPOSAL_PROCESS.json',guard=read('METADATA_PROPOSAL_PROCESS.json')),
    dict(path='LAUNCH_CHECKS.json',guard=guard())]
gc=Counter()
for item in guarded:gc.update(item['guard'].get('counts',{}))
assert sum(v for k,v in gc.items() if k in ('model_constructor_attempts','model_forward_attempts','backward_attempts','optimizer_constructor_attempts','optimizer_update_attempts'))==0
write('TEST_AND_PROCESS_LEDGER.json',dict(tests=tests,unique_latest_passed=len(latest),
    unique_test_last_outcomes=latest,tests_are_multiple_runs=True,guarded_process_records=guarded,
    guard_counts=dict(gc),forwarded_guarded_cuda_calls=sum(i['guard']['forwarded_cuda_calls'] for i in guarded),
    pre_import_and_unlisted_native_cuda='UNKNOWN',first_test_setup_guard_snapshot='NOT_RETAINED; zero test bodies executed',
    first_metadata_preparation_guard_snapshot='NOT_RETAINED: refreshed after evidence retry metadata; retained final process only; see COMMANDS_AND_FAILURE_HISTORY.md',
    pid_namespace='Sandbox processes can each report pid=2/ppid=1; PID alone is not a globally unique process identity',
    non_compute_commands='Shell/git/rg/sed/curl and stdlib-only archival scripts do not initialize tensors; not represented as instrumented Torch processes',
    worker_concurrency_max=2,compute_threads_each=1,compute_threads_concurrent_max=2,
    run_counters=dict(counters),synthetic_fixture_scope='Full20/target only synthetic; tests never construct a neural network or optimizer'))
protected=subprocess.check_output(['git','ls-files','src','main.py','tools','tests','configs'],cwd=R,text=True).splitlines()
unchanged=[]
for p in protected:
    data=subprocess.check_output(['git','show','bda1ae147d5912e8847642db36b5822584420563:'+p],cwd=R)
    assert (R/p).read_bytes()==data,p
    unchanged.append(dict(path=p,sha256=hashlib.sha256(data).hexdigest()))
new=['src/p2_observation.py','src/p2_observation_guard.py','tools/p2_observation_artifacts_cpu.py','tests/test_p2_observation_artifacts.py']
write('SOURCE_SCOPE.json',dict(start_head='bda1ae147d5912e8847642db36b5822584420563',
    actual_head_before_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),
    unchanged_production_test_config_files=unchanged,new_files=[dict(path=p,sha256=file_hash(R/p)) for p in new],
    model_parameter_delta=0,production_gate_changes=0,default_path_changes=0))
env=read('RESOURCE_PLAN.json')['environment']
from importlib.metadata import version
env=dict(env,torchvision=version('torchvision'),opencv=__import__('cv2').__version__,
    python_executable=sys.executable,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
    torch_compute_threads=1)
prompt=R.parent/'RSJG_P2_Observation_Provenance_CPU_Prompt_bda1ae1.md'
write('TASK_AUTHORIZATION.json',dict(prompt_path=str(prompt),prompt_sha256=file_hash(prompt),
    start_commit='bda1ae147d5912e8847642db36b5822584420563',
    authorized='CPU observation preparation, provenance evidence, synthetic tests, text/code archive commit/push',
    not_authorized=['real target/future/teacher','networks','optimizers','GPU','training','SDD/skills/apps changes'],
    cumulative_task_origin_utc='2026-10-05T07:46:57Z'))
write('ENVIRONMENT_AND_ARTIFACT_BINDING.json',dict(schema='rsjg-p2-observation-archive-binding-v1',
    environment=env,environment_sha256=digest(env),producer_hash=producer_hash(),
    numerical_producer_scope='src/p2_observation.py and three original pure-function modules, not entire execution code',
    execution_source_files=[dict(path=p,sha256=file_hash(R/p)) for p in new[:3]],
    catalog_sha256=file_hash(D/'OBSERVATION_ARTIFACT_CATALOG.json'),catalog_shards=catalog['record_files'],
    runtime_manifest_hash=m['manifest_hash'],runtime_manifest_file_sha256=file_hash(D/'P2_RUNTIME_MANIFEST.json'),
    run_records=[dict(path=name,sha256=file_hash(D/name)) for name,_ in runs],
    binding_scope='Each artifact and sidecar byte hash -> catalog shard hash -> this certificate with actual process/environment/code; no rewrite of historical sidecars',
    plan_producer_hash=read('RESOURCE_PLAN.json')['producer_hash'],
    plan_producer_difference='Plan predates synthetic serialization filename fix; no real canary/export used the failed producer'))
missing={k:dict(Counter(r['role'] for r in reg.records if not r['artifacts'].get(k))) for k in ('target','teacher','deployment')}
wall=max(r['budget']['wall_seconds'] for _,r in runs)
peak=max(r['budget']['peak_rss_bytes'] for _,r in runs)
results=dict(observation_materialization='COMPLETE',observation_payload_contract='PASSED',
    recording_provenance='UNKNOWN',release_binding='PARTIAL_AUTHOR_DISTRIBUTION_CHAIN_NO_LOCAL_MEMBER_BINDING',
    map_provenance='UNKNOWN',production_launch_readiness='BLOCKED_RECORDING_IDENTITY_UNVERIFIED',
    expected=4249,completed=4249,missing=[],failed=[],hash_mismatch=0,numerical_unknown=0,
    roles={k:aggregate(v) for k,v in by_role.items()},sources={k:aggregate(v) for k,v in sorted(by_source.items())},
    total=aggregate(rows),real_run_counters=dict(counters),unique_cpu_tests_latest_passed=len(latest),
    per_source_read_accounting_file='PER_SOURCE_READ_ACCOUNTING.json',
    real_target_artifacts_created=0,real_teacher_artifacts_created=0,qualified_clean_goal_count=0,qualified_clean_base_count=0,
    real_model_constructors=0,real_model_forward_count=0,real_model_backward_count=0,real_optimizer_updates=0,
    gpu_compute_calls=0,training_runs=0,full_source_pickle_deserializations=0,
    deliberate_cuda_query_attempts=gc.get('cuda_attempts',0)+gc.get('cuda_native_attempts',0),
    intercepted_cuda_queries=gc.get('cuda_intercepted',0)+gc.get('cuda_native_intercepted',0),
    forwarded_guarded_cuda_calls=0,uncovered_cuda_enumeration='UNKNOWN: before torch import / unlisted native entrypoints',
    first_test_setup_guard_snapshot='NOT_RETAINED; tests executed=0',
    first_metadata_preparation_guard_snapshot='NOT_RETAINED; only final metadata retry process snapshot retained',
    performance_benefit_tested=False,sdd_changed=False,skills_or_apps_changed=False,
    local_output=str(R/'outputs/joint_dependency_v2/p2_observation_bda1ae1_20261005'),
    runtime_manifest_hash=m['manifest_hash'],source_rows_hash=digest([r['source'] for r in reg.records]),
    train_order_hash=m['train_order_hash'],preparation_grant_hash=read('PREPARATION_GRANT.json')['manifest_hash'],
    artifact_bytes=catalog['total_artifact_bytes'],maximum_single_artifact_bytes=catalog['max_artifact_bytes'],
    canary_out_of_raster_slots=sum(r['out_of_raster_obs_slots'] for r in read('CANARY_ACCEPTANCE.json')['catalog']),
    canary_boundary_interventions=0,canary_rule_interpretation='No unauthorized temporal queries or modified raster boundary handling; not zero raster-exterior coordinates',
    out_of_raster_definition='native pixel /8 outside downsampled grid center range [0,w-1] x [0,h-1]; not equivalent to outside original continuous raster area',
    budget=dict(cumulative_wall_at_full_verify_seconds=wall,materialization_peak_rss_bytes=peak,
        conservative_concurrent_rss_estimate_bytes=peak+2*1024**3,concurrent_compute_threads_max=2,
        measured_all_process_combined_peak_rss='UNKNOWN: data phases measured; earlier test/metadata process peaks not retained; estimate is not measured aggregate',
        limit_wall_seconds=5400,limit_rss_bytes=8*1024**3,limit_artifact_bytes=32*1024**3,
        total_task_including_archival_wall='Final remote receipt timestamp; same 07:46:57 UTC origin, no restarted budget'),
    independent_missing_conditions=dict(recording_windows=4249,map_qualification_windows=4249,
        purpose_separated_targets=dict(train=3484,inner_valid=320,outer='NOT_AUTHORIZED_NOT_REQUIRED'),
        fresh_goal=1,fresh_base=1,new_deployment_cache_windows=4249,train_only_teachers=3484,shared_initial_state=1,
        l2_adapter='NOT_IMPLEMENTED',future_stage_authorization='NOT_GRANTED',raw_missing_artifact_refs=missing),
    real_launch_checks=checks,proposal_status='PROPOSAL_NOT_APPROVED',
    history_boundaries=['Old five checkpoints remain protocol-violating; not loaded or requalified',
        'Historical CUDA query coverage gap, historical forward/backward/label exposure retained',
        'Prior pairing effect, seed2036 non-bitwise resume and first-divergence causal limits unchanged',
        'Real full20 input parity NOT_RUN; new-objective benefit NOT_TESTED'],
    next_step='Request explicit approval for the separately specified UNIV grouping proposal; no training and no automatic protocol migration')
write('RESULTS.json',results)
write('FINALIZATION_PROCESS.json',dict(guard=guard(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    real_payload_reads=0,scope='Metadata, existing parser launch-check, hashes of source/archive only'))
print(json.dumps(dict(status=results['observation_materialization'],roles=reg.counts,tests=len(latest),
    counters=dict(counters),peak=peak,wall=wall,outside=results['total']['out_of_raster_obs_slots'])))
