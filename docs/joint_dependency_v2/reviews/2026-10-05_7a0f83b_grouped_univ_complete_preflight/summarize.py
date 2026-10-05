"""Summarize actual receipts only; never infer execution from planned tests."""
import ast,datetime,hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from src.p2_protocol import file_hash
from src.p2_observation import atomic_json
D=Path(__file__).resolve().parent;O=R/'outputs/joint_dependency_v2/grouped_univ_preflight_7a0f83b_20261005'
def read(name,default=None):
    p=D/name
    return json.loads(p.read_text()) if p.exists() else default
def lines(name):
    p=D/name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
ledger=lines('PROCESS_LEDGER.jsonl');optim=lines('OPTIMIZER_LEDGER.jsonl');gpu=lines('GPU_BUDGET_LEDGER.jsonl')
samples=[x for x in ledger if x['event']=='SAMPLE'];ends=[x for x in ledger if x['event']=='END']
starts={x['job']:x for x in ledger if x['event']=='START'};by_job={}
for x in samples:by_job.setdefault(x['job'],[]).append(x['time'])
gaps=[b-a for ts in by_job.values() for a,b in zip(ts,ts[1:])]
coverage=dict(nominal_sampling_seconds=1,actual_max_sample_gap_seconds=max(gaps,default=0),
    open_bookkeeping_jobs_at_snapshot=sorted(set(starts)-{x['job'] for x in ends}),
    snapshot_scope='all completed acceptance jobs; current summarizer and later publication housekeeping are outside this end-of-job snapshot',
    aggregate_sampled_peak_rss=max((x['aggregate_rss'] for x in samples),default=0),
    process_high_water_by_namespace_pid_start={k:v for x in ends for k,v in x['process_high_water'].items()},
    jobs=ends,failed_jobs=[x for x in ends if x['exit_code'] or x['stop']],
    gaps=['First bootstrap/read-only discovery commands were not process-tree instrumented; no retroactive zero claim.',
        'One-second nominal sampling can miss aggregate transient peaks. Per-process HWM is distinct from aggregate RSS.',
        'OS thread count includes inactive runtime pools; numerical compute configured <=4 per sequential job.',
        'Initial export ran an earlier CLI orchestration version; stable numerical producer SHA is sidecar-bound. Final verify uses current code.',
        'Early supervised commands predate entrypoint hash instrumentation; command and numerical producer identity remain archived.',
        'No certification of import-before-hook/native CUDA event completeness; no historical RSS gap repaired.'])
atomic_json(coverage,D/'RESOURCE_SUMMARY.json')
readiness=read('READINESS_MATRIX.json',{'status':'NOT_COMPLETED'})
binding=read('AUTHOR_RELEASE_BINDING.json',{})
artifacts={n:read(n) for n in ['ARTIFACT_EXPORT_RESULT.json','ARTIFACT_VERIFY_RESULT.json','FULL_NATIVE_PACK_ARTIFACT_AUDIT.json']}
smokes={m:read('MODEL_SMOKE_'+m+'.json') for m in ('cpu','cuda','reload')}
smoke_history=[json.loads(p.read_text()) for p in sorted((D/'model_smoke_attempts').glob('*.json'))]
names=['src/models/goal_pretrain.py','src/models/model.py','src/losses.py',
    'src/jdv2_objective_state.py','src/models/social_encoder.py',
    'src/models/joint_dependency_v2/dynamic_relation.py',
    'src/models/joint_dependency_v2/joint_energy.py','src/models/joint_dependency_v2/joint_sampler.py']
network={}
for n in names:
    p=R/n
    if not p.exists():continue
    old=subprocess.run(['git','show','7a0f83bb9341617b4f214cefc3563927961c08d6:'+n],cwd=R,capture_output=True)
    network[n]={'current_sha256':file_hash(p),'starting_sha256':hashlib.sha256(old.stdout).hexdigest() if old.returncode==0 else None,
        'bytes_unchanged':old.returncode==0 and p.read_bytes()==old.stdout}
atomic_json({'files':network,'allowed_model_delta':'optional dataset=None constructor injection only; exact AST regression in CPU suite',
    'loss_or_Stage_A_objective_change':False,'prior_evidence_reused':'cb5160f/dcf5fde/c9dbfec/4f11f44; no new full diffusion bank/Stage-A training',
    'architecture_receipts':{k:[{'case':x['case'],'architecture':x.get('architecture')} for x in (v or {}).get('results',[]) if x.get('architecture')] for k,v in smokes.items()}},D/'NETWORK_AND_LOSS_AUDIT.json')
attempts=[x for x in optim if x['event']=='ATTEMPT'];success=[x for x in optim if x['event']=='SUCCESS']
result=dict(status=readiness['status'],generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    start_head='7a0f83bb9341617b4f214cefc3563927961c08d6',source_worktree='executed source hashes recorded; final publication commit supplied externally to avoid self-reference',
    formal_long_training_started=False,family='p2_grouped_univ_hotel_v1',counts={'train':2537,'inner_valid':1267,'outer':445},
    expected_target_counts={'train':2537,'inner_valid':1267,'outer':0},artifact_evidence=artifacts,
    author_archive={'status':binding.get('status'),'sha256':binding.get('archive_sha256'),'bytes':binding.get('archive_bytes'),
        'local_member_references':len(binding.get('members',[])),
        'unique_archive_members':len({x['member'] for x in binding.get('members',[])}),
        'matching_references':sum(x['match'] for x in binding.get('members',[]))},
    semantic_status='UNKNOWN_LEARNED_RASTER_TRAINING_SCOPE',readiness=readiness,
    cpu_tests=read('CPU_TEST_RESULT.json'),smokes=smokes,fresh_initial=read('FRESH_INITIAL.json'),
    model_call_accounting=dict(scope='derived from completed harness cases; internal submodule/native forward counts not instrumented',
        completed_smoke_model_constructions=sum(len(v.get('results',[])) for v in smoke_history),
        fresh_unupdated_model_constructions=1+int((D/'FRESH_INITIAL_PRIOR_BINDING.json').exists()),
        real_backward_calls=len(attempts),
        real_loss_calls=len(attempts)+sum(len(v.get('results',[])) for v in smoke_history),
        top_level_observation_only_inferences=sum(bool(r.get('inference')) for v in smoke_history for r in v.get('results',[]))),
    real_optimizer={'attempts':len(attempts),'successful':len(success),'unsuccessful_attempts':len(attempts)-len(success),
        'skips':0,'phase_failures':[x for x in optim if x['event']=='FAILURE'],
        'agent_exposure_attempts':sum(len(x['members']) for x in attempts),'roles':['train'] if attempts else [],'cap':24},
    protected_uses={'outer_targets':0,'outer_metrics':0,'inner_outer_gradients':0,'teacher':0,'production_inner_prediction_or_selection':0},
    gpu={'conservatively_charged_seconds':sum(x['charged_seconds'] for x in gpu),'budget_seconds':1800,
        'peak_reserved_bytes':max((x.get('peak_reserved',0) for v in smoke_history for x in v.get('results',[])),default=0)},
    measured_owned_job_wall=sum(x['seconds'] for x in ends),cpu_data_wall_limit_seconds=21600,
    new_local_output_bytes=sum(p.stat().st_size for p in O.rglob('*') if p.is_file()),
    conditional_pipeline=read('PIPELINE_CONDITIONS.json'),resource_coverage='RESOURCE_SUMMARY.json',
    historical_boundaries=['five old disqualified weights remain rejected','historical first-difference causal chain OPEN',
        'seed2036 not bitwise resumed','old paired deltas preserved; new protocol is not same experiment',
        'smoke proves neither convergence/performance nor complete CUDA restart determinism'])
atomic_json(result,D/'RESULTS.json')
print(json.dumps({'status':result['status'],'attempts':len(attempts),'successful':len(success),
    'gpu_seconds':result['gpu']['conservatively_charged_seconds'],'new_bytes':result['new_local_output_bytes']}))
