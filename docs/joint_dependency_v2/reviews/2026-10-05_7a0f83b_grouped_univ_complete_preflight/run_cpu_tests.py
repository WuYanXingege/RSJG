"""Measured pytest invocation; preserve all failed attempts in process ledger."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from src.p2_observation import atomic_json
from src.p2_protocol import file_hash
D=Path(__file__).resolve().parent;O=R/'outputs/joint_dependency_v2/grouped_univ_preflight_7a0f83b_20261005'
stamp=time.time_ns();log=O/('pytest_'+str(stamp)+'.log')
cmd=[sys.executable,'-B','-m','pytest','-p','tools.grouped_test_instrumentation','-q','tests/test_p2_grouped_preflight.py','tests/test_p2_role_loader_step_cap.py']
start=time.monotonic()
with log.open('w') as f:p=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
guard_file=D/('OBSERVATION_GUARDED_TESTS_'+str(stamp)+'.json')
guard_cmd=[sys.executable,'-B',str(R/'docs/joint_dependency_v2/reviews/2026-10-05_bda1ae1_p2_observation_provenance_cpu/run_tests.py'),str(guard_file),'tests/test_p2_observation_artifacts.py']
guard=subprocess.run(guard_cmd,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True)
code=p.returncode or guard.returncode
result=dict(status='PASS' if code==0 else 'FAILED',argv=cmd,exit=code,seconds=time.monotonic()-start,log=str(log),log_sha256=file_hash(log),summary=log.read_text()[-10000:],
    observation_guarded=dict(argv=guard_cmd,exit=guard.returncode,result_path=str(guard_file),result_sha256=file_hash(guard_file) if guard_file.exists() else None,stdout=guard.stdout,stderr=guard.stderr))
atomic_json(result,D/('CPU_TEST_ATTEMPT_'+str(stamp)+'.json'));atomic_json(result,D/'CPU_TEST_RESULT.json')
print(json.dumps(result));raise SystemExit(code)
