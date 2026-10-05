#!/usr/bin/env python3
"""CPU regression wrapper; all new records go to the new review only."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p2_protocol import ROOT,file_hash
from src.p2_observation import atomic_json
from src.p2_grouped_training import source_fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();D=a.archive.resolve();O=a.output.resolve();stamp=str(time.time_ns());rows=[]
env=dict(os.environ,RSJG_TEST_ARCHIVE=str(D))
commands=[([sys.executable,'-B','-m','pytest','-p','tools.grouped_test_instrumentation','-q',
    'tests/test_p2_grouped_preflight.py','tests/test_p2_role_loader_step_cap.py','tests/test_official_prior.py'],env),
    ([sys.executable,'-B',str(ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_bda1ae1_p2_observation_provenance_cpu/run_tests.py'),
    str(D/('GUARDED_OBSERVATION_'+stamp+'.json')),'tests/test_p2_observation_artifacts.py'],dict(env,CUDA_VISIBLE_DEVICES=''))]
for i,(cmd,environment) in enumerate(commands):
    start=time.monotonic();log=O/('cpu_'+stamp+'_'+str(i)+'.log')
    with log.open('w') as f:p=subprocess.run(cmd,cwd=ROOT,env=environment,stdout=f,stderr=subprocess.STDOUT)
    rows.append(dict(argv=cmd,exit=p.returncode,seconds=time.monotonic()-start,log=str(log),sha256=file_hash(log),summary=log.read_text()[-14000:]))
result=dict(status='PASS' if all(r['exit']==0 for r in rows) else 'FAILED',runs=rows,
    source=source_fingerprint())
atomic_json(result,D/('CPU_TEST_ATTEMPT_'+stamp+'.json'));atomic_json(result,D/'CPU_TEST_RESULT.json')
print(json.dumps(result));raise SystemExit(0 if result['status']=='PASS' else 1)
