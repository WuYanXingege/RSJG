"""Actually invoke production and conditional entrypoints; preserve their exits."""
import json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from src.p2_protocol import file_hash
from src.p2_observation import atomic_json
D=Path(__file__).resolve().parent;mp=D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json'
m=json.loads(mp.read_text());rows=[]
def run(name,args,expected,needle):
    cmd=[sys.executable,'-B',*args];start=time.monotonic()
    p=subprocess.run(cmd,cwd=R,text=True,capture_output=True)
    row=dict(name=name,argv=cmd,exit=p.returncode,expected_exit=expected,seconds=time.monotonic()-start,
        stdout=p.stdout,stderr=p.stderr,entrypoint_sha256=file_hash(R/args[0]))
    row['expected_block']=p.returncode==expected and needle in (p.stdout+p.stderr);rows.append(row)
    if not row['expected_block']:
        atomic_json(dict(status='FAILED_ENTRY',entries=rows),D/'ENTRYPOINT_EXECUTION.json');raise RuntimeError(name)
    return row
run('qualified_inner_selection',['tools/grouped_inner_selection.py','--archive',str(D)],2,'')
assert json.loads((D/'INNER_SELECTION_RESULT.json').read_text())['status']=='EXTERNAL_EVIDENCE_BLOCKED'
r=run('production_full_loader',['tools/grouped_pack_audit.py','--archive',str(D),'--production'],1,'EXTERNAL_EVIDENCE_BLOCKED')
atomic_json(dict(status='EXTERNAL_EVIDENCE_BLOCKED',argv=r['argv'],exit=r['exit'],
    stderr=r['stderr'],file_reads_by_production_provider=0,scope='genuine production gate rejected before PackReader provider access'),D/'PRODUCTION_LOADER_RESULT.json')
run('combined_preflight',['tools/grouped_preflight.py','check','--archive',str(D)],2,'EXTERNAL_EVIDENCE_BLOCKED')
for stage in ('joint','cache','shared','l1'):
    run('pipeline_'+stage,['tools/grouped_pipeline.py','check','--stage',stage,'--manifest',str(mp),
        '--manifest-hash',m['manifest_hash']],2,'WAITING')
atomic_json(dict(status='EXPECTED_GATES_CONFIRMED_NOT_READY',entries=rows),D/'ENTRYPOINT_EXECUTION.json')
print(json.dumps({'status':'EXPECTED_GATES_CONFIRMED_NOT_READY','entries':len(rows)}))
