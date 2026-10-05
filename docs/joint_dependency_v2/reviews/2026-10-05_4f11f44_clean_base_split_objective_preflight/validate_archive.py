"""Final archive-only validation; no tensors, model imports or network."""
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import yaml
D=Path(__file__).resolve().parent
R=D.parents[3]
SOURCE="4f11f44c3a490c491eb310b4fa7529806c71c1ed"
def sha(b):return hashlib.sha256(b).hexdigest()
def read(name):return json.loads((D/name).read_text())
def stable(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
if '--hashes' in sys.argv:
    paths=sorted([p for p in D.rglob('*') if p.is_file() and p.name!='EVIDENCE_MANIFEST.json']+
        [R/'tools/jdv2_clean_objective_preflight.py',R/'tests/test_jdv2_clean_objective_preflight.py',D.parent/'README.md'])
    print(json.dumps(dict(source_commit=SOURCE,self_excluded=True,files=[
        dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in paths]),indent=2))
    raise SystemExit
json_count=0
for p in D.rglob('*.json'):
    json.loads(p.read_text());json_count+=1
for p in [R/'tools/jdv2_clean_objective_preflight.py',R/'tests/test_jdv2_clean_objective_preflight.py',*D.glob('*.py')]:
    ast.parse(p.read_text())
m=read('SPLIT_ROLE_MANIFEST.json');rows=[]
for part in m['record_files']:
    raw=(D/part['path']).read_bytes()
    assert sha(raw)==part['sha256']
    rows.extend(json.loads(line) for line in raw.splitlines())
assert len(rows)==4249 and Counter(r['logical_role'] for r in rows)==m['counts']
order=read('BATCH_ORDER_SEED3101.json')
assert len(order['records'])==3484 and stable(order['records'])==order['order_sha256']
for arm in ('A0','A1'):
    assert yaml.safe_load((D/(arm+'_BLOCKED.yaml')).read_text())==read('RESOLVED_CONFIGS.json')['configs'][arm]
with (D/'SOURCE_EXPOSURE_MATRIX.csv').open() as f:
    exposure=list(csv.DictReader(f))
assert len(exposure)==8
assert Counter(x['p2_role'] for x in exposure)=={'train':6,'outer':1,'inner_valid':1}
missing=[]
for p in D.glob('*.md'):
    for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        target=link.split('#')[0]
        if target and '://' not in target and not (p.parent/target).exists() and target!='EVIDENCE_MANIFEST.json':
            missing.append((p.name,link))
assert not missing,missing
tracked=subprocess.check_output(['git','diff','--name-only',SOURCE],cwd=R,text=True).splitlines()
assert tracked==['docs/joint_dependency_v2/reviews/README.md'],tracked
production=[]
for name in subprocess.check_output(['git','ls-tree','-r','--name-only',SOURCE,'src','configs','main.py'],cwd=R,text=True).splitlines():
    b=(R/name).read_bytes()
    baseline=subprocess.check_output(['git','show',SOURCE+':'+name],cwd=R)
    assert b==baseline,name
    production.append(dict(path=name,bytes=len(b),sha256=sha(b)))
out=dict(status='PASS',source_commit=SOURCE,json_files_parsed=json_count,windows=4249,
    shard_sha_matches=True,order_sha_matches=True,yaml_matches_static_json=True,
    csv_sources=8,local_document_links_valid=True,protected_tensor_reads=0,
    real_models=0,cuda=0,checkpoint_loads=0,production_code_changes=0,
    tracked_changes_before_staging=tracked,production_files_unchanged=production,
    note='Archive metadata and source byte validation only; no test or checkpoint audit rerun')
print(json.dumps(out,indent=2))
