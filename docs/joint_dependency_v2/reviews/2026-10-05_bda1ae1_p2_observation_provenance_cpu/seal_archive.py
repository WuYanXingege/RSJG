"""Text-only archive seal; no Torch, source trajectory or payload reads."""
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[3]
extra=['src/p2_observation.py','src/p2_observation_guard.py',
    'tools/p2_observation_artifacts_cpu.py','tests/test_p2_observation_artifacts.py',
    'docs/joint_dependency_v2/reviews/README.md']
target=D/'EVIDENCE_MANIFEST.json'
paths=[p for p in D.rglob('*') if p.is_file() and p!=target]+[R/p for p in extra]
rows=[]
for p in sorted(paths):
    assert p.suffix in {'.json','.jsonl','.md','.py'} and not p.is_symlink(),p
    b=p.read_bytes();text=b.decode('utf-8');assert '\0' not in text
    if p.suffix=='.json':json.loads(text)
    if p.suffix=='.jsonl':
        for line in text.splitlines():json.loads(line)
    rows.append(dict(path=str(p.relative_to(R)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
result=dict(schema='rsjg-p2-observation-evidence-v1',
    fixed_start_commit='bda1ae147d5912e8847642db36b5822584420563',
    files=rows,files_count=len(rows),bytes=sum(r['bytes'] for r in rows),
    excluded=['This self-referential manifest','local tensors/static/raw inputs','post-commit remote receipt'],
    binary_payloads_archived=0,scope='Code/tests/text/metadata only; remote verification uses full final commit for all changed files including this manifest')
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(files=len(rows),bytes=result['bytes'],largest=max(r['bytes'] for r in rows))))
