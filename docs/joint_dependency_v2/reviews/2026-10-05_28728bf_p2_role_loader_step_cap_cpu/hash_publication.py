"""Hash only the explicitly authorized publication scope; never scan outputs/data."""
import ast,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[3]
changed=subprocess.check_output(['git','diff','--name-only'],cwd=R,text=True).splitlines()
new=['src/p2_protocol.py','src/p2_data.py','src/p2_cache.py','src/p2_bounded.py','src/p2_checkpoint.py','tests/test_p2_role_loader_step_cap.py','tools/p2_metadata_plan.py']
paths=set(changed+new+[str(p.relative_to(R)) for p in D.rglob('*') if p.is_file()])
self_path=str((D/'EVIDENCE_MANIFEST.json').relative_to(R))
rows=[]
for rel in sorted(paths-{self_path}):
    p=R/rel
    assert not rel.startswith(('outputs/','data/')),rel
    assert p.suffix in {'.py','.md','.json','.jsonl','.yaml'},rel
    b=p.read_bytes();s=b.decode()
    assert not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|ghp_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9]{25,}',s),rel
    if p.suffix=='.py':ast.parse(s)
    if p.suffix=='.json':json.loads(s)
    if p.suffix=='.jsonl':
        for line in s.splitlines():json.loads(line)
    rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
manifest=dict(input_commit='28728bf96a290ccd83c5c637769deb987f477b07',
    self_excluded=True,scope='explicit source/tests/tools/review/index; no real checkpoints/payloads',
    archive_authorization='user prompt sections 1,9,10: ordinary commit/push all this-round files; fixed-SHA byte readback',
    text_only=True,credential_pattern_check='passed',files=rows)
(D/'EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(file_count_with_manifest=len(rows)+1,max_bytes=max(r['bytes'] for r in rows),total_bytes=sum(r['bytes'] for r in rows))))
