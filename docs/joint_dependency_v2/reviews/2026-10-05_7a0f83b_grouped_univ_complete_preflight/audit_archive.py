"""Final metadata/catalog/source consistency. Large tensors verified separately."""
import json,sys
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from src.p2_protocol import Registry,file_hash,path,data_binding,sealed
from src.p2_observation import atomic_json
D=Path(__file__).resolve().parent
def read(name):return json.loads((D/name).read_text())
m=read('QUALIFIED_INPUT_REVIEW_MANIFEST.json');reg=Registry(m,m['manifest_hash'])
binding=data_binding(reg);counts=Counter();seen=set();rewrapped=0;reuse=0
for shard in read('ARTIFACT_VERIFY_RESULT.json')['catalog_shards']:
    p=path(shard['path']);assert file_hash(p)==shard['sha256']
    values=[json.loads(x) for x in p.read_text().splitlines()];assert len(values)==shard['count']
    for v in values:
        key=(v['window_id'],v['kind']);assert key not in seen;seen.add(key)
        r=reg.by_id[v['window_id']];assert r['role']==v['role']
        assert r['artifacts'][v['kind']]==v['artifact']
        counts[v['role']+'/'+v['kind']]+=1
        if 'sidecar' in v:
            assert file_hash(path(v['sidecar']['path']))==v['sidecar']['sha256']
            side=json.loads(path(v['sidecar']['path']).read_text())
            assert side==v['metadata'] and sealed(side)==side
            if v['kind']=='observation':rewrapped+=1
        else:
            assert v['kind']=='observation' and v['content_sha256']==v['reuse_chain']['old_content_sha256'];reuse+=1
assert rewrapped==947 and reuse==3302
assert counts=={'train/observation':2537,'inner_valid/observation':1267,'outer/observation':445,
    'train/target':2537,'inner_valid/target':1267}
checked_sources={}
for mode in ('cpu','cuda','reload'):
    v=read('MODEL_SMOKE_'+mode+'.json')
    assert v['status']=='PASS' and v['smoke_purpose_authorized_before_provider'] is True
    assert v['data_binding']==binding
    for name,sha in v['source']['files'].items():
        assert file_hash(R/name)==sha,(mode,name)
        checked_sources[name]=sha
fresh=read('FRESH_INITIAL.json');assert fresh['updates']==0 and fresh['data_binding']==binding
assert fresh['state_sha256']==read('FRESH_INITIAL_PRIOR_BINDING.json')['state_sha256']
assert fresh['formal_config_sha256']==file_hash(D/'FRESH_GOAL_BLOCKED.yaml')
assert file_hash(path(fresh['path']))==fresh['sha256']
assert read('CPU_TEST_RESULT.json')['status']=='PASS'
assert read('FULL_NATIVE_PACK_ARTIFACT_AUDIT.json')['status']=='ARTIFACT_VALIDATION_ONLY_NOT_PRODUCTION'
assert read('READINESS_MATRIX.json')['status']=='EXTERNAL_EVIDENCE_BLOCKED'
all_json=list(D.rglob('*.json'))
for p in all_json:json.loads(p.read_text())
for p in D.rglob('*'):
    if p.is_file():assert p.suffix in {'.py','.json','.jsonl','.md','.yaml'},str(p)
atomic_json(dict(status='PASS',catalog_items=len(seen),counts=dict(counts),observation_rewrapped=rewrapped,
    observation_reused= reuse,data_binding=binding,parsed_json_files=len(all_json),checked_final_sources=checked_sources,
    tensor_bytes='checked by full artifact verify; not redundantly deserialized here',
    fresh_config_and_file_sha_verified=True,fresh_seed_reinitialization_state_equal=True,
    final_smoke_source_and_purpose_verified=True),D/'ARCHIVE_CONSISTENCY_AUDIT.json')
print(json.dumps({'status':'PASS','catalog_items':len(seen),'data_binding':binding}))
