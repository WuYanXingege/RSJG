"""Metadata migration and complete old text/archive read; no real tensor read."""
import copy,json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from src.p2_protocol import Registry,digest,file_hash,sealed
from src.p2_grouped import FAMILY,SCHEMA,START,MOVED_HASH,role_for,group_for,order,COUNTS
from src.p2_observation_guard import install
_,guard=install()
from src.p2_observation import atomic_json
from src.p2_data import base_packing_plan
D=Path(__file__).resolve().parent
P=R/'docs/joint_dependency_v2/reviews/2026-10-05_bda1ae1_p2_observation_provenance_cpu'
reads=[]
for p in sorted(P.rglob('*')):
    if p.is_file():
        text=p.read_text()
        if p.suffix=='.json':json.loads(text)
        if p.suffix=='.jsonl':
            for line in text.splitlines():json.loads(line)
        reads.append(dict(path=str(p.relative_to(R)),sha256=file_hash(p),bytes=p.stat().st_size))
e=json.loads((P/'EVIDENCE_MANIFEST.json').read_text())
for f in e['files']:
    # Live source files are intentionally about to change; immutable old archive must not.
    if f['path'].startswith(str(P.relative_to(R))+'/'):assert file_hash(R/f['path'])==f['sha256']
old=json.loads((P/'P2_RUNTIME_MANIFEST.json').read_text());reg=Registry(old,old['manifest_hash'])
moved=json.loads((P/'PROPOSAL_MOVED_MEMBERS.json').read_text());assert moved['members_hash']==MOVED_HASH
records=copy.deepcopy(reg.records);changed=[]
for r in records:
    newrole=role_for(r['source']['source_sha256'])
    if newrole!=r['role']:changed.append(r['window_id'])
    r['role']=newrole;r['source']['logical_role']=newrole;r['isolation_group']=group_for(r['source'])
    r['artifacts']['observation']=None
assert set(changed)=={r['window_id'] for r in moved['members']} and len(changed)==947
assets=json.loads((P/'MAP_ASSET_PROVENANCE.json').read_text())
q=dict(release_binding=dict(status='UNKNOWN',evidence=[]),group_isolation=dict(status='UNKNOWN',evidence=[]),
    known_cross_role_conflict=False,scenes={s:dict(geometry_status='UNKNOWN',semantic_status='UNKNOWN',
        protocol_permitted=False,evidence=[]) for s in assets})
m=sealed(dict(schema=SCHEMA,family=FAMILY,source_commit=START,original_source_rows_hash=reg.manifest.get('registered_source_rows_hash',
    digest([r['source'] for r in reg.records])),registered_source_rows_hash=digest([r['source'] for r in records]),
    original_manifest_hash=old['manifest_hash'],moved_member_hash=MOVED_HASH,records=records,
    train_order=order(records),train_order_hash=digest(order(records)),down_factor=8,
    qualification=q,assets=assets,parents=dict(goal=None,base=None),cache=None,shared_initial_state=None,
    stage_a_reference_total_steps=2537,preflight_status='PREPARATION_ONLY_PROVENANCE_PENDING'))
new=Registry(m,m['manifest_hash']);assert new.counts==COUNTS
pack={}
for role,expected,exposures in [('train',175,11122),('inner_valid',391,24955)]:
    rows=base_packing_plan(new,role)
    assert len(rows)==expected and sum(map(len,rows))==exposures
    assert all(len({new.by_id[w]['source']['source_sha256'] for w,_ in row})==1 for row in rows)
    pack[role]=dict(packs=expected,agent_window_exposures=exposures,packing_hash=digest(rows),pack_sizes=[len(x) for x in rows])
atomic_json(m,D/'PREPARATION_MANIFEST.json')
atomic_json(dict(status='AUTHORIZED_IMPLEMENTED',moved=changed,moved_count=947,moved_member_hash=MOVED_HASH,
    new_source_rows_hash=m['registered_source_rows_hash'],original_source_rows_hash=m['original_source_rows_hash'],
    train_order_hash=m['train_order_hash'],packing=pack,stage_a_reference_total_steps=2537,
    old_protocol_unchanged=True,window_ids_preserved=True),D/'MIGRATION_RECEIPT.json')
atomic_json(dict(files=reads,all_old_catalog_runtime_shards_parsed=True,old_immutable_archive_hashes_valid=True,
    actual_payload_verification='NEXT_DATA_PHASE_NOT_YET_CLAIMED',guard=guard()),D/'INPUT_READS.json')
print(json.dumps(dict(manifest=m['manifest_hash'],rows=m['registered_source_rows_hash'],packing=pack)))
