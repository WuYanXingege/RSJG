"""Metadata-only evidence/proposal preparation, without changing production roles."""
import copy,hashlib,json,subprocess
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import sys
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from src.p2_observation_guard import install
_,guard=install()
from src.p2_protocol import Registry,digest,file_hash
from src.p2_data import base_packing_plan
from src.p2_observation import atomic_json
D=Path(__file__).resolve().parent
P=R/'docs/joint_dependency_v2/reviews/2026-10-05_28728bf_p2_role_loader_step_cap_cpu'
m=json.loads((P/'P2_RUNTIME_MANIFEST.json').read_text());reg=Registry(m,m['manifest_hash'])
# Validate the entire old archive without modifying it; living index is excluded here.
old=json.loads((P/'EVIDENCE_MANIFEST.json').read_text())
checked=[]
for f in old['files']:
    if f['path'].startswith(str(P.relative_to(R))+'/'):
        assert file_hash(R/f['path'])==f['sha256'];checked.append(f)
atomic_json(dict(input_commit='bda1ae147d5912e8847642db36b5822584420563',
    actual_start_head='bda1ae147d5912e8847642db36b5822584420563',input_relation='IDENTICAL',
    old_archive_hash_checks=checked,source_rows_digest=digest([r['source'] for r in reg.records]),
    all_records_parsed=4249,old_five_checkpoint_reads=0),D/'INPUT_EVIDENCE_READS.json')
sources=json.loads((D/'RECORDING_RELEASE_BINDING.json').read_text())
search_file=D/'PRIMARY_SOURCE_SEARCH.json'
for s in sources:
    s.update(status='PARTIAL',recording_status='UNKNOWN',
        release_owner='GDTS author cites Goal-SAR dataset package; local member binding UNKNOWN',
        release_version_or_commit='GDTS download script at 297d508558c10831983ea4b19c2b3e657459a449; not archive version',
        evidence_paths_and_sha256=[dict(path=str(search_file.relative_to(R)),sha256=file_hash(search_file))])
atomic_json(sources,D/'RECORDING_RELEASE_BINDING.json')
# Do not change MAP_ASSET_PROVENANCE: it is already hashed into canary sidecars.
maps=json.loads((D/'MAP_ASSET_PROVENANCE.json').read_text())
usage={}
for scene in maps:
    refs=[r for r in reg.records if r['source']['scene_family']==scene]
    usage[scene]=dict(source_names=sorted({Path(r['source']['original_source_path']).name for r in refs}),
        roles=dict(Counter(r['role'] for r in refs)),assets=maps[scene],
        tensor_image='pred_mask.png -> cv2 grayscale -> 6 float32 one-hot channels',
        channel_order=['unlabeled','pavement','road','structure','terrain','tree'],
        tensor_map='RGB.jpg -> original uint8 ToTensor /255 -> 3 float32 channels',
        semantic_producer='UNKNOWN',training_labels_and_weights='UNKNOWN',
        protocol_permitted=False,original_geometry_reuse_evidence='UNKNOWN',
        actual_current_code_binding='scene_family inherited from archived cohort; only static files opened')
atomic_json(usage,D/'MAP_SOURCE_USAGE_AND_GAPS.json')
moved=[r for r in reg.records if Path(r['source']['original_source_path']).name in {'students001.txt','students003.txt'}]
assert len(moved)==947 and all(r['role']=='train' for r in moved)
proposal=copy.deepcopy(reg.records)
ids={r['window_id'] for r in moved}
for r in proposal:
    if r['window_id'] in ids:r['role']='inner_valid'
view=SimpleNamespace(refs_unordered=lambda role:[r for r in proposal if r['role']==role])
packs={role:base_packing_plan(view,role) for role in ('train','inner_valid')}
counts=dict(Counter(r['role'] for r in proposal))
assert counts=={'train':2537,'inner_valid':1267,'outer':445}
members=[dict(window_id=r['window_id'],source_sha256=r['source']['source_sha256'],
    source_name=Path(r['source']['original_source_path']).name,old_role='train',proposed_role='inner_valid') for r in moved]
atomic_json(dict(status='PROPOSAL_NOT_APPROVED',count=len(members),members_hash=digest(members),members=members),
    D/'PROPOSAL_MOVED_MEMBERS.json')
atomic_json(dict(status='PROPOSAL_NOT_APPROVED',family_draft='p2_grouped_univ_hotel_v0',
    schema_draft='rsjg-p2-grouped-proposal-v0',counts=counts,
    source_role_mapping={Path(r['source']['original_source_path']).name:r['role'] for r in proposal},
    fixed_input_rows_hash=digest([r['source'] for r in reg.records]),moved_windows=947,
    moved_member_hash=digest(members),packing={role:dict(packs=len(rows),agent_window_exposures=sum(map(len,rows)),
        packing_hash=digest(rows)) for role,rows in packs.items()},
    goal_max_updates=len(packs['train'])*150,joint_max_updates=len(packs['train'])*250,
    nominal_update_ratio_to_existing=len(packs['train'])/556,
    wall_time_estimate='UNKNOWN; cannot infer actual throughput or convergence',
    proposed_l1_native_windows=2537,proposed_l1_reference_total_steps='REQUIRES_NEW_PREREGISTRATION; not silently keep or change 3484',
    production_changes=0,new_train_order=None,recording_independence_proven=False,
    map_qualification='UNKNOWN'),D/'SPLIT_GROUPING_PROPOSAL.json')
atomic_json(guard(),D/'METADATA_PROPOSAL_PROCESS.json')
print(json.dumps(dict(proposal=counts,packs={k:len(v) for k,v in packs.items()},old_archive_hashes=len(checked))))
