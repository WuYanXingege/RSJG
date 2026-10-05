"""Independent grouped-UNIV contract. No generic UNKNOWN bypass."""
import copy,json
from collections import Counter
from pathlib import Path
from src.p2_protocol import (ROOT,Registry,REGISTERED_ROWS,HOTEL,INNER,digest,sealed,path,
    file_hash,identity,expected_role)

FAMILY='p2_grouped_univ_hotel_v1'
SCHEMA='rsjg-p2-grouped-role-v1'
CACHE_SCHEMA='jdv2-p2-grouped-cache-v1'
START='7a0f83bb9341617b4f214cefc3563927961c08d6'
MOVED_HASH='bdcdd21cf6dd8995bea27d8881519704e37f8027a6d3b07b569cf884ce449ea9'
UNIV={INNER,'a6d87f278d94136fe39b8be91555487a29ac77259ae403b9dba2d5c18caf7b5b',
      'e25798b660634330aa89f8bb259425de720e84d0873902726c1d1f4ccff21d6c'}
COUNTS={'train':2537,'inner_valid':1267,'outer':445}
def role_for(sha):return 'outer' if sha==HOTEL else 'inner_valid' if sha in UNIV else 'train'
def grouped(reg):return reg.manifest.get('family')==FAMILY
def order(records):
    return [r['window_id'] for r in sorted((r for r in records if r['role']=='train'),
        key=lambda r:digest(['rsjg.grouped.batch-order.v1',3101,r['source']['source_id'],r['source']['id_sha256']]))]
def group_for(source):
    scene=source['scene_family']
    return 'UNIV_CONSERVATIVE' if scene=='univ' else 'ZARA_CONSERVATIVE' if scene in {'zara1','zara2'} else scene.upper()
def init_registry(reg,manifest,expected_hash):
    m=copy.deepcopy(manifest)
    if m.get('schema')!=SCHEMA or m.get('family')!=FAMILY or sealed(m)['manifest_hash']!=expected_hash or m.get('manifest_hash')!=expected_hash:
        raise RuntimeError('GROUPED manifest/schema/hash')
    rows=m.get('records')
    if rows is None:
        rows=[]
        for shard in m['record_files']:
            p=path(shard['path'])
            if file_hash(p)!=shard['sha256']:raise RuntimeError('GROUPED shard hash')
            loaded=[json.loads(x) for x in p.read_text().splitlines()]
            if len(loaded)!=shard['count']:raise RuntimeError('GROUPED shard count')
            rows.extend(loaded)
    original=copy.deepcopy(rows)
    for r in original:
        r['role']=expected_role(r['source']['source_sha256']);r['source']['logical_role']=r['role']
    if digest([r['source'] for r in original])!=REGISTERED_ROWS:
        raise RuntimeError('GROUPED immutable original rows/address/cohort')
    # Reuse all legacy structural checks with their actual, original metadata roles.
    old_order=[r['window_id'] for r in sorted((r for r in original if r['role']=='train'),
        key=lambda r:digest(['rsjg.p2.batch-order.v1',3101,r['source']['source_id'],r['source']['id_sha256']]))]
    old=sealed(dict(schema='rsjg-p2-role-v1',family='p2_hotel_uni_examples',records=original,
        train_order=old_order,train_order_hash=digest(old_order)))
    Registry(old,old['manifest_hash'])
    for r in rows:
        if r['role']!=role_for(r['source']['source_sha256']) or r['source']['logical_role']!=r['role']:
            raise RuntimeError('GROUPED registered source role')
        if r.get('isolation_group')!=group_for(r['source']):raise RuntimeError('GROUPED isolation assignment')
    if dict(Counter(r['role'] for r in rows))!=COUNTS:raise RuntimeError('GROUPED role counts')
    if m['registered_source_rows_hash']!=digest([r['source'] for r in rows]):raise RuntimeError('GROUPED new rows hash')
    if m['original_source_rows_hash']!=REGISTERED_ROWS or m['moved_member_hash']!=MOVED_HASH:raise RuntimeError('GROUPED migration anchor')
    if m['train_order']!=order(rows) or m['train_order_hash']!=digest(m['train_order']):raise RuntimeError('GROUPED order')
    groups={}
    for r in rows:groups.setdefault(r['isolation_group'],set()).add(r['role'])
    if any(len(v)!=1 for v in groups.values()):raise RuntimeError('GROUPED cross-role isolation conflict')
    reg.manifest=m;reg.records=rows;reg.by_id={r['window_id']:r for r in rows};reg.counts=COUNTS.copy()
def data_binding(reg):
    return digest(dict(schema=SCHEMA,family=FAMILY,
        rows=[dict(source=r['source'],isolation_group=r['isolation_group']) for r in reg.records],
        assets=reg.manifest['assets'],order_hash=reg.manifest['train_order_hash'],
        qualification_contract='grouped-asset-specific-semantic-v1',
        policy='train-gradient_inner-post-prediction-metric_outer-observation-only-v1'))
def qualification_issues(reg,roles=('train','inner_valid')):
    q=reg.manifest['qualification'];issues=[]
    if q.get('known_cross_role_conflict'):issues.append('KNOWN_CROSS_ROLE_CONFLICT')
    if q.get('release_binding',{}).get('status')!='VERIFIED_AUTHOR_ARCHIVE_MEMBERS':issues.append('AUTHOR_RELEASE_MEMBER_BINDING')
    if q.get('group_isolation',{}).get('status')!='VERIFIED_CROSS_ROLE_GROUPS':issues.append('CROSS_ROLE_GROUP_EVIDENCE')
    scenes={r['source']['scene_family'] for r in reg.records if r['role'] in roles}
    for scene in sorted(scenes):
        a=q.get('scenes',{}).get(scene,{})
        if a.get('geometry_status')!='VERIFIED_AUTHOR_BINDING':issues.append('GEOMETRY_'+scene)
        if a.get('semantic_status') not in {'VERIFIED_PROVIDED_STATIC','VERIFIED_TRAIN_ONLY_PREPROCESSOR'} or a.get('protocol_permitted') is not True:
            issues.append('SEMANTIC_QUALIFICATION_'+scene)
        else:
            # Distribution membership cannot be reused as a segmentation-training certificate.
            qualified=False
            for ref in a.get('semantic_evidence',[]):
                p=path(ref['path'])
                if not p.is_file() or file_hash(p)!=ref['sha256']:continue
                evidence=json.loads(p.read_text())
                if evidence.get('schema')!='grouped-semantic-provenance-v1' or evidence.get('asset_sha256')!=reg.manifest['assets'][scene]['pred_mask.png']['sha256']:continue
                if evidence.get('uses_future_trajectories') is not False or evidence.get('uses_holdout_training_labels') is not False or evidence.get('license_permitted') is not True:continue
                if evidence.get('producer_type')=='learned':
                    if len(evidence.get('producer_checkpoint_sha256',''))!=64 or not evidence.get('training_scene_groups') or not set(evidence['training_scene_groups'])<={'ETH','ZARA_CONSERVATIVE'}:continue
                elif evidence.get('producer_type')!='provided_static':continue
                anchors=evidence.get('author_evidence',[])
                if anchors and all(path(x['path']).is_file() and file_hash(path(x['path']))==x['sha256'] for x in anchors):qualified=True
            if not qualified:issues.append('SEMANTIC_ARTIFACT_SPECIFIC_EVIDENCE_'+scene)
        for name in ('H.txt','RGB.jpg','pred_mask.png'):
            ref=reg.manifest['assets'][scene][name]
            if not path(ref['path']).is_file() or file_hash(path(ref['path']))!=ref['sha256']:issues.append('ASSET_BYTES_'+scene+'_'+name)
    # Positive status alone is not evidence: every positive qualification needs a bound receipt.
    entries=[q.get('release_binding',{}),q.get('group_isolation',{})]+[q.get('scenes',{}).get(s,{}) for s in scenes]
    for entry in entries:
        if any(str(v).startswith('VERIFIED') for k,v in entry.items() if k.endswith('status')):
            refs=entry.get('evidence',[])
            if not refs or any(not path(r['path']).is_file() or file_hash(path(r['path']))!=r['sha256'] for r in refs):
                issues.append('QUALIFICATION_EVIDENCE_RECEIPT')
    return sorted(set(issues))
def require_provenance(reg,roles=('train','inner_valid')):
    issues=qualification_issues(reg,roles)
    if issues:raise RuntimeError('EXTERNAL_EVIDENCE_BLOCKED:'+','.join(issues))
def authorize(reg,wid,purpose):
    r=reg.by_id[wid];role=r['role']
    policies={('train','gradient'):(True,False),('train','jdv2_gradient'):(True,True),
        ('train','teacher_build_train'):(True,False),('inner_valid','selection'):(True,False)}
    if purpose=='candidate_build':target,teacher=False,False
    elif (role,purpose) in policies:target,teacher=policies[role,purpose]
    else:raise PermissionError('GROUPED role/purpose denied before provider')
    require_provenance(reg,(role,))
    return dict(role=role,purpose=purpose,allow_target=target,allow_teacher=teacher,augmentation=False)
