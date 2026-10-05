"""Purpose-bound local preparation; never grants production qualification.

Every numeric raw lookup is routed to an authorized (window,t,frame,agent)
slot. Reading source bytes for SHA is not a coordinate authorization.
"""
import copy,json,os,tempfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from src.p2_protocol import ROOT,Registry,digest,file_hash,path,sealed,FileProviders,validate_payload
from src.p2_grouped import FAMILY,SCHEMA,COUNTS
from src.p2_observation import (atomic_json,integer,load_assets,content_hash,
    validate_observation,producer_hash,CONTRACT)
from src.data_src.data_utils import world2pixel_numpy
from src.models.model_utils.cnn_big_images_utils import create_CNN_inputs_loop

def write_shards(rows,folder,prefix):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);refs=[]
    for i in range(0,len(rows),125):
        p=folder/(prefix+'_%03d.jsonl'%(i//125))
        raw=''.join(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for r in rows[i:i+125])
        if p.exists() and p.read_text()!=raw:raise RuntimeError('immutable shard already differs')
        if not p.exists():
            fd,tmp=tempfile.mkstemp(dir=folder,prefix='.shard-')
            with os.fdopen(fd,'w') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            os.replace(tmp,p)
        refs.append(dict(path=str(p.relative_to(ROOT)),sha256=file_hash(p),count=len(rows[i:i+125])))
    return refs

def write_manifest(m,rows,dest):
    m=copy.deepcopy(m);m.pop('records',None);m.pop('record_files',None)
    dest=Path(dest)
    m['record_files']=write_shards(rows,dest.parent/(dest.stem.lower()+'_shards'),'rows')
    m=sealed(m);atomic_json(m,dest);return m

def make_grant(reg):
    return sealed(dict(schema='grouped-purpose-preparation-v1',family=FAMILY,
        parent_manifest_hash=reg.manifest['manifest_hash'],rows_hash=reg.manifest['registered_source_rows_hash'],
        members_hash=digest([(r['window_id'],r['role'],digest(r['source'])) for r in reg.records]),
        uses={'train':['observation_validation','gradient_target_preparation','PREFLIGHT_SMOKE'],
              'inner_valid':['observation_validation','metric_target_preparation'],
              'outer':['observation_validation']},production=False,outer_future=False,teacher=False))

class PreparationGrant:
    def __init__(self,reg,value):
        if value!=make_grant(reg):raise RuntimeError('preparation grant parent/hash/purpose')
        self.reg=reg;self.value=value
    def authorize(self,r,purpose):
        expected=self.reg.by_id.get(r.get('window_id'))
        if expected is None or {k:v for k,v in r.items() if k!='artifacts'}!={k:v for k,v in expected.items() if k!='artifacts'}:raise RuntimeError('source/member identity')
        if purpose not in self.value['uses'][r['role']]:raise PermissionError('preparation role/purpose')
        s=r['source'];f=s['frame_ids'];a=s['agent_ids']
        if len(f)!=20 or f!=sorted(set(f)) or len(a)!=len(set(a)) or not a:raise RuntimeError('slot identity')
        return digest([self.value['manifest_hash'],r['window_id'],purpose])

def target_purpose(r):
    if r['role']=='outer':raise PermissionError('outer target always forbidden')
    return 'gradient_target_preparation' if r['role']=='train' else 'metric_target_preparation'

def scan_targets(records,grant,ledger):
    arrays={};filled={};routes=defaultdict(list)
    if not records:return arrays
    s=records[0]['source'];p=path(s['original_source_path'])
    for r in records:
        token=grant.authorize(r,target_purpose(r));v=r['source']
        if (v['source_sha256'],v['original_source_path'])!=(s['source_sha256'],s['original_source_path']):raise RuntimeError('mixed source')
        wid=r['window_id'];arrays[wid]=np.empty((12,len(v['agent_ids']),2),dtype=np.float64)
        filled[wid]=np.zeros(arrays[wid].shape[:2],dtype=bool)
        for t,frame in enumerate(v['frame_ids'][8:],8):
            for j,agent in enumerate(v['agent_ids']):routes[(integer(frame),integer(agent))].append((wid,t,j,token))
    if file_hash(p)!=s['source_sha256']:raise RuntimeError('source bytes before decode')
    ledger['raw_hash_bytes']+=p.stat().st_size
    seen=set()
    with p.open('rb') as f:
        for line in f:
            ledger['raw_scan_bytes']+=len(line);tokens=line.split()
            if not tokens:continue
            if len(tokens)!=4:raise RuntimeError('four-column source')
            key=integer(tokens[0]),integer(tokens[1]);slots=routes.get(key)
            if slots is None:continue
            if key in seen:raise RuntimeError('duplicate authorized key')
            seen.add(key);xy=float(tokens[2]),float(tokens[3])
            if not np.isfinite(xy).all():raise RuntimeError('nonfinite target')
            ledger['unique_future_xy_decodes']+=1
            for wid,t,j,token in slots:
                if not 8<=t<20 or not token:raise PermissionError('future slot grant')
                arrays[wid][t-8,j]=xy;filled[wid][t-8,j]=True
                ledger['window_local_target_slots']+=1
    if not all(x.all() for x in filled.values()):raise RuntimeError('missing fixed cohort slot')
    if file_hash(p)!=s['source_sha256']:raise RuntimeError('source changed during scan')
    ledger['raw_hash_bytes']+=p.stat().st_size
    return arrays

def build_target(xy,r,assets):
    n=len(r['source']['agent_ids'])
    if xy.shape!=(12,n,2) or xy.dtype!=np.float64:raise RuntimeError('target raw shape/dtype')
    px=world2pixel_numpy(xy.reshape(-1,2),assets['H_inv']).reshape(12,n,2)
    if r['source']['scene_family'] in ('eth','hotel'):px=px[...,[1,0]]
    coord=torch.from_numpy(px.copy())
    value=dict(abs_pixel_coord=coord,seq_list=torch.ones(12,n,dtype=torch.float64),
        input_traj_maps=create_CNN_inputs_loop(coord.float()/8,assets['tensor_image']))
    validate_target(value,r)
    return value

def validate_target(v,r):
    validate_payload('target',v);n=len(r['source']['agent_ids'])
    if set(v)!={'abs_pixel_coord','seq_list','input_traj_maps'}:raise RuntimeError('target fields')
    if v['abs_pixel_coord'].shape!=(12,n,2) or v['abs_pixel_coord'].dtype!=torch.float64:raise RuntimeError('target coord')
    if v['seq_list'].shape!=(12,n) or v['seq_list'].dtype!=torch.float64 or not v['seq_list'].eq(1).all():raise RuntimeError('target mask')
    maps=v['input_traj_maps']
    if maps.shape[:2]!=(n,12) or maps.dtype!=torch.float32 or maps.min()<0:raise RuntimeError('target maps')
    if not maps.flatten(2).max(-1).values.eq(1).all():raise RuntimeError('target peak normalization')

def atomic_tensor(value,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise RuntimeError('refuse tensor overwrite')
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.grouped-')
    try:
        with os.fdopen(fd,'wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
        fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def save_payload(value,r,kind,root,grant,extra=None):
    p=Path(root)/kind/(r['window_id']+'.pt');side=p.with_suffix('.json')
    purpose='observation_validation' if kind=='observation' else target_purpose(r)
    grant.authorize(r,purpose)
    expected=dict(schema='grouped-artifact-v1',window_id=r['window_id'],kind=kind,role=r['role'],
        source_identity=digest(r['source']),grant_hash=grant.value['manifest_hash'],purpose=purpose,
        parent_manifest_hash=grant.reg.manifest['manifest_hash'],content_sha256=content_hash(value),
        source_sha256=r['source']['source_sha256'],frames=r['source']['frame_ids'][:8] if kind=='observation' else r['source']['frame_ids'][8:],
        agents=r['source']['agent_ids'],producer_sha256=file_hash(__file__),extra=extra or {})
    if p.exists() or side.exists():
        if not p.is_file() or not side.is_file():raise RuntimeError('incomplete artifact pair')
        meta=json.loads(side.read_text())
        if any(meta.get(k)!=v for k,v in expected.items()) or sealed(meta)!=meta:raise RuntimeError('existing sidecar mismatch')
        if file_hash(p)!=meta['sha256']:raise RuntimeError('existing file mismatch')
    else:
        atomic_tensor(dict(window_id=r['window_id'],kind=kind,source_identity=digest(r['source']),payload=value),p)
        meta=sealed(dict(expected,sha256=file_hash(p)));atomic_json(meta,side)
    rr=copy.deepcopy(r);rr['artifacts'][kind]=dict(path=str(p),sha256=meta['sha256'])
    read=FileProviders().read(rr,kind)
    if content_hash(read)!=expected['content_sha256']:raise RuntimeError('readback content')
    return dict(window_id=r['window_id'],kind=kind,role=r['role'],artifact=rr['artifacts'][kind],
        sidecar=dict(path=str(side),sha256=file_hash(side)),metadata=meta,bytes=p.stat().st_size)

def migrate_observation(old,r,grant):
    # Verify original wrapper, sidecar, source/grant/assets/producer and all tensors.
    ref=old['artifacts']['observation'];p=path(ref['path']);side=p.with_suffix('.json')
    m=json.loads(side.read_text());grant.authorize(r,'observation_validation')
    if sealed(m)!=m or m['artifact_sha256']!=ref['sha256'] or m['source_identity']!=digest(old['source']):raise RuntimeError('old sidecar identity')
    if m['producer_hash']!=producer_hash() or m['contract']!=CONTRACT:raise RuntimeError('old producer')
    if m['asset_binding_hash']!=digest(grant.reg.manifest['assets'][r['source']['scene_family']]):raise RuntimeError('old assets')
    if m['grant_hash']!='fc383470111b55b704bb305629e7ac3dee2e1b865a616d34b67b9d9c68909959':raise RuntimeError('old grant')
    value=FileProviders().read(old,'observation');validate_observation(value,r)
    if content_hash(value)!=m['content_sha256']:raise RuntimeError('old tensor content')
    return value,dict(old_artifact=ref,old_sidecar_sha256=file_hash(side),old_content_sha256=m['content_sha256'])
