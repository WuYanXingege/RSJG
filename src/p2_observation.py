"""Explicit observation preparation, NOT a production provenance bypass.

Only per-window obs8 routes may decode numeric coordinates. No real target API.
"""
import copy
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time

import numpy as np
import torch

from src.p2_protocol import Registry,REGISTERED_ROWS,ROOT,digest,file_hash,path,sealed,identity,FileProviders
from src.data_src.data_utils import world2pixel_numpy
from src.models.model_utils.cnn_big_images_utils import create_tensor_image,create_CNN_inputs_loop

START_COMMIT='bda1ae147d5912e8847642db36b5822584420563'
GRANT_SCHEMA='rsjg-p2-observation-preparation-v1'
CONTRACT='original-pixel-f64-obs8-cnn-f32-v1'
SOURCE_COUNTS={'biwi_eth.txt':139,'biwi_hotel.txt':445,'crowds_zara01.txt':705,
    'crowds_zara02.txt':998,'crowds_zara03.txt':695,'students001.txt':425,
    'students003.txt':522,'uni_examples.txt':320}
DENIED_FLAGS=['allow_real_future_coordinate_queries','allow_full_source_pickle_deserialization',
    'allow_real_target_materialization','allow_model_calls','allow_training','allow_outer_metrics',
    'production_provenance_guard_changed']


def atomic_json(value,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.obs-json-',dir=target.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,target)
        d=os.open(target.parent,os.O_RDONLY);os.fsync(d);os.close(d)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def registry_from_file(manifest,expected):
    reg=Registry(json.loads(path(manifest).read_text()),expected)
    counts=Counter(Path(r['source']['original_source_path']).name for r in reg.records)
    if counts!=SOURCE_COUNTS:raise RuntimeError('registered source counts differ')
    return reg


def make_grant(reg):
    return sealed(dict(schema=GRANT_SCHEMA,start_head=START_COMMIT,fixed_input=START_COMMIT,
        parent_manifest_hash=reg.manifest['manifest_hash'],
        registered_source_rows_hash=digest([r['source'] for r in reg.records]),
        window_ids_hash=digest([r['window_id'] for r in reg.records]),
        purposes=['artifact_preparation','artifact_validation'],obs_steps=8,
        **{k:False for k in DENIED_FLAGS}))


class PreparationGate:
    def __init__(self,registry,grant,expected_hash):
        self.registry=registry;self.grant=grant
        if grant!=make_grant(registry) or grant.get('manifest_hash')!=expected_hash:
            raise RuntimeError('preparation grant/hash/member binding')
    def authorize(self,record,purpose):
        if purpose not in self.grant['purposes']:raise PermissionError('preparation purpose denied')
        if record!=self.registry.by_id.get(record.get('window_id')):
            raise RuntimeError('preparation record identity/address changed')
        s=record['source']
        if len(set(s['frame_ids']))!=20 or s['frame_ids']!=sorted(s['frame_ids']):
            raise RuntimeError('frame order')
        if not s['agent_ids'] or len(set(s['agent_ids']))!=len(s['agent_ids']):
            raise RuntimeError('agent namespace/order')
        return record


def integer(token):
    try:d=Decimal(token.decode() if isinstance(token,bytes) else str(token))
    except InvalidOperation:raise RuntimeError('invalid ID token')
    if not d.is_finite() or d!=d.to_integral_value() or not -(2**63)<=d<2**63:
        raise RuntimeError('non-integral ID token')
    return int(d)


def scan_observations(records,gate,ledger):
    """One source scan, window-local route slots (NOT a global frame permission)."""
    if not records:return {}
    for r in records:gate.authorize(r,'artifact_preparation')
    s=records[0]['source'];source=path(s['original_source_path'])
    if any(r['source']['source_sha256']!=s['source_sha256'] or
           r['source']['original_source_path']!=s['original_source_path'] for r in records):
        raise RuntimeError('mixed/unbound source')
    if file_hash(source)!=s['source_sha256']:raise RuntimeError('raw source hash before coordinate parse')
    ledger['raw_hash_passes']+=1;ledger['raw_hash_bytes']+=source.stat().st_size
    routes=defaultdict(list);arrays={};filled={}
    for r in records:
        wid=r['window_id'];a=r['source']['agent_ids'];f=r['source']['frame_ids'][:8]
        arrays[wid]=np.empty((8,len(a),2),dtype=np.float64)
        filled[wid]=np.zeros((8,len(a)),dtype=bool)
        for t,frame in enumerate(f):
            for j,agent in enumerate(a):routes[(integer(frame),integer(agent))].append((wid,t,j))
    seen=set()
    with source.open('rb') as handle:
        for line in handle:
            ledger['raw_scan_bytes']+=len(line)
            tokens=line.split()
            if not tokens:continue
            if len(tokens)!=4:raise RuntimeError('raw row must have four columns')
            key=(integer(tokens[0]),integer(tokens[1]));ledger['id_rows_parsed']+=1
            slots=routes.get(key)
            if slots is None:continue
            if key in seen:raise RuntimeError('duplicate authorized raw key')
            seen.add(key)
            xy=(float(tokens[2]),float(tokens[3]))
            if not np.isfinite(xy).all():raise RuntimeError('nonfinite observation coordinate')
            ledger['unique_raw_obs_xy_decodes']+=1
            for wid,t,j in slots:
                if not 0<=t<8:raise PermissionError('window future route')
                arrays[wid][t,j]=xy;filled[wid][t,j]=True
                ledger['window_local_observation_slots']+=1
    if not all(v.all() for v in filled.values()):raise RuntimeError('missing registered observation slots')
    # Detect a source modified between pre-hash and streaming.
    if file_hash(source)!=s['source_sha256']:raise RuntimeError('raw source changed during scan')
    ledger['raw_hash_passes']+=1;ledger['raw_hash_bytes']+=source.stat().st_size
    return arrays


def inspect_assets(scene):
    from PIL import Image
    folder=ROOT/'data/eth5'/scene
    result={}
    for name in ('H.txt','RGB.jpg','pred_mask.png'):
        p=folder/name
        item=dict(path=str(p.resolve()),sha256=file_hash(p),bytes=p.stat().st_size)
        if name!='H.txt':
            with Image.open(p) as im:item.update(size=list(im.size),mode=im.mode)
        result[name]=item
    if result['RGB.jpg']['size']!=result['pred_mask.png']['size']:
        raise RuntimeError('RGB semantic raster size mismatch')
    result['qualification']='UNKNOWN'
    result['binding_scope']='current source scene_family and existing preprocessing; original recording/map reuse NOT certified'
    return result


def load_assets(binding):
    import cv2
    from PIL import Image
    cv2.setNumThreads(1)
    for k in ('H.txt','RGB.jpg','pred_mask.png'):
        if file_hash(binding[k]['path'])!=binding[k]['sha256']:raise RuntimeError('static asset hash')
    H=np.loadtxt(binding['H.txt']['path'])
    if H.shape!=(3,3) or not np.isfinite(H).all() or np.linalg.det(H)==0:
        raise RuntimeError('invalid H')
    with Image.open(binding['RGB.jpg']['path']) as im:rgb=np.array(im)
    semantic=cv2.imread(binding['pred_mask.png']['path'],flags=0)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or semantic is None:
        raise RuntimeError('invalid RGB/semantic image')
    if rgb.shape[:2]!=semantic.shape or not np.isin(semantic,np.arange(6)).all():
        raise RuntimeError('semantic classes/shape')
    onehot=np.stack([semantic==i for i in range(6)],axis=-1).astype(np.float32)
    return dict(H=H,H_inv=np.linalg.inv(H),tensor_image=create_tensor_image(onehot,8),
        tensor_map=create_tensor_image(rgb,8),original_shape=list(rgb.shape[:2]),
        labels=sorted(map(int,np.unique(semantic))))


def build_payload(xy8,record,assets):
    n=len(record['source']['agent_ids'])
    if xy8.shape!=(8,n,2) or xy8.dtype!=np.float64 or not np.isfinite(xy8).all():
        raise RuntimeError('obs shape/dtype/finite')
    pixel=world2pixel_numpy(xy8.reshape(-1,2),assets['H_inv']).reshape(8,n,2)
    if record['source']['scene_family'] in ('eth','hotel'):pixel=pixel[..., [1,0]]
    if not np.isfinite(pixel).all():raise RuntimeError('homogeneous projection nonfinite')
    coord=torch.from_numpy(pixel.copy())
    maps=create_CNN_inputs_loop(coord.float()/8,assets['tensor_image'])
    payload=dict(abs_pixel_coord=coord,seq_list=torch.ones(8,n,dtype=torch.float64),
        frame_ids=torch.tensor(record['source']['frame_ids'][:8],dtype=torch.int64)[:,None].repeat(1,n),
        input_traj_maps=maps,tensor_image=assets['tensor_image'],tensor_map=assets['tensor_map'],
        scene_index=torch.zeros(n,dtype=torch.int64),scene_ptr=torch.tensor([0,n],dtype=torch.int64),
        batch_format_version=torch.tensor(2,dtype=torch.int64))
    validate_observation(payload,record)
    h,w=assets['tensor_image'].shape[-2:]
    outside=((coord[...,0]/8<0)|(coord[...,0]/8>w-1)|(coord[...,1]/8<0)|(coord[...,1]/8>h-1)).sum()
    return payload,int(outside)


def validate_observation(p,r):
    from src.p2_protocol import validate_payload
    validate_payload('observation',p)
    n=len(r['source']['agent_ids']);h,w=p['tensor_image'].shape[-2:]
    specs={'abs_pixel_coord':((8,n,2),torch.float64),'seq_list':((8,n),torch.float64),
        'frame_ids':((8,n),torch.int64),'input_traj_maps':((n,8,h,w),torch.float32),
        'tensor_image':((6,h,w),torch.float32),'tensor_map':((3,h,w),torch.float32),
        'scene_index':((n,),torch.int64),'scene_ptr':((2,),torch.int64),'batch_format_version':((),torch.int64)}
    if set(p)!=set(specs):raise RuntimeError('observation fields')
    for k,(shape,dtype) in specs.items():
        if tuple(p[k].shape)!=shape or p[k].dtype!=dtype or p[k].device.type!='cpu':
            raise RuntimeError('observation shape/dtype/device '+k)
    expected=torch.tensor(r['source']['frame_ids'][:8])[:,None].expand(8,n)
    if not torch.equal(p['frame_ids'],expected):raise RuntimeError('observation frame order')
    if not p['seq_list'].eq(1).all() or not p['scene_index'].eq(0).all():raise RuntimeError('cohort/mask')
    if p['scene_ptr'].tolist()!=[0,n] or p['batch_format_version'].item()!=2:raise RuntimeError('scene pointer/version')
    if not ((p['tensor_image']==0)|(p['tensor_image']==1)).all() or not p['tensor_image'].sum(0).eq(1).all():
        raise RuntimeError('semantic one-hot')
    if p['tensor_map'].min()<0 or p['tensor_map'].max()>1:raise RuntimeError('RGB scale')
    peaks=p['input_traj_maps'].flatten(2).max(-1).values
    if not torch.equal(peaks,torch.ones_like(peaks)) or p['input_traj_maps'].min()<0:
        raise RuntimeError('original Gaussian normalization')


def content_hash(payload):
    h=hashlib.sha256()
    for k,v in sorted(payload.items()):
        h.update(k.encode());h.update(str((tuple(v.shape),v.dtype)).encode())
        h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


def producer_hash():
    return digest({p:file_hash(ROOT/p) for p in [
        'src/p2_observation.py','src/models/model_utils/cnn_big_images_utils.py',
        'src/models/model_utils/sampling_2D_map.py','src/data_src/data_utils.py']})


def verify_artifact(r,root,gate,asset_binding,permit):
    gate.authorize(r,'artifact_validation')
    p=Path(root)/(r['window_id']+'.pt');side=p.with_suffix('.json')
    if not p.is_file() or not side.is_file():raise RuntimeError('missing artifact/sidecar pair')
    meta=json.loads(side.read_text())
    if sealed(meta)['manifest_hash']!=meta.get('manifest_hash'):raise RuntimeError('sidecar self hash')
    expected=dict(window_id=r['window_id'],source_identity=digest(r['source']),
        raw_sha256=r['source']['source_sha256'],grant_hash=gate.grant['manifest_hash'],
        asset_binding_hash=digest(asset_binding),producer_hash=producer_hash(),contract=CONTRACT)
    if any(meta.get(k)!=v for k,v in expected.items()):raise RuntimeError('sidecar binding')
    actual=file_hash(p)
    if meta['artifact_sha256']!=actual or actual==r['source'].get('source_batch_sha256'):
        raise RuntimeError('artifact hash/old batch alias')
    if p.resolve()==path(r['source']['source_batch_path']).resolve():raise RuntimeError('full pickle denied')
    permit(p,actual)
    record=copy.deepcopy(r);record['artifacts']['observation']=dict(path=str(p.resolve()),sha256=actual)
    provider=FileProviders();payload=provider.read(record,'observation')
    validate_observation(payload,r)
    if content_hash(payload)!=meta['content_sha256']:raise RuntimeError('payload content hash')
    return dict(window_id=r['window_id'],role=r['role'],source_sha256=r['source']['source_sha256'],
        observation=dict(path=str(p.resolve()),sha256=actual),sidecar=dict(path=str(side.resolve()),sha256=file_hash(side)),
        content_sha256=meta['content_sha256'],bytes=p.stat().st_size,sidecar_bytes=side.stat().st_size,
        agent_count=len(r['source']['agent_ids']),scene=r['source']['scene_family'],
        shape=list(payload['input_traj_maps'].shape),out_of_raster_obs_slots=meta['out_of_raster_obs_slots'],
        qualification='OBSERVATION_ARTIFACT_PREPARED_PROVENANCE_PENDING')


def save_artifact(payload,r,root,gate,binding,permit,outside=0):
    gate.authorize(r,'artifact_preparation')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    p=root/(r['window_id']+'.pt');side=p.with_suffix('.json')
    if p.exists() or side.exists():return verify_artifact(r,root,gate,binding,permit)
    validate_observation(payload,r)
    fd,tmp=tempfile.mkstemp(prefix='.observation-',dir=root);os.close(fd)
    try:
        with open(tmp,'wb') as f:
            torch.save(dict(window_id=r['window_id'],kind='observation',
                source_identity=digest(r['source']),payload=payload),f)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
        meta=sealed(dict(window_id=r['window_id'],source_identity=digest(r['source']),
            raw_sha256=r['source']['source_sha256'],grant_hash=gate.grant['manifest_hash'],
            asset_binding_hash=digest(binding),producer_hash=producer_hash(),contract=CONTRACT,
            artifact_sha256=file_hash(p),content_sha256=content_hash(payload),obs_steps=8,
            out_of_raster_obs_slots=outside,units='native pixel; source world metres per registered scene',
            qualification='OBSERVATION_ARTIFACT_PREPARED_PROVENANCE_PENDING'))
        atomic_json(meta,side)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return verify_artifact(r,root,gate,binding,permit)


class Budget:
    def __init__(self,root,seconds=5400,max_bytes=32*1024**3,rss_bytes=8*1024**3,clock=time.time):
        self.root=Path(root);self.seconds=seconds;self.max_bytes=max_bytes;self.rss_bytes=rss_bytes;self.clock=clock
        p=self.root/'budget.json'
        if p.exists():self.state=json.loads(p.read_text())
        else:self.state=dict(start=clock(),seconds=seconds,max_bytes=max_bytes,rss_bytes=rss_bytes)
        if any(self.state[k]!=v for k,v in [('seconds',seconds),('max_bytes',max_bytes),('rss_bytes',rss_bytes)]):
            raise RuntimeError('budget must not reset on resume')
        self.root.mkdir(parents=True,exist_ok=True);atomic_json(self.state,p)
        self.bytes=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
    def check(self,extra=0):
        if self.clock()-self.state['start']>=self.seconds:raise RuntimeError('BUDGET_WALL_LIMIT')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>self.rss_bytes:raise RuntimeError('BUDGET_RSS_LIMIT')
        if self.bytes+extra>self.max_bytes:raise RuntimeError('BUDGET_STORAGE_LIMIT')
        if shutil.disk_usage(self.root).free<extra+1024**3:raise RuntimeError('BUDGET_DISK_FREE')
    def report(self):
        return dict(wall_seconds=self.clock()-self.state['start'],bytes=self.bytes,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limits=self.state)


def estimate_record(r,assets):
    w,h=assets['RGB.jpg']['size'];n=len(r['source']['agent_ids'])
    return 4*(h//8)*(w//8)*(8*n+9)+8*8*n*4+24*n+32768
