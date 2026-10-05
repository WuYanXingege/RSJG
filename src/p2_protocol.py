"""P2 metadata-first contracts. No torch/model imports at module import time.

The archived membership is immutable. Artifact certification is a *new* manifest,
never a re-label of historical candidates. UNKNOWN fails before payload reads.
"""
from __future__ import annotations
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

SCHEMA = 'rsjg-p2-role-v1'
FAMILY = 'p2_hotel_uni_examples'
CACHE_SCHEMA = 'jdv2-p2-cache-v2'
ROOT = Path(__file__).resolve().parents[1]
REGISTERED_ROWS = 'a3df43f81bd1c198d369b1607bf3e02ed7104c9197a094e9d93fcf6a0254e4c1'
HOTEL = '9caa771bb9153d6b809dd0916b6f86761b641e6bbb15e766c1de3133fbbb7fcf'
INNER = '61f432c0ab3070ed0ef150fbeabcd7baf839cab5495a46e6105bd747f0a092a7'
BAD_BASES = {
 '126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950',
 'ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e',
 '699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb',
 '95f32cc2ae636524001b0f97f19c7d1e64cfc436f6526e6072216ccd73303765',
 '87ae976b99f44a2817acb58b82054d173c5784cb7dcaa5cc97190678c86d0650',
}
def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def path(p):
    p=Path(p)
    return p if p.is_absolute() else ROOT/p
def sealed(v):
    v=copy.deepcopy(v);v.pop('manifest_hash',None)
    return dict(v,manifest_hash=digest(v))
def enabled(args):return getattr(args,'p2_mode','off')!='off'
def bounded(args):return enabled(args) and args.p2_mode=='l1'
def identity(source):
    return digest({k:source[k] for k in ('source_sha256','id_sha256','source_batch_path','physical_split','physical_index')})
def expected_role(s):
    return 'outer' if s==HOTEL else 'inner_valid' if s==INNER else 'train'

def data_binding(registry):
    return digest([dict(source=r['source'],recording=r['recording'],map=r['map_provenance'])
                   for r in registry.records])

class Registry:
    def __init__(self,manifest,expected_hash,*,registered_rows=REGISTERED_ROWS):
        m=copy.deepcopy(manifest)
        if m.get('schema')!=SCHEMA or m.get('family')!=FAMILY:raise RuntimeError('P2 schema/family')
        if sealed(m)['manifest_hash']!=m.get('manifest_hash') or expected_hash!=m['manifest_hash']:
            raise RuntimeError('P2 manifest hash mismatch')
        self.manifest=m
        if 'records' in m:
            self.records=m['records']
        else:
            self.records=[]
            for shard in m['record_files']:
                p=path(shard['path'])
                if file_hash(p)!=shard['sha256']:raise RuntimeError('P2 runtime shard hash')
                self.records.extend(json.loads(line) for line in p.read_text().splitlines())
        self.by_id={}
        if digest([r['source'] for r in self.records])!=registered_rows:
            raise RuntimeError('P2 registered membership/address mismatch')
        aliases={};roles={};addresses=set();fingerprints=set()
        recordings={}
        for r in self.records:
            s=r['source'];role=r['role'];wid=r['window_id']
            if wid!=identity(s) or wid in self.by_id:raise RuntimeError('P2 duplicate/invalid window ID')
            if role!=s['logical_role'] or role!=expected_role(s['source_sha256']):raise RuntimeError('P2 source role')
            if Path(s['source_batch_path']).name!=s['cache_filename']:raise RuntimeError('P2 filename binding')
            if s['window_steps']!=20 or len(s['frame_ids'])!=20 or r['boundary']!=[8,12]:
                raise RuntimeError('P2 observation/future boundary')
            a=(s['source_batch_path'],s['physical_split'],s['physical_index'])
            if a in addresses:raise RuntimeError('P2 duplicate physical address')
            addresses.add(a)
            for key in ('id_sha256','content_sha256'):
                token=(key,s[key])
                if token in fingerprints:raise RuntimeError('P2 duplicate member')
                fingerprints.add(token)
            roles.setdefault(s['source_sha256'],set()).add(role)
            for alias in r['aliases']:
                if Path(alias).name=='biwi_hotel.txt' and s['source_sha256']!=HOTEL:raise RuntimeError('P2 protected alias')
                if Path(alias).name=='uni_examples.txt' and s['source_sha256']!=INNER:raise RuntimeError('P2 protected alias')
                if alias in aliases and aliases[alias]!=s['source_sha256']:raise RuntimeError('P2 alias mismatch')
                aliases[alias]=s['source_sha256']
            rec=r['recording']
            if rec.get('status')=='VERIFIED':
                if not rec.get('evidence_sha256') or not rec.get('recording_id') or not rec.get('id_namespace'):
                    raise RuntimeError('P2 recording evidence missing')
                recordings.setdefault(rec['recording_id'],set()).add(role)
            self.by_id[wid]=r
        if any(len(v)>1 for v in roles.values()):raise RuntimeError('P2 source crosses roles')
        if any(len(v)>1 for v in recordings.values()):raise RuntimeError('BLOCKED_REGISTERED_SPLIT_CONFLICT')
        self.counts=dict(Counter(r['role'] for r in self.records))
        if registered_rows==REGISTERED_ROWS and self.counts!={'train':3484,'inner_valid':320,'outer':445}:
            raise RuntimeError('P2 counts')
        if set(m['train_order'])!=set(r['window_id'] for r in self.records if r['role']=='train') or len(m['train_order'])!=self.counts.get('train',0):
            raise RuntimeError('P2 train order membership')
        if digest(m['train_order'])!=m['train_order_hash']:raise RuntimeError('P2 order hash')
        if registered_rows==REGISTERED_ROWS:
            ordered=sorted(self.refs_unordered('train'),key=lambda r:digest([
                'rsjg.p2.batch-order.v1',3101,r['source']['source_id'],r['source']['id_sha256']]))
            if m['train_order']!=[r['window_id'] for r in ordered]:raise RuntimeError('P2 registered seed3101 order')
    def refs_unordered(self,role):
        return [r for r in self.records if r['role']==role]
    def require_provenance(self):
        if any(r['recording'].get('status')!='VERIFIED' for r in self.records):
            raise RuntimeError('BLOCKED_RECORDING_IDENTITY_UNVERIFIED')
        for r in self.records:
            sm=r['map_provenance']
            if sm.get('status') not in {'PROVIDED_STATIC_ASSET','LEARNED_PREPROCESSOR'} or not sm.get('evidence_sha256') or not sm.get('protocol_permitted'):
                raise RuntimeError('BLOCKED_MAP_PROVENANCE')
            if sm['status']=='LEARNED_PREPROCESSOR' and sm.get('training_roles')!=['train']:
                raise RuntimeError('BLOCKED_MAP_TRAINING_PROVENANCE')
    def authorize(self,wid,purpose):
        self.require_provenance()
        r=self.by_id[wid];role=r['role']
        policies={
          ('train','gradient'):(True,False),
          ('train','jdv2_gradient'):(True,True),
          ('train','teacher_build_train'):(True,False),
          ('inner_valid','selection'):(True,False),
        }
        if purpose=='candidate_build':target,teacher=False,False
        elif (role,purpose) in policies:target,teacher=policies[(role,purpose)]
        else:raise PermissionError('P2 role/purpose denied before payload: '+role+'/'+purpose)
        return dict(role=role,purpose=purpose,allow_target=target,allow_teacher=teacher,
                    augmentation=role=='train' and purpose in {'gradient','jdv2_gradient'})
    def refs(self,role):
        return ([self.by_id[x] for x in self.manifest['train_order']] if role=='train'
                else [r for r in self.records if r['role']==role])

def load_registry(args):
    p=getattr(args,'p2_manifest',None)
    if not p:raise RuntimeError('MISSING_P2_MANIFEST')
    return Registry(json.loads(path(p).read_text()),getattr(args,'p2_manifest_hash',None))

class FileProviders:
    """Only typed independent artifacts; never fall back to full source pickle."""
    def __init__(self):self.counts=Counter()
    def read(self,r,kind):
        ref=r['artifacts'].get(kind)
        if not ref or not ref.get('path') or not ref.get('sha256'):
            raise RuntimeError('MISSING_OBSERVATION_ONLY_ARTIFACT' if kind=='observation' else 'MISSING_P2_'+kind.upper())
        p=path(ref['path'])
        if p.resolve()==path(r['source']['source_batch_path']).resolve():
            raise RuntimeError('P2 full physical batch is not an independent artifact')
        if ref['sha256']==r['source'].get('source_batch_sha256'):
            raise RuntimeError('P2 full batch bytes cannot be relabelled observation-only')
        if not p.is_file() or file_hash(p)!=ref['sha256']:raise RuntimeError('P2 artifact hash binding')
        import torch
        self.counts[kind]+=1
        blob=torch.load(p,map_location='cpu',weights_only=True)
        if blob.get('window_id')!=r['window_id'] or blob.get('kind')!=kind or blob.get('source_identity')!=digest(r['source']):
            raise RuntimeError('P2 artifact identity mismatch')
        value=blob['payload']
        validate_payload(kind,value)
        if kind=='observation':
            import torch
            if value['abs_pixel_coord'].shape[1]!=len(r['source']['agent_ids']):
                raise RuntimeError('P2 observation agent count binding')
            expected=torch.tensor(r['source']['frame_ids'][:8])
            frames=value.get('frame_ids')
            if frames is None or not torch.equal(frames[:,0].cpu(),expected) or not frames.eq(frames[:,:1]).all():
                raise RuntimeError('P2 observation frame binding')
        return value

def validate_payload(kind,value):
    if not isinstance(value,dict):raise RuntimeError('P2 payload must be dict')
    allowed={
      'observation':{'abs_pixel_coord','seq_list','input_traj_maps','tensor_image','tensor_map','scene_index','scene_ptr','frame_ids','batch_format_version'},
      'target':{'abs_pixel_coord','seq_list','input_traj_maps','future_position_world'},
      'teacher':{'future_pair_descriptor','contains_future_supervision'},
      'deployment':{'cache_id','scene_index','scene_ptr','frame_ids','edge_index','edge_feat','edge_weight','goal_candidates_map','goal_candidates_world','candidate_log_prior','contains_future_supervision'},
    }
    if kind not in allowed or set(value)-allowed[kind]:raise RuntimeError('P2 unexpected payload fields: '+kind)
    if kind in {'observation','target'}:
        steps=8 if kind=='observation' else 12
        for name,dim in [('abs_pixel_coord',0),('seq_list',0),('input_traj_maps',1)]:
            if name not in value or value[name].shape[dim]!=steps:raise RuntimeError('P2 '+kind+' temporal boundary')
        if kind=='observation' and 'frame_ids' in value and value['frame_ids'].shape[0]!=8:raise RuntimeError('P2 observation frame boundary')
    if kind=='deployment' and value.get('contains_future_supervision',False):raise RuntimeError('P2 deployment future flag')
    if kind=='deployment':
        import torch
        for key in ('goal_candidates_map','goal_candidates_world'):
            v=value.get(key)
            if v is None or v.ndim!=3 or v.shape[1:]!=(21,2) or v.dtype!=torch.float32:
                raise RuntimeError('P2 deployment K/order/dtype')
        n=value['goal_candidates_world'].shape[0]
        prior=value.get('candidate_log_prior')
        if prior is None or prior.shape!=(n,21) or prior.dtype!=torch.float32:
            raise RuntimeError('P2 deployment prior')
        edge=value.get('edge_index')
        if edge is None or edge.dtype!=torch.int64 or edge.ndim!=2 or edge.shape[0]!=2:
            raise RuntimeError('P2 deployment graph')
    import torch
    for v in value.values():
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
            raise RuntimeError('P2 nonfinite payload')

def read_record(registry,wid,purpose,providers):
    policy=registry.authorize(wid,purpose)
    r=registry.by_id[wid]
    obs=providers.read(r,'observation')
    target=providers.read(r,'target') if policy['allow_target'] else None
    teacher=providers.read(r,'teacher') if policy['allow_teacher'] else None
    return dict(observation=obs,target=target,teacher=teacher,identity=wid,
                policy=policy,ref=r)

def combine_train(obs,target):
    """Only after gradient authorization; never the prediction/selection route."""
    import torch
    if target is None:raise RuntimeError('P2 training target missing')
    value=dict(obs)
    for k,dim in [('abs_pixel_coord',0),('seq_list',0),('input_traj_maps',1)]:
        value[k]=torch.cat((obs[k],target[k]),dim=dim)
    return value

def split_window_payload(batch,source):
    """Explicit offline separation primitive, no file reads or implicit migration.

    Caller must already have authorization for this full window. Never use a
    union of frames from neighboring windows to construct protected targets.
    """
    if batch['abs_pixel_coord'].shape[:2]!=(20,len(source['agent_ids'])):
        raise RuntimeError('P2 separation shape')
    import torch
    if 'frame_ids' not in batch or not torch.equal(batch['frame_ids'][:,0].cpu(),
            torch.tensor(source['frame_ids'])):raise RuntimeError('P2 separation frame identity')
    obs={};target={}
    for k,v in batch.items():
        if k in {'abs_pixel_coord','seq_list','frame_ids'}:
            obs[k]=v[:8].clone()
            if k!='frame_ids':target[k]=v[8:].clone()
        elif k=='input_traj_maps':obs[k]=v[:,:8].clone();target[k]=v[:,8:].clone()
        elif k in {'tensor_image','tensor_map','scene_index','scene_ptr','batch_format_version'}:obs[k]=v
        else:raise RuntimeError('P2 separation unknown field')
    validate_payload('observation',obs);validate_payload('target',target)
    return obs,target

def validate_parent(metadata,registry,role,actual_sha):
    if actual_sha in BAD_BASES:raise RuntimeError('KNOWN_PROTOCOL_VIOLATION')
    if metadata.get('artifact_role')!=role or metadata.get('sha256')!=actual_sha:
        raise RuntimeError('P2 parent role/hash')
    if metadata.get('data_binding')!=data_binding(registry) or metadata.get('gradient_roles')!=['train'] or metadata.get('selection_role')!='inner_valid':
        raise RuntimeError('P2 parent source/selection')
    if metadata.get('classification')!='QUALIFIED' or not metadata.get('initial_state_sha256'):
        raise RuntimeError('P2 parent provenance')
    return metadata

def base_metadata(args,registry,epoch,initial_hash,exposure,metric):
    import subprocess
    return dict(schema=SCHEMA,artifact_role='p2_goal' if args.p2_mode=='goal' else 'p2_base',
        manifest_hash=registry.manifest['manifest_hash'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        role_manifest_source_commit=registry.manifest['source_commit'],
        data_binding=data_binding(registry),
        recording_evidence_hash=digest([r['recording'] for r in registry.records]),
        map_evidence_hash=digest([r['map_provenance'] for r in registry.records]),
        initialization='fresh_goal' if args.p2_mode=='goal' else 'clean_goal_fresh_history_diffusion',
        seed=args.seed,initial_state_sha256=initial_hash,goal_parent_sha256=getattr(args,'goal_pretrain_checkpoint_sha256',None),
        gradient_roles=['train'],selection_role='inner_valid',selection_metric=metric,
        selection_epoch=epoch,tie='strict improvement; earlier epoch',exposure_hash=digest(exposure),
        artifact_scope='weights and epoch metadata; real CUDA exact resume NOT certified',resumable=False)

def launch_guard(args):
    """Every public entry calls this before RNG, device, net or checkpoint load."""
    if not enabled(args):return None
    if getattr(args,'clean_split_protocol',False):raise RuntimeError('P1 and P2 mutually exclusive')
    mode=args.p2_mode
    expected_phase={'goal':'goal_pretrain','joint':'train','cache':'build-jdv2-cache','l1':'train'}
    if mode not in expected_phase or args.phase!=expected_phase[mode]:raise RuntimeError('P2 phase/train-only dispatch')
    if getattr(args,'pretrain_path',None) or getattr(args,'load_checkpoint',None):raise RuntimeError('P2 fresh only; no full warm start/resume')
    reg=load_registry(args)
    reg.require_provenance()
    if getattr(args,'data_augmentation',False):raise RuntimeError('P2 augmentation not certified; explicitly disable')
    if getattr(args,'num_workers',0)!=0:raise RuntimeError('P2 requires num_workers=0')
    if mode in {'cache','l1'}:
        graph_contract={'num_goal_candidates':21,'graph_type':'radius_ttc',
                        'graph_radius':6.,'ttc_threshold':8.,'trajectory_dt':.4}
        if any(getattr(args,k,None)!=v for k,v in graph_contract.items()):
            raise RuntimeError('P2 registered candidate/graph contract')
    if mode in {'goal','joint'}:
        if args.goal_model_type!='independent' or args.training_stage!='baseline':raise RuntimeError('P2 base independent only')
        if args.num_epochs>({'goal':150,'joint':250}[mode]):raise RuntimeError('P2 base epoch cap')
        if getattr(args,'batch_size',64)!=64:raise RuntimeError('P2 native base packing=64')
        if getattr(args,'best_metric','auto')!='auto':raise RuntimeError('P2 base registered selection metric')
    if mode!='goal':
        ref=reg.manifest.get('parents',{}).get('goal' if mode=='joint' else 'base')
        if not ref or not ref.get('path'):raise RuntimeError('MISSING_CLEAN_GOAL_PARENT' if mode=='joint' else 'MISSING_CLEAN_BASE')
        actual=file_hash(path(ref['path']))
        validate_parent(ref,reg,'p2_goal' if mode=='joint' else 'p2_base',actual)
        expected=getattr(args,'goal_pretrain_checkpoint' if mode=='joint' else 'jdv2_source_checkpoint',None)
        if not expected or path(expected).resolve()!=path(ref['path']).resolve():raise RuntimeError('P2 configured parent path')
    for r in reg.records:
        if mode in {'goal','joint'} and r['role']=='outer':continue
        if mode=='l1' and r['role']!='train':continue
        kinds=['observation']
        if (mode in {'goal','joint'} and r['role']!='outer') or (mode in {'cache','l1'} and r['role']=='train'):kinds.append('target')
        if mode=='l1':kinds+=['deployment']+(['teacher'] if r['role']=='train' else [])
        for kind in kinds:
            ref=r['artifacts'].get(kind)
            if not ref or not ref.get('path') or not ref.get('sha256'):
                raise RuntimeError('MISSING_OBSERVATION_ONLY_ARTIFACT' if kind=='observation' else 'MISSING_P2_'+kind.upper())
            if path(ref['path']).resolve()==path(r['source']['source_batch_path']).resolve():
                raise RuntimeError('P2 independent artifacts cannot be full physical batches')
            if not path(ref['path']).is_file() or file_hash(path(ref['path']))!=ref['sha256']:
                raise RuntimeError('P2 artifact absent/hash mismatch before device')
    if mode=='l1':
        validate_bounds(args,len(reg.refs('train')))
        if not reg.manifest.get('shared_initial_state'):raise RuntimeError('NOT_CREATED_SHARED_INITIAL_STATE')
        if args.goal_model_type!='joint_dependency_v2' or args.training_stage!='joint_goal':
            raise RuntimeError('P2 L1 Stage-A only')
        required={'jdv2_latent_objective':'strict_no_z','use_scene_latent':False,
                  'num_goal_candidates':21,'num_samples':20,'jdv2_relation_modes':4,
                  'jdv2_energy_rank':8,'num_refinement_steps':2,'learning_rate':1e-4,
                  'optimizer':'Adam','scheduler':'ExponentialLR','clip':1.,
                  'use_trajectory_bank_cache':False,'shuffle_train_batches':False}
        if any(getattr(args,k,None)!=v for k,v in required.items()):raise RuntimeError('P2 L1 frozen settings')
        if args.seed!=3101 or args.num_epochs!=1 or getattr(args,'amp_dtype','bf16')=='fp16':
            raise RuntimeError('P2 L1 seed/epoch/AMP contract')
        state=reg.manifest['shared_initial_state']
        if state.get('base_sha256')!=reg.manifest['parents']['base']['sha256'] or state.get('data_binding')!=data_binding(reg):
            raise RuntimeError('P2 shared initial state provenance')
        if not state.get('sha256') or file_hash(path(state['path']))!=state['sha256']:
            raise RuntimeError('P2 initial state hash')
        cache=reg.manifest.get('cache',{})
        if cache.get('schema_version')!=CACHE_SCHEMA or cache.get('producer_sha256')!=state['base_sha256'] or cache.get('data_binding')!=data_binding(reg):
            raise RuntimeError('P2 candidate producer mismatch')
        pair=reg.manifest.get('pair_run')
        if not pair or not pair.get('failure_path') or not isinstance(pair.get('monotonic_start'),(int,float)):
            raise RuntimeError('MISSING_SHARED_PAIR_BUDGET_LEDGER')
        import math,time
        if not math.isfinite(pair['monotonic_start']) or not 0<=pair['monotonic_start']<=time.monotonic():
            raise RuntimeError('P2 invalid shared monotonic budget origin')
        if path(pair['failure_path']).exists():raise RuntimeError('PAIR_ALREADY_STOPPED')
    return reg

def validate_bounds(args,native):
    if not bounded(args):return
    if args.num_workers!=0 or args.coupling_grad_accum_steps!=1:raise RuntimeError('P2 cap requires no prefetch/accumulation=1')
    if not 0<args.p2_max_update_attempts<=min(500,native) or args.p2_reference_total_steps!=native:
        raise RuntimeError('P2 attempt/reference cap')
    if not args.p2_skip_validation or not args.p2_no_automatic_resume:raise RuntimeError('P2 bounded no validation/resume')
    if not 0<args.p2_arm_seconds<=900 or not 0<args.p2_stage_seconds<=2700 or not 0<args.p2_reserved_bytes<=10*1024**3:
        raise RuntimeError('P2 resource limits')

def dispatch(args,goal_factory,trainer_factory,cache_factory):
    launch_guard(args)
    from src.utils import set_seed
    if getattr(args,'reproducibility',False):
        set_seed(seed_value=args.seed,use_cuda=args.use_cuda)
    if args.p2_mode=='goal':return goal_factory(args).train()
    if args.p2_mode=='cache':return cache_factory(args).build()
    return trainer_factory(args).train()
