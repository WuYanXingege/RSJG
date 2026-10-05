"""CPU-only P2 source audit and blocked plan generator; never a launcher.

No production/model imports. All checkpoint loads are explicit, once per path,
CPU-only, after source protection. Do not pass these templates to main.py.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
P1 = 'outputs/joint_dependency_v2/eth/joint_dependency_v2/clean_split_protocol/manifest.json'
P0 = 'outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v2a/adoption_protocol/split_audit.json'
CACHE = 'outputs/joint_dependency_v2/cache/eth_full_stage_a'
HOTEL = '9caa771bb9153d6b809dd0916b6f86761b641e6bbb15e766c1de3133fbbb7fcf'
INNER = '61f432c0ab3070ed0ef150fbeabcd7baf839cab5495a46e6105bd747f0a092a7'
BASE_ETH = '126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950'
BASE_UNIV = 'ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e'
RUNS = 'outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/'
CANDIDATES = [
    ('gdts_eth','../GDTS/output/eth/saved_models/best_model.pt',BASE_ETH,'eth'),
    ('gdts_univ','../GDTS/output/univ/saved_models/best_model.pt',BASE_UNIV,'univ'),
    ('canonical_a',RUNS+'jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt','699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb','eth'),
    ('p1_clean_a_best',RUNS+'jdv2_stage_a_clean_eth_seed2035/saved_models/best_model.pt',None,'p1'),
    ('p1_clean_a_last',RUNS+'jdv2_stage_a_clean_eth_seed2035/saved_models/last_model.pt',None,'p1'),
]
ALLOWED_DIFF = {'jdv2_goal_objective','jdv2_mc_backend','jdv2_neighbor_seed','run_name'}


def stable(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def role(sha, aliases=()):
    names={Path(p).name for p in aliases}
    if sha==HOTEL or 'biwi_hotel.txt' in names:return 'outer'
    if sha==INNER or 'uni_examples.txt' in names:return 'inner_valid'
    return 'train'


def assert_source_isolation(records):
    roles=defaultdict(set)
    for r in records:roles[r['source_sha256']].add(r['logical_role'])
    if any(len(v)>1 for v in roles.values()):raise ValueError('content alias crosses roles')


def overlap(records):
    tokens=defaultdict(Counter);unknown=0
    for r in records:
        if r.get('frame_ids') is None or r.get('agent_ids') is None:
            unknown+=1;continue
        for a in r['agent_ids']:
            for f in r['frame_ids']:tokens[r['logical_role']][(r['source_sha256'],a,f)]+=1
    roles=sorted(tokens)
    return dict(unknown_windows=unknown,
        windows_sharing_agent_time={role:sum(any(tokens[role][(r['source_sha256'],a,f)]>1 for a in r['agent_ids'] for f in r['frame_ids']) for r in records if r['logical_role']==role and r.get('agent_ids') is not None and r.get('frame_ids') is not None) for role in roles},
        repeated_agent_time={k:sum(v-1 for v in c.values()) for k,c in tokens.items()},
        unique_agent_time={k:len(c) for k,c in tokens.items()},
        cross_role={a+'__'+b:len(tokens[a].keys() & tokens[b].keys()) for i,a in enumerate(roles) for b in roles[i+1:]})


def classify(gradient_roles, selection_roles, positive_complete=False):
    if set(gradient_roles)&{'outer','inner_valid'} or 'outer' in selection_roles:
        return 'KNOWN_PROTOCOL_VIOLATION'
    return 'QUALIFIED' if positive_complete else 'PROVENANCE_UNVERIFIED'


def recording_overlap(records):
    """Identifier collisions are warnings, NOT proof of same trajectory."""
    tokens=defaultdict(set);sources={}
    for r in records:
        sha=r['source_sha256'];sources[sha]=r
        tokens[sha].update((a,f) for a in r['agent_ids'] for f in r['frame_ids'])
    return [dict(source_a=sources[a]['original_source_path'],source_b=sources[b]['original_source_path'],
        role_a=sources[a]['logical_role'],role_b=sources[b]['logical_role'],scene=sources[a]['scene_family'],
        shared_agent_frame_identifiers=len(tokens[a]&tokens[b]),
        identity_status='UNKNOWN: cross-file agent/frame namespace and recording identity not proven')
        for i,a in enumerate(tokens) for b in list(tokens)[i+1:] if sources[a]['scene_family']==sources[b]['scene_family']]


def cache_permission(record, teacher, purpose):
    if teacher and (record['logical_role']!='train' or purpose!='training'):
        raise PermissionError('logical role prohibits teacher, irrespective of physical split')
    # This audit reads no protected serialized records, even deployment ones.
    if record['logical_role']!='train':raise PermissionError('protected payload: metadata/hash only')


def compare_configs(a,b):
    diff={k:{'A0':a.get(k),'A1':b.get(k)} for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    if set(diff)-ALLOWED_DIFF:raise ValueError('non-objective config difference')
    return dict(allowed_fields=sorted(ALLOWED_DIFF),differences=diff,passed=True)


def validate_plan(plan):
    required=('base_path','base_sha256','cache_path','cache_manifest_sha256','initial_state_path','initial_state_sha256')
    if plan.get('status')!='READY_CLEAN_OBJECTIVE_PILOT_PLAN_NO_EXECUTION':
        raise RuntimeError('TEMPLATE_BLOCKED: no launch permitted')
    if any(plan.get(k) is None for k in required):raise RuntimeError('missing provenance')
    if plan.get('base_classification')!='QUALIFIED':raise RuntimeError('unqualified base')
    if plan['base_sha256']!=plan.get('cache_producer_sha256'):raise RuntimeError('cache producer mismatch')
    if not plan.get('loader_supported') or not plan.get('step_cap_supported'):
        raise RuntimeError('P2 loader/step cap not implemented')
    return True


class Reader:
    def __init__(self):
        self.counts=Counter();self.hashes={};self.reads=[];self.loaded=set()
    def hash(self,path):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        key=str(path)
        if key not in self.hashes:
            h=hashlib.sha256();size=0
            with path.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b);size+=len(b)
            self.hashes[key]={'sha256':h.hexdigest(),'bytes':size}
            self.counts['file_hash_operations']+=1;self.counts['file_hash_bytes']+=size
        return self.hashes[key]['sha256']
    def json(self,path):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        self.counts['json_metadata_reads']+=1;self.reads.append(str(path));self.hash(path)
        return json.loads(path.read_text())
    def text(self,path):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        self.counts['text_reads']+=1;self.reads.append(str(path));self.hash(path)
        return path.read_text()


def dump(out,name,value):
    p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def audit(out):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('require CUDA_VISIBLE_DEVICES empty')
    out=out.resolve()
    if out.exists():raise FileExistsError('new output directory required; preserve previous attempts')
    out.mkdir(parents=True)
    started=time.monotonic();reader=Reader();counts=reader.counts
    def stop(*_):raise TimeoutError('CPU audit 3600-second budget exhausted')
    signal.signal(signal.SIGALRM,stop);signal.alarm(3600)
    status='INCOMPLETE_CPU_BUDGET'
    try:
        p1=reader.json(P1);p0=reader.json(P0)
        assert reader.hash(P1)=='c8136967954bac5477b422da5d7f5b05fa979856a1318c4b323836ffff9f57ad'
        assert reader.hash(P0)==p1['source_audit_sha256']
        # Establish protection before ANY serialized artifact read.
        protection={'outer':{'content_sha256':HOTEL,'name_family':'biwi_hotel.txt'},
            'inner_valid':{'content_sha256':INNER,'name_family':'uni_examples.txt'},
            'policy':'no protected future/cache deserialization; existing metadata + whole-file hash only'}
        dump(out,'PROTECTED_SOURCES.json',protection)
        aliases=defaultdict(list)
        for fold in ('eth','hotel','univ','zara1','zara2'):
            for split in ('train','val','test'):
                for path in sorted((ROOT/'data/eth5'/fold/split).glob('*.txt')):
                    aliases[reader.hash(path)].append(str(path.relative_to(ROOT)))
        source_rows={}
        records=[]
        for physical,section in p0['splits'].items():
            for idx,r in enumerate(section['records']):
                sha=r['source_sha256']; logical=role(sha,aliases[sha])
                assert reader.hash(r['source_path'])==sha
                source_rows.setdefault(sha,dict(source_id='sha256:'+sha,sha256=sha,aliases=aliases[sha],
                    scene_family=r['scene_name'],logical_role=logical,physical_counts=Counter()))['physical_counts'][physical]+=1
                records.append(dict(source_id='sha256:'+sha,source_sha256=sha,logical_role=logical,
                    original_source_path=r['source_path'],scene_family=r['scene_name'],physical_split=physical,
                    physical_index=idx,cache_filename=r['cache_id'],source_batch_path=r['cache_path'],
                    source_batch_sha256=r['cache_sha256'],candidate_path=f'{CACHE}/{physical}/{idx:06d}.pt',
                    frame_ids=r['frame_ids'],agent_ids=r['agent_ids'],window_steps=len(r['frame_ids']),
                    id_sha256=r['exact_id_sha256'],content_sha256=r['content_sha256']))
        assert_source_isolation(records)
        # P0 valid/test duplicate: deterministic physical test representative only.
        records.sort(key=lambda r:({'train':0,'test':1,'valid':2}[r['physical_split']],r['physical_index']))
        unique={};duplicates=[]
        for r in records:
            key=(r['id_sha256'],r['content_sha256'])
            if key in unique:
                duplicates.append({'physical_split':r['physical_split'],'physical_index':r['physical_index'],
                    'representative_split':unique[key]['physical_split'],'representative_index':unique[key]['physical_index'],
                    'id_sha256':r['id_sha256'],'content_sha256':r['content_sha256']})
            else:unique[key]=r
        selected=list(unique.values());assert len(selected)==4249 and len(duplicates)==139
        for r in selected:
            assert len(r['frame_ids'])==20 and len(r['agent_ids'])>0
            r['frame_range']=[min(r['frame_ids']),max(r['frame_ids'])]
        logical_counts=dict(Counter(r['logical_role'] for r in selected))
        files=[]
        for i in range(0,len(selected),400):
            path=f'windows/part_{i//400:02d}.jsonl'
            (out/'windows').mkdir(exist_ok=True)
            (out/path).write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in selected[i:i+400]))
            files.append(dict(path=path,records=len(selected[i:i+400]),sha256=hashlib.sha256((out/path).read_bytes()).hexdigest()))
        order=sorted([r for r in selected if r['logical_role']=='train'],key=lambda r:(r['source_id'],r['frame_ids'][0],r['physical_split'],r['physical_index']))
        # A fixed cryptographic permutation avoids coupling to global/worker RNG.
        order=sorted(order,key=lambda r:stable(['rsjg.p2.batch-order.v1',3101,r['source_id'],r['id_sha256']]))
        batch_order=[dict(source_id=r['source_id'],id_sha256=r['id_sha256'],physical_split=r['physical_split'],physical_index=r['physical_index'],agents=len(r['agent_ids'])) for r in order]
        dump(out,'BATCH_ORDER_SEED3101.json',dict(seed=3101,namespace='rsjg.p2.batch-order.v1',
            rule='sort SHA256(compact sorted JSON [namespace,seed,source_id,id_sha256]); same stored permutation both arms',
            order_sha256=stable(batch_order),updates_per_native_epoch=len(order),H=min(500,len(order)),
            first_H_agent_exposures=sum(r['agents'] for r in batch_order[:500]),records=batch_order))
        split=dict(schema='rsjg-p2-audit-view-v1-NOT-production-manifest',protocol='P2 HOTEL outer / uni_examples inner',
            counts=logical_counts,physical_counts=dict(Counter(r['physical_split'] for r in records)),
            distinct_content_sources=len(source_rows),independent_recordings='UNKNOWN',sources=list(source_rows.values()),record_files=files,
            duplicate_physical_windows=duplicates,overlap=overlap(selected),
            old_eth139='P0 development-exposed ETH source is eligible P2 TRAIN (remaining non-protected source), included once via physical test addressing; never a new final test',
            membership_hash=stable([(r['source_id'],r['id_sha256'],r['physical_split'],r['physical_index'],r['logical_role']) for r in selected]),
            historical_metadata_authenticated=True,new_source_batch_deserializations=0,
            loader_support=False,unsupported_reasons=['hardcoded P1 counts/indices','source_block only excludes last train source','final role only physical test','no P2 outer lock or prospective obs-only source reader'])
        dump(out,'SPLIT_ROLE_MANIFEST.json',split)
        # Native metadata inspection is guarded: no model/optimizer constructors,
        # CUDA discovery/init/sync, arbitrary torch.load, or protected cache reads.
        import torch
        import yaml
        def forbidden(name):
            def fn(*a,**kw):counts[name]+=1;raise RuntimeError('forbidden '+name)
            return fn
        for name in ('_lazy_init','init','is_available','device_count','synchronize','get_device_name','get_device_properties'):
            setattr(torch.cuda,name,forbidden('cuda_forbidden_attempts'))
        torch.nn.Module.__init__=forbidden('real_model_construction_attempts')
        torch.nn.Module._call_impl=forbidden('real_model_call_attempts')
        original_load=torch.load
        torch.load=forbidden('unauthorized_torch_load_attempts')
        from src.joint_dependency_v2_cache import FUTURE_SUPERVISION_KEYS,FORBIDDEN_LEARNED_CACHE_KEYS
        # Installed optimizer signature: inspect only; never instantiate.
        opt_defaults={k:v.default for k,v in inspect.signature(torch.optim.Adam.__init__).parameters.items()
            if k in ('betas','eps','weight_decay','amsgrad','maximize','foreach','capturable','differentiable','fused')}
        def tensor_sha(v):return hashlib.sha256(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        def summarize_state(state):
            groups=defaultdict(dict);shapes={}
            for k,v in state.items():
                if not torch.is_tensor(v):continue
                assert v.device.type=='cpu'
                shapes[k]=dict(shape=list(v.shape),dtype=str(v.dtype),numel=v.numel(),sha256=tensor_sha(v))
                group=k.split('.')[0];groups[group][k]=shapes[k]
            return dict(keys=shapes,groups={k:dict(tensors=len(v),state_elements=sum(x['numel'] for x in v.values()),
                sha256=stable(v)) for k,v in groups.items()})
        cp_rows=[];states={}
        for name,path,pin,protocol in CANDIDATES:
            resolved=(ROOT/path).resolve()
            if not resolved.exists():
                cp_rows.append(dict(id=name,path=path,status='MISSING'));continue
            assert resolved not in reader.loaded and len(reader.loaded)<20
            actual=reader.hash(path)
            if pin:assert actual==pin,(name,'hash mismatch')
            counts['checkpoint_deserialization_attempts']+=1;reader.loaded.add(resolved)
            cp=original_load(resolved,map_location='cpu',weights_only=False)
            counts['checkpoint_deserializations']+=1
            st=cp.get('model_state_dict',cp);summary=summarize_state(st);states[name]=summary
            metadata={k:v for k,v in cp.items() if k not in ('model_state_dict','optimizer_state_dict','scheduler_state_dict') and isinstance(v,(str,int,float,bool,type(None)))}
            for k in ('architecture_config','ablation_config','goal_pretrain_initialization'):
                if k in cp:metadata[k]=cp[k]
            config_path=('../GDTS/output/'+protocol+'/config.yaml') if protocol in ('eth','univ') and name.startswith('gdts_') else str(Path(path).parents[1]/'config.yaml')
            config=yaml.safe_load(reader.text(config_path))
            assert config['test_set']==('eth' if protocol=='p1' else protocol)
            train_aliases=[p for ps in aliases.values() for p in ps if '/'+('eth' if protocol=='p1' else protocol)+'/train/' in p]
            grad_roles=sorted({role(sha,ps) for sha,ps in aliases.items() if any(p in train_aliases for p in ps)})
            if protocol=='p1':grad_roles=['train','outer']
            classification=classify(grad_roles,['inner_valid'] if protocol=='p1' else ['train'])
            # Metadata and logs link a run family, not a sample-perfect SGD trace.
            cp_rows.append(dict(id=name,path=path,sha256=actual,pin_matched=pin is not None,
                classification=classification,metadata=metadata,config_path=config_path,config_sha256=reader.hash(config_path),
                protocol=protocol,gradient_source_roles=grad_roles,selection_source_role='inner_valid' if protocol=='p1' else 'train',
                source_evidence='saved baseline training config + raw source content membership + completed loss curve; Stage-A also frozen parent equality',
                exact_historical_sgd_sample_trace='NOT_RECORDED',parent_manifest='UNKNOWN unless explicit metadata',
                state=summary))
            del cp,st
        equality={}
        for name in ('canonical_a','p1_clean_a_best','p1_clean_a_last'):
            equality[name]={k:states[name]['groups'].get(k,{}).get('sha256')==states['gdts_eth']['groups'][k]['sha256'] for k in ('goal_module','registrar','diffnet')}
        # Read one safe train deployment and its teacher; never protected roles.
        example=next(r for r in selected if r['logical_role']=='train' and r['physical_split']=='train')
        cache_records={}
        for teacher in (False,True):
            cache_permission(example,teacher,'training')
            path=example['candidate_path'];path=path.replace('.pt','.teacher.pt') if teacher else path
            counts['teacher_cache_reads' if teacher else 'deployment_cache_reads']+=1
            rec=original_load(ROOT/path,map_location='cpu',weights_only=False)
            future=sorted(set(rec)&FUTURE_SUPERVISION_KEYS);learned=sorted(set(rec)&FORBIDDEN_LEARNED_CACHE_KEYS)
            assert not learned and (teacher or not future)
            cache_records['teacher' if teacher else 'deployment']=dict(path=path,sha256=reader.hash(path),
                logical_role='train',keys=sorted(rec),future_keys=future,learned_keys=learned,
                tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in rec.items() if torch.is_tensor(v)})
            del rec
        manifests={k:reader.json(v) for k,v in [('eth',CACHE+'/manifest.json'),('univ','outputs/joint_dependency_v2/cache/univ_full_stage_a/manifest.json')]}
        cache=dict(existing_manifests=manifests,safe_train_record_inspection=cache_records,
            producer_hash_matches_existing_file=manifests['eth']['goal_checkpoint_hash']==BASE_ETH,
            p2_compatible=False,reason='both producer bases trained on protected HOTEL/inner labels; no relabel/reuse for fresh P2 base',
            old_builder_all_splits_teacher=True,old_builder_protected_future_access=True,
            necessary_patch='build deployment from observation+map only on protected roles; teacher generation and loading ONLY logical train',
            frozen_base_component_copy_exact=equality,new_candidate_cache=None,new_candidate_hash=None,
            required_generation_counts={'deployment_unique_windows':len(selected),'train_teacher_sidecars':logical_counts['train'],'protected_teacher_sidecars':0})
        dump(out,'CACHE_COMPATIBILITY.json',cache)
        dump(out,'BASE_PROVENANCE.json',dict(candidates=cp_rows,frozen_component_equality=equality,
            counts=dict(Counter(r.get('classification',r.get('status')) for r in cp_rows)),
            qualified_prediction_bases=0,inventory_scope='five selected artifacts; ETH epoch090/100 same disallowed run family not deserialized; no SDD or full-disk scan',
            components={'goal_history_diffusion':'per-component actual tensor-byte hashes above; baseline joint training includes all three',
            'map_semantic':'existing pred_mask.png one-hot + homography; external segmentation training/fit lineage UNKNOWN; no per-dataset learned normalization found in inspected preprocessing',
            'stage_a':'fresh state NOT_CREATED_AUTH_REQUIRED; historical canonical and P1 heads trained on HOTEL',
            'candidate':'producer SHA pins identity only, not source eligibility','teacher':'safe train sidecar inspected; protected future tensors not read'}))
        # AST-only defaults; do not call parser device resolution or main.
        defaults={}
        parser_source=reader.text('src/parser.py')
        for n in ast.walk(ast.parse(parser_source)):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args:
                try:key=ast.literal_eval(n.args[0]).lstrip('-').replace('-','_')
                except (ValueError,TypeError):continue
                for kw in n.keywords:
                    if kw.arg=='default':
                        try:defaults[key]=ast.literal_eval(kw.value)
                        except (ValueError,TypeError):pass
        canonical=yaml.safe_load(reader.text('configs/joint_dependency_v2/jdv2_stage_a_clean_eth.yaml'))
        config={**defaults,**canonical}
        config.update(seed=3101,phase='train',run_name='p2_l1_a0_seed3101_BLOCKED',num_epochs=1,
            jdv2_goal_objective='mean_energy',jdv2_mc_backend='cpu_v1',jdv2_neighbor_seed=None,
            jdv2_source_checkpoint=None,jdv2_source_checkpoint_hash=None,pretrain_path=None,
            jdv2_cache_root=None,jdv2_cache_manifest_hash=None,jdv2_cache_source_commit=None,
            clean_split_manifest_path=None,clean_split_manifest_hash=None,load_checkpoint=None,
            shuffle_train_batches=False,jdv2_stage_progress=0.,model_selection_split='internal_train',
            final_test_access='blocked')
        configs={}
        for arm in ('A0','A1'):
            cfg=dict(config)
            if arm=='A1':cfg.update(jdv2_goal_objective='expected_conditional_mc',jdv2_mc_backend='cuda_fp32_bf16_v1',
                jdv2_neighbor_seed=int.from_bytes(hashlib.sha256(b'rsjg.p2.mc-neighbor.v1:3101').digest()[:8],'big')%(2**63),run_name='p2_l1_a1_seed3101_BLOCKED')
            configs[arm]=cfg
            (out/(arm+'_BLOCKED.yaml')).write_text('# TEMPLATE_BLOCKED: audit template, NOT accepted production P2 config; use --check-plan.\n'+yaml.safe_dump(cfg,sort_keys=True))
        dump(out,'PAIRED_CONFIG_DIFF.json',compare_configs(configs['A0'],configs['A1']))
        dump(out,'RESOLVED_CONFIGS.json',dict(status='STATIC_DEFAULT_MERGE_NOT_PRODUCTION_RESOLVED',
            parser='AST literal defaults + approved P1 scientific fields + P2 blocked override; NO check_and_add_additional_args call',configs=configs))
        L=len(order);H=min(500,L)
        plan=dict(status='TEMPLATE_BLOCKED',base_path=None,base_sha256=None,cache_path=None,cache_manifest_sha256=None,
            cache_producer_sha256=None,initial_state_path=None,initial_state_sha256=None,base_classification='MISSING_QUALIFIED',
            loader_supported=False,step_cap_supported=False,seed=3101,seeds_L2=[3101,3102,3103],
            H=H,native_epoch_updates=L,accumulation=1,reference_num_epochs=1,total_steps=L,
            progress='before attempt t (1-based), completed=(t-1), progress=(t-1)/3484; stop on any skip/nonfinite/pending error',
            beta_first=0.,beta_last=.1*min(((H-1)/L)/.2,1.),post_pilot_progress=H/L,
            scheduler='ExponentialLR gamma=0.995 at complete native epoch only; 0 scheduler steps in partial L1',
            optimizer=dict(name='Adam',learning_rate=1e-4,clip=1.,installed_defaults=opt_defaults),
            caps=dict(per_arm_gpu_seconds=900,total_gpu_seconds=2700,peak_reserved_bytes=10*1024**3),
            execution_authorized=False)
        dump(out,'PILOT_PLAN.json',plan)
        dump(out,'L1_PROGRESS_TABLE.json',dict(rows=[dict(attempt=t+1,completed_before=t,progress=t/L,beta=.1*min((t/L)/.2,1.),post=.5,prior=.5,lr=1e-4) for t in range(H)]))
        with (out/'SOURCE_EXPOSURE_MATRIX.csv').open('w',newline='') as f:
            fields=['source_id','aliases','p2_role','historical_read','baseline_eth_gradient','baseline_univ_gradient','canonical_A_gradient','P1_A_gradient','checkpoint_selection','method_development','prospective_new_gradient']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for sha,r in source_rows.items():
                names=r['aliases'];inner=sha==INNER;outer=sha==HOTEL
                writer.writerow(dict(source_id=r['source_id'],aliases=';'.join(names),p2_role=r['logical_role'],historical_read='YES',
                    baseline_eth_gradient='YES' if any('/eth/train/' in p for p in names) else 'NO_CONFIGURED',
                    baseline_univ_gradient='YES' if any('/univ/train/' in p for p in names) else 'NO_CONFIGURED',
                    canonical_A_gradient='YES' if any('/eth/train/' in p for p in names) else 'NO_CONFIGURED',
                    P1_A_gradient='NO_HEAD_GRADIENT_BUT_BASE_EXPOSED' if inner else ('YES' if any('/eth/train/' in p for p in names) else 'NO_CONFIGURED'),
                    checkpoint_selection='YES_P1_INNER' if inner else ('NO_DIRECT_SELECTION_FOUND' if outer else 'ETH_AND_UNIV_HELDOUT_SELECTION_WHERE_APPLICABLE'),
                    method_development='YES_HOTEL_R1_R4_TEACHER_LOSS_READ_BACKWARD_NO_UPDATE' if outer else 'YES_PRIOR_PROTOCOL_OR_METHOD_DATA',
                    prospective_new_gradient='TRAIN_ONLY' if r['logical_role']=='train' else 'NO'))
        # Preserve required evidence hashes and fully parse all required JSON.
        prior=Path('docs/joint_dependency_v2/reviews/2026-10-05_c9dbfec_cuda_amp_loss_smoke')
        for n in ('RESULTS.json','PREFLIGHT.json','SOURCE_SCOPE.json'):reader.json(prior/n)
        for n in ('REVIEW.md','DEVICE_AMP_CONTRACT.md'):reader.text(prior/n)
        theory=Path('docs/joint_dependency_v2/reviews/2026-10-04_79ea3fa_innovation_theory_architecture')
        for n in ('REVIEW.md','CURRENT_AND_PROPOSED_ARCHITECTURE.md','EXPERIMENT_PREREGISTRATION.md','IMPLEMENTATION_PSEUDOCODE.md'):reader.text(theory/n)
        for n in ('JDV2_EVALUATION_SPLIT_AUDIT.md','JDV2_CLEAN_SPLIT_REPRODUCTION_PLAN.md','JDV2_CLEAN_SPLIT_REPRODUCTION_PREFLIGHT.md'):reader.text('docs/joint_dependency_v2/'+n)
        reader.json(str(Path(P1).with_name('preflight_results.json')))
        status='AUDIT_COMPLETE'
        dump(out,'RESULTS.json',dict(audit_status=status,CLEAN_PILOT_READINESS='BLOCKED_BASE_PROTOCOL_VIOLATION',
            secondary_blocks=['BLOCKED_CLEAN_BASE_MISSING','BLOCKED_SPLIT_OR_CACHE_PROVENANCE','BLOCKED_PROTOCOL_LOADER_SUPPORT'],
            base_counts=dict(Counter(r.get('classification',r.get('status')) for r in cp_rows)),
            protocol_counts=logical_counts,H=H,new_initialization='NOT_CREATED_AUTH_REQUIRED',
            historical_exposure='HOTEL R1-R4 already read teacher/loss/backward; zero updates there does not mean untouched; earlier base/heads trained on HOTEL',
            no_execution=True,production_changes=0,open=['efficacy','novelty','pair causality','full CUDA resume','GPU observer neutrality']))
    finally:
        signal.alarm(0)
        for k in ('protected_future_tensor_reads','source_batch_deserializations','outer_metric_computations','real_model_construction_attempts','real_model_call_attempts','cuda_forbidden_attempts','optimizer_updates','data_training','data_evaluation','sampler_calls','diffusion_calls','production_code_changes'):
            counts.setdefault(k,0)
        dump(out,'EXECUTION.json',dict(status=status,elapsed_seconds=time.monotonic()-started,counts=counts,
            file_hashes=reader.hashes,metadata_reads=reader.reads,
            counter_scope='this isolated audit process; preliminary shell source/JSON inspection is separately documented, no checkpoint/cache deserialization before this process'))
    print(json.dumps(dict(status=status,output=str(out),counts=counts,seconds=time.monotonic()-started),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--check-plan',type=Path)
    a=p.parse_args()
    if a.check_plan:
        try:validate_plan(json.loads(a.check_plan.read_text()))
        except RuntimeError as e:print(str(e));return 2
        print('Plan metadata valid; this tool NEVER launches training');return 0
    if not a.output:p.error('--output NEW_DIRECTORY required for audit/plan-only')
    audit(a.output);return 0


if __name__=='__main__':raise SystemExit(main())
