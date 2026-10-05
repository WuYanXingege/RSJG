"""Generate NEW P2 runtime metadata from immutable archived identities only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import yaml
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.p2_protocol import *
from src.p2_data import base_packing_plan
from src.p2_cache import task_plan
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
D=args.output.resolve();D.mkdir(parents=True,exist_ok=True)
if (D/'P2_RUNTIME_MANIFEST.json').exists():raise RuntimeError('new runtime output required')
P=R/'docs/joint_dependency_v2/reviews/2026-10-05_4f11f44_clean_base_split_objective_preflight'
reads=[]
def read(p):
    b=p.read_bytes();reads.append(dict(path=str(p.relative_to(R)),bytes=len(b),sha256=file_hash(p)))
    return json.loads(b)
old=read(P/'SPLIT_ROLE_MANIFEST.json');rows=[]
for f in old['record_files']:
    p=P/f['path'];assert file_hash(p)==f['sha256']
    b=p.read_bytes();reads.append(dict(path=str(p.relative_to(R)),bytes=len(b),sha256=file_hash(p)))
    rows.extend(json.loads(s) for s in b.splitlines())
assert digest(rows)==REGISTERED_ROWS
sources={s['sha256']:s for s in old['sources']}
records=[]
for s in rows:
    records.append(dict(window_id=identity(s),source=s,role=s['logical_role'],
        aliases=sources[s['source_sha256']]['aliases'],boundary=[8,12],
        recording=dict(status='UNKNOWN',recording_id=None,id_namespace=None,evidence_sha256=None),
        map_provenance=dict(status='UNKNOWN',asset_kind='static semantic raster; generating model lineage unknown',
                            protocol_permitted=False,evidence_sha256=None),
        permissions=dict(gradient=s['logical_role']=='train',teacher=s['logical_role']=='train',
                         selection=s['logical_role']=='inner_valid',outer_metrics=False),
        artifacts={k:None for k in ('observation','target','teacher','deployment')}))
prior=read(P/'BATCH_ORDER_SEED3101.json')
lookup={(r['source']['source_id'],r['source']['id_sha256']):r['window_id'] for r in records}
order=[lookup[(r['source_id'],r['id_sha256'])] for r in prior['records']]
manifest=sealed(dict(schema=SCHEMA,family=FAMILY,source_commit='28728bf96a290ccd83c5c637769deb987f477b07',
    records=records,train_order=order,train_order_hash=digest(order),down_factor=8,
    parents=dict(goal=None,base=None),shared_initial_state=None,cache=None,pair_run=None,
    status='BLOCKED_RECORDING_IDENTITY_UNVERIFIED'))
reg=Registry(manifest,manifest['manifest_hash'])
def write(name,value):
    p=D/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
# Runtime shard addresses/hash, no duplicate large monolithic JSON.
parts=[]
for start in range(0,len(records),300):
    p=D/('records/part_%02d.jsonl'%(start//300));p.parent.mkdir(exist_ok=True)
    p.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in records[start:start+300]))
    parts.append(dict(path=str(p.relative_to(R)),sha256=file_hash(p),count=len(records[start:start+300])))
manifest.pop('records');manifest['record_files']=parts;manifest=sealed(manifest)
write('P2_RUNTIME_MANIFEST.json',manifest)
reg=Registry(manifest,manifest['manifest_hash'])
packs={role:base_packing_plan(reg,role) for role in ('train','inner_valid')}
packing={role:dict(packs=len(v),agent_window_exposures=sum(map(len,v)),packing_sha256=digest(v),
    max_agents=max(map(len,v)),all_members_preserved=True) for role,v in packs.items()}
write('BASE_PACKING_PLAN.json',dict(batch_size=64,rule='archived source/window order; keep every agent; flush source boundary and at64',roles=packing,
    stage_a_native_updates=3484,goal_update_cap=150*len(packs['train']),joint_update_cap=250*len(packs['train'])))
write('ROLE_ADDRESS_DRY_RUN.json',dict(status='METADATA_VALID_BUT_NOT_AUTHORIZED',counts=reg.counts,
    membership_sha256=REGISTERED_ROWS,runtime_hash=manifest['manifest_hash'],order_hash=manifest['train_order_hash'],
    archived_order_hash=prior['order_sha256'],same_order=True,physical_train_in_train=3345,physical_test_in_train=139,
    duplicate_eth_removed=139,missing_observation_only_artifacts=4249,recording_unknown=4249,
    real_checkpoint_loads=0,real_payload_reads=0,base_count_qualified=0,cache=None,initial_state=None,reads=reads))
tasks=task_plan(reg)
write('CACHE_TASK_PLAN.json',dict(schema=CACHE_SCHEMA,task_hash=digest(tasks),deployment=len(tasks),
    teachers=sum(t['teacher'] for t in tasks),protected_teachers=0,executed_real=0,producer=None,
    address_rule='NEW root / original physical split / original index.pt; identity wrapper required',
    tasks_example=tasks[:2]))
# Accurate real parser fields only; defaults supplied by main's get_parser.
common=dict(p2_manifest=str((D/'P2_RUNTIME_MANIFEST.json').relative_to(R)),p2_manifest_hash=manifest['manifest_hash'],
    dataset='eth5',test_set='eth',device='cuda:0',seed=3101,num_workers=0,data_augmentation=False,
    clean_split_protocol=False,final_test_access='blocked',pretrain_path=None,load_checkpoint=None,
    shuffle_train_batches=False,use_wandb=False,optimizer='Adam',scheduler='ExponentialLR',clip=1.)
goal=dict(common,p2_mode='goal',phase='goal_pretrain',goal_model_type='independent',
    training_stage='baseline',run_name='p2_clean_goal_seed3101',num_epochs=150,learning_rate=.001,batch_size=64)
joint=dict(common,p2_mode='joint',phase='train',goal_model_type='independent',training_stage='baseline',
    run_name='p2_clean_joint_seed3101',goal_pretrain_checkpoint=None,num_epochs=250,learning_rate=.001,batch_size=64)
for name,config in [('CLEAN_GOAL_P2.yaml',goal),('CLEAN_JOINT_P2.yaml',joint)]:
    (D/name).write_text('# TEMPLATE_BLOCKED; use --p2_config FILE --p2_launch_check\n'+yaml.safe_dump(config,sort_keys=True))
for arm in ('A0','A1'):
    config=yaml.safe_load((P/(arm+'_BLOCKED.yaml')).read_text())
    config.update(common,p2_mode='l1',run_name='p2_l1_'+arm.lower()+'_seed3101',p2_max_update_attempts=500,
        p2_reference_total_steps=3484,p2_skip_validation=True,p2_no_automatic_resume=True,
        p2_arm_seconds=900.,p2_stage_seconds=2700.,p2_reserved_bytes=10*1024**3)
    (D/(arm+'_BLOCKED.yaml')).write_text('# TEMPLATE_BLOCKED; null clean base/cache/init; NOT AUTHORIZED\n'+yaml.safe_dump(config,sort_keys=True))
write('GENERATION_LEDGER.json',dict(metadata_reads=len(reads),hashes=len(reads)+len(parts),
    model_constructions=0,model_calls=0,cuda=0,real_checkpoint_loads=0,real_future_reads=0,
    note='Only historical independent metadata; no raw source, pickle, checkpoint or teacher opened'))
print(json.dumps(dict(counts=reg.counts,packing=packing,runtime_hash=manifest['manifest_hash'])))
