"""Read-only membership and provenance checks shared by training and export."""
import hashlib,json
from pathlib import Path
import numpy as np

WORKSPACE=Path(__file__).resolve().parents[1]
MODEL_SIGNATURE='bcep_relative_geometry_v2'
PURPOSES={'internal_train':'training_diagnostics','internal_dev':'internal_development','interface_smoke':'interface_check','historical_reference':'historical_regression'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def code_identity():
    paths=list((WORKSPACE/'bcep').rglob('*.py'))+[WORKSPACE/'scripts/train_bcep.py',WORKSPACE/'scripts/evaluate_bcep_bank.py']
    return {str(p.relative_to(WORKSPACE)):digest(p) for p in sorted(paths)}

def require(condition,message):
    if not condition:raise ValueError(message)

def manifest_members(path):
    path=Path(path);m=json.loads(path.read_text());rows=[]
    for descriptor in m['shards']:
        shard=(path.parent/descriptor['path']).resolve()
        require(digest(shard)==descriptor['sha256'],'shard hash mismatch')
        with np.load(shard,allow_pickle=True) as z:
            columns=[z[k] for k in ['source_sequences','scene_ids','frame_ids','agent_ids']]
            require(all(len(x)==descriptor['scenes'] for x in columns),'shard identity lengths differ')
            for seq,scene,frames,agents in zip(*columns):
                rows.append(dict(index=len(rows),sequence=str(seq),scene_id=str(scene),frame_ids=[int(x) for x in frames],agent_ids=[str(x) for x in agents]))
    require(len(rows)==m['statistics']['scenes'],'manifest count mismatch')
    require(len({x['scene_id'] for x in rows})==len(rows),'duplicate scenes')
    for r in rows:
        require(len(r['agent_ids'])==len(set(r['agent_ids'])) and len(r['agent_ids'])>0,'duplicate/empty agents')
        require(r['frame_ids']==sorted(set(r['frame_ids'])),'frame ordering differs')
    return m,rows

def load_locked_split(split_path,contract_path):
    split_path=Path(split_path);contract=json.loads(Path(contract_path).read_text())
    require(digest(split_path)==contract['split_hash'],'split hash mismatch')
    split=json.loads(split_path.read_text());source=Path(split['source_manifest'])
    require(digest(source)==split['source_sha256']==contract['source_manifest_hash'],'source manifest mismatch')
    m,rows=manifest_members(source)
    require(rows==split['members'],'locked member identity mismatch')
    require(m['metadata']['unit']==split['unit']==contract['unit']=='m','unit mismatch')
    require(m['metadata']['dt_seconds']==split['dt']==contract['dt'],'dt mismatch')
    require(m['metadata']['pred_length']==split['pred_length']==contract['pred_length'],'horizon mismatch')
    require(split['scale_source']=='internal_train_only','invalid scale provenance')
    scales=np.asarray(split['scales']);require(scales.shape==(split['pred_length'],2) and np.isfinite(scales).all() and (scales>0).all(),'invalid locked training scales')
    train,dev=split['train_indices'],split['dev_indices']
    for name,indices in [('train',train),('dev',dev)]:
        require(all(type(i) is int and 0<=i<len(rows) for i in indices),'invalid index')
        require(len(set(indices))==len(indices)==contract[f'{name}_windows'],'duplicate/omitted subset index')
    require(set(train).isdisjoint(dev) and set(train)|set(dev)==set(range(len(rows))),'train/dev overlap or omission')
    train_seq={rows[i]['sequence'] for i in train};dev_seq={rows[i]['sequence'] for i in dev}
    require(train_seq.isdisjoint(dev_seq) and dev_seq==set(split['dev_sequences']),'sequence leakage')
    for k,v in [('train_member_hash',[rows[i] for i in train]),('dev_member_hash',[rows[i] for i in dev])]:
        require(canonical_hash(v)==contract[k],f'{k} mismatch')
    return split,contract,rows

def check_item(item,member):
    actual=dict(sequence=item['source_sequence'],scene_id=item['scene_id'],agent_ids=item['agent_ids'],frame_ids=item['frame_ids'])
    require(all(actual[k]==member[k] for k in actual),'dataset item identity differs from lock')

def checkpoint_identity(contract_path,split_path):
    return dict(contract=digest(contract_path),split=digest(split_path),source=code_identity(),weights_origin='random_initialization',teacher_weights=False,model_signature=MODEL_SIGNATURE)

def validate_checkpoint(payload,contract_path,split_path):
    require(payload.get('schema')=='bcep-pilot-v2','checkpoint schema/source mismatch')
    expected=checkpoint_identity(contract_path,split_path)
    require(payload.get('identity')==expected,'checkpoint source, split or code identity mismatch')
    require(type(payload.get('update')) is int and payload['update']>=0,'missing checkpoint updates')
    require(type(payload.get('completed_epoch')) is int,'missing checkpoint epoch')
    require(payload.get('selection_metric_name')=='dev_nll_final_agent_micro','wrong checkpoint selector')
