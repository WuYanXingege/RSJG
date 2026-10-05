"""Read-only provenance and sorted train-window selection; NO CUDA/models."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
import yaml
from src.parser import get_parser, check_and_add_additional_args
from src.data_loader import dataset_set_name
from src.batch_cache_io import load_batch_cache
from src.joint_dependency_v2_cache import load_cache_record
from src.models.model import GDTS, create_dataset
from torch.utils.data import default_collate
from tools.jdv2_stage_a_bank import CONFIG, CONFIG_SHA, CHECKPOINT, CHECKPOINT_SHA, CACHE, CACHE_SHA, SOURCE_BATCHES, sha256


def settings():
    parser=get_parser(); parser.set_defaults(**yaml.safe_load((ROOT/CONFIG).read_text()))
    a=parser.parse_args([])
    a.device='cpu'; a.amp_enabled=False; a.amp_dtype='fp32'; a.phase='train'
    a.load_checkpoint=None; a.pretrain_path=None; a.use_wandb=False
    a=check_and_add_additional_args(a)
    a.model_dir='/tmp/rsjg_cuda_loss_smoke_unused'
    return a


def describe(v):
    if torch.is_tensor(v):
        return dict(shape=list(v.shape),dtype=str(v.dtype),stride=list(v.stride()),
                    sha256=hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest())
    if isinstance(v,dict): return {k:describe(x) for k,x in v.items() if k!='scene'}
    return str(type(v).__name__)


def prepare():
    for path,expected in ((CONFIG,CONFIG_SHA),(CHECKPOINT,CHECKPOINT_SHA),(CACHE+'/manifest.json',CACHE_SHA)):
        assert sha256(ROOT/path)==expected, path
    a=settings()
    ds=dataset_set_name(a,'train')  # Full unchanged production manifest validation.
    records=[]
    for index,name in enumerate(ds.ids):
        data,identity=load_batch_cache(str(Path(ds.path_to_folder)/name))
        n=data['abs_pixel_coord'].shape[1]
        if n<=8:
            records.append((str(identity['data_file_path']),int(identity['frame_ids'][0]),name,index,n))
        del data
    selected={}; cache_reads=0
    for source,frame,name,index,n in sorted(records):
        path=ROOT/CACHE/'train'/f'{index:06d}.pt'
        cache=load_cache_record(str(path),allow_future_supervision=False); cache_reads+=1
        e=cache['edge_index'].shape[1]
        if e>32: continue
        category='empty' if e==0 else 'edge'
        if category in selected: continue
        selected[category]=dict(source=source,frame=frame,filename=name,index=index,N=n,E=e,
            source_batch_sha256=sha256(Path(ds.path_to_folder)/name),
            cache_sha256=sha256(path),teacher_sha256=sha256(path.with_name(f'{index:06d}.teacher.pt')))
        if len(selected)==2: break
    assert len(selected)==2, 'BLOCKED_MISSING_ARTIFACT'
    shell=SimpleNamespace(args=a,device=torch.device('cpu'),dataset=create_dataset(a.dataset,load_visual_data=False),active_goal_model_type='joint_dependency_v2')
    for row in selected.values():
        batch,identity=default_collate([ds[row['index']]])
        prepared,sequence=GDTS.prepare_inputs(shell,batch,identity)
        row['prepared']=describe(prepared); row['sequence']=describe(sequence)
    assert not torch.cuda.is_initialized()
    return dict(status='READY_CPU_PROVENANCE',checkpoint=dict(path=CHECKPOINT,sha256=CHECKPOINT_SHA),
        config=dict(path=CONFIG,sha256=CONFIG_SHA),cache_manifest_sha256=CACHE_SHA,
        validated_manifest=ds.jdv2_manifest,selection=selected,
        selection_rule='sorted (source absolute path, first frame, physical filename); first N<=8 E<=32 of each type',
        fallback_used=False,source_batch_deserializations=len(ds.ids)+2,
        candidate_cache_deserializations=cache_reads+2,teacher_cache_deserializations=2,
        checkpoint_deserializations=0,checkpoint_hash_reads=1,
        exploratory_reads_before_script={'train_deployment_000000':1,'train_batch_0001':1},
        resolved_cpu_args=vars(a),cuda_initialized=False)


if __name__=='__main__':
    log=io.StringIO()
    with contextlib.redirect_stdout(log): result=prepare()
    result['preparation_log']=log.getvalue()
    print(json.dumps(result,indent=2,allow_nan=False))
