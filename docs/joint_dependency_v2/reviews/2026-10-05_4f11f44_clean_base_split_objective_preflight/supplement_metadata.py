"""Read only saved JSON/log/source evidence. No torch or model imports."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

D=Path(__file__).resolve().parent
R=D.parents[3]
reads=[]
def read(path):
    p=Path(path);p=p if p.is_absolute() else R/p
    b=p.read_bytes();reads.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    return b.decode()
b=json.loads(read(D/'BASE_PROVENANCE.json'))
rows={c['id']:c for c in b['candidates']}
logs={}
for fold in ['eth','univ']:
    path='../GDTS/output/'+fold+'/log_curve.txt'
    data=list(csv.DictReader(io.StringIO(read(path))))
    best=min(data,key=lambda row:float(row['valid_ADE']))
    logs[fold]=dict(path=path,epochs_logged=len(data),best_ADE_epoch=int(best['epoch']),
        checkpoint_epoch=rows['gdts_'+fold]['metadata']['epoch'],epoch_matches=int(best['epoch'])==rows['gdts_'+fold]['metadata']['epoch'],
        positive_goal_and_diffusion_losses=all(float(x['goal_BCE_loss'])>0 and float(x['diffusion_loss'])>0 for x in data),
        selection_rule='independent best_metric auto -> legacy pixel ADE; not new heldout metrics')
path='outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_clean_eth_seed2035/'
protocol=json.loads(read(path+'evaluation_protocol.json'))
lines=read(path+'training.log').splitlines()
full=[s[:160] for s in lines if 'Train batches' in s and '3790/3790' in s]
initial=read(path+'training_command.txt').strip()
result=dict(baseline_completed_training_evidence=logs,p1_run=dict(training_command=initial,
    train_members=len(protocol['train_cache_ids']),valid_members=len(protocol['valid_cache_ids']),
    source=protocol['internal_validation_source'],full_train_epoch_progress=full[-1:],
    timing_boundary='tqdm training loop 02:52=172s; epoch diagnostic 381.22035s includes validation; neither establishes P2/MC/base speed',
    checkpoint_steps={k:rows[k]['metadata']['stage_optimizer_steps_completed'] for k in ['p1_clean_a_best','p1_clean_a_last']}))
# Every definition in these architecture modules is unchanged from preregistration.
modules=['social_encoder.py','relation_inference.py','joint_dependency_v2/unary_goal.py',
    'joint_dependency_v2/dynamic_relation.py','joint_dependency_v2/joint_energy.py','joint_dependency_v2/future_teacher.py',
    'joint_dependency_v2/dependency_corrector.py','joint_dependency_v2/joint_sampler.py',
    'model_utils/U_net_CNN.py','model_utils/hist_traj_rnn_encoder.py','diffusion.py']
checks={}
for rel in modules:
    p='src/models/'+rel;current=read(p)
    old=subprocess.check_output(['git','show','79ea3fa:'+p],cwd=R,text=True)
    equal=ast.dump(ast.parse(current),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
    checks[p]=dict(all_definitions_unchanged=equal,current_sha256=hashlib.sha256(current.encode()).hexdigest())
    assert equal,p
keys=rows['canonical_a']['state']['keys']
head_names=['social_encoder','relation_inference','jdv2_future_teacher','jdv2_unary','jdv2_dynamic_relation','jdv2_joint_energy']
head=sum(v['numel'] for k,v in keys.items() if k.split('.')[0] in head_names)
assert head==331845
result['architecture']=dict(module_definition_checks=checks,registered_heads=head,effective_endpoint_path=head-64,
    frozen_corrector=30851,registered_base_parameters=7826588,base_active_forward_parameters=5723804,
    base_buffer_values=12793,unused_diffusion_template_parameters=2102784,
    parameter_note='state elements include positional and schedule buffers; exclude them from registered parameter totals',
    new_parameters=0,new_buffers=0,checkpoint_keys_shapes_file='BASE_PROVENANCE.json')
for p in ['src/data_loader.py','src/clean_split_protocol.py','src/joint_dependency_v2_cache.py','src/parser.py',
    'src/trainer.py','src/models/model.py','src/jdv2_objective_state.py','src/models/goal_pretrain.py',
    'src/data_src/experiment_src/experiment_eth5.py','src/data_src/scene_src/scene_base.py','src/data_pre_process.py',
    '../GDTS/src/data_src/experiment_src/experiment_eth5.py','../GDTS/src/trainer.py','../GDTS/src/models/model.py']:
    read(p)
result['source_anchors']={'P1_count_guard':'src/clean_split_protocol.py:65',
    'source_block_only':'src/data_loader.py:333','teacher_permission':'src/data_loader.py:277',
    'all_split_teacher_build':'src/joint_dependency_v2_cache.py:293',
    'no_step_cap':'src/trainer.py:1540','loss_progress':'src/trainer.py:1495',
    'epoch_scheduler':'src/trainer.py:1392','selection_only_stageB_tie':'src/trainer.py:248',
    'static_future_separation':'src/models/model.py:1391',
    'main_no_config_cli':'src/parser.py:1197'}
result['reads']=reads
result['counts']={'metadata_log_source_reads':len(reads),'sha256_operations':len(reads),
    'checkpoint_deserializations':0,'cache_sidecar_reads':0,'protected_future_tensor_reads':0,
    'models':0,'cuda':0,'optimizer':0,'training':0}
print(json.dumps(result,indent=2))
