from pathlib import Path
import torch
from bcep.contracts import digest,canonical_hash
from bcep_joint.common import read
from joint_flow_fourfold.inputs import Inputs as FoldInputs,state_hash
from scene_pair_selector.inputs import Inputs as Zara2Inputs
from joint_flow_dependency.model import EdgeDependence

SOURCE=Path(__file__).resolve().parents[1];ROOT=SOURCE.parent
FOUR=ROOT/'RSJG_EBJD_assets/experiments/joint_flow_fourfold_confirmation_20261122'
RUN=ROOT/'RSJG_EBJD_assets/experiments/joint_marginal_residual_20261124'
BASE='4969c12adb5675aa654b2e748dec8da0f6e32adf'
BASE_SOURCE=ROOT/'RSJG_zara1_hotel_frozen_test_20261124'
EXPECTED={'zara1':(6,'fc7d9cca99769b12f2366eda20396130f547534248c82451e1041d640ff8389d'),
 'hotel':(19,'1806aabf478641021584aca02f4040e8e9c2431fe80c457d5adabba259d073d0'),
 'zara2':(35,'ea13cd7fd99a02c44c3c9af9a14a0cc9b5166b629e5c338694cca97f1c8c502a')}

def load_inputs(fold):
 if fold=='zara2':
  inputs=Zara2Inputs();folder=ROOT/'RSJG_EBJD_assets/experiments/zara2_joint_flow_dependency_20261122'
  selected=read(folder/'TRAINING_RESULT.json');checkpoint=folder/'dependence/epoch_035.pt'
 else:
  inputs=FoldInputs(FOUR/fold/'UPSTREAM_FROZEN.json',RUN/fold,'cpu')
  selected=read(FOUR/fold/'B_FROZEN.json');checkpoint=Path(selected['rank1_checkpoint'])
 if fold in EXPECTED:assert (selected['rank1_epoch'],selected['rank1_sha256'])==EXPECTED[fold]
 assert digest(checkpoint)==selected['rank1_sha256']
 identity=dict(fold=fold,teacher_epoch=selected['rank1_epoch'],teacher_sha256=selected['rank1_sha256'],
  upstream=inputs.identity,train_members=canonical_hash([inputs.split['members'][i] for i in inputs.split['train_indices']]),
  dev_members=canonical_hash([inputs.split['members'][i] for i in inputs.split['selection_dev_indices']]))
 return inputs,checkpoint,identity

def load_teacher(fold,device='cuda'):
 inputs,checkpoint,identity=load_inputs(fold);model,provider=inputs.load_frozen(device)
 dependence=EdgeDependence();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert saved['completed_epoch']==identity['teacher_epoch'];dependence.load_state_dict(saved['model'],strict=True)
 for m in (model,provider,dependence):m.to(device).eval().requires_grad_(False)
 identity=dict(identity,states={k:state_hash(m) for k,m in dict(backbone=model,provider=provider,dependence=dependence).items()})
 return inputs,model,provider,dependence,identity
