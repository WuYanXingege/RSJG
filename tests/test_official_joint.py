"""Fresh-joint parent, initialization and configuration contracts."""
import copy,json
from pathlib import Path
import pytest,torch
from src.p2_protocol import ROOT,Registry,data_binding
from src.p2_grouped_training import qualify_parent,initialized_model
from tools.grouped_model_smoke import arguments

D=ROOT/'docs/joint_dependency_v2/reviews/2026-10-06_6d04a87_official_prior_fresh_joint'
G=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_2c25c1f_official_prior_fresh_goal'
def registry():
    m=json.loads((D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').read_text());return Registry(m,m['manifest_hash'])

def test_selected_parent_and_protocol_identity():
    reg=registry();p=qualify_parent(reg.manifest['parents']['goal'],reg,'goal')
    assert p['progress']['epoch_completed']==96
    assert reg.manifest['parents']['goal']['sha256']=='f6a52cf228d733bd684aef043d843ba7d942e40495d5eec354ce1e4e3507ec13'
    gm=json.loads((G/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').read_text());old=Registry(gm,gm['manifest_hash'])
    assert data_binding(reg)==data_binding(old)

def test_parent_tampering_rejected():
    reg=registry();ref=copy.deepcopy(reg.manifest['parents']['goal']);ref['sha256']='0'*64
    with pytest.raises(RuntimeError,match='actual hash'):qualify_parent(ref,reg,'goal')

def test_joint_config_is_fresh_and_frozen():
    reg=registry();c=json.loads((D/'FRESH_JOINT_CONFIG.yaml').read_text())
    assert (c['p2_mode'],c['phase'],c['num_epochs'],c['learning_rate'])==('joint','train',250,.0001)
    assert c['goal_pretrain_checkpoint']==reg.manifest['parents']['goal']['path']
    assert c['pretrain_path'] is None and c['load_checkpoint'] is None and c['amp_dtype']=='fp32' and c['batch_size']==64

def test_production_initialization_loads_goal_only():
    reg=registry();a=arguments('joint','cpu',D);a.goal_pretrain_checkpoint=reg.manifest['parents']['goal']['path']
    model,initial,non_goal,parent=initialized_model(a,reg,torch.device('cpu'))
    selected={k[len('goal_module.'):]:v for k,v in parent['model_state_dict'].items() if k.startswith('goal_module.')}
    from src.p2_checkpoint import state_hash
    assert state_hash(model.goal_module.state_dict())==state_hash(selected)
    assert state_hash({k:v for k,v in model.state_dict().items() if not k.startswith('goal_module.')})==non_goal
    assert state_hash(model.state_dict())==initial
    assert model.set_losses_coeffs()=={'diffusion_loss':1,'goal_BCE_loss':20}
