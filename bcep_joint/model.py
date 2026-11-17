import math
import torch
from ebjd.model import EBJDModel
from ebjd.unified import UnifiedCouplingBridge
from .common import forbid_weight_reads

def build(config):
    assert config['schema']=='bcep-sixlayer-random-downstream-v1'
    assert config['initialization_role']=='random_all_downstream' and config['teacher_used'] is False
    assert (config['hidden_dim'],config['heads'],config['ff_dim'],config['fjab_blocks'])==(192,4,768,6)
    assert config['precision']=='fp32' and config['proposal_eta']==0 and not config['rollout'] and not config['projection']
    assert not any('matched_gdts' in k or 'pretrained' in k for k in config)
    with forbid_weight_reads():
        b=EBJDModel(goal_scale=config['scales']['goal'],bridge_scale=config['scales']['bridge'],crop_width_m=config['crop_width_m'],map_agent_chunk=config['map_agent_chunk'],hidden_dim=192,attention_heads=4,ff_dim=768,history_social_blocks=config['history_social_blocks'],ctf_feedback_basis='legacy',proposal_conditioned=True)
        b.encoder.activation_checkpointing=True;b.denoiser.activation_checkpointing=True
        b.encoder.checkpoint_preserve_rng_state=False;b.denoiser.checkpoint_preserve_rng_state=False
        m=UnifiedCouplingBridge(b);m.training_options={'coarse':.25,'geometry':.05,'map':.10}
    assert len(b.denoiser.blocks)==6
    return m

def optimizer_for(model,c):
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        (no_decay if p.ndim<=1 or name.endswith('bias') or 'norm' in name.lower() else decay).append(p)
    return torch.optim.AdamW([dict(params=decay,lr=1e-4,weight_decay=c['weight_decay']),dict(params=no_decay,lr=1e-4,weight_decay=0.)],betas=tuple(c['betas']),eps=c['eps'])

def set_lr(optimizer,epoch,horizon=30):
    assert horizon>5
    factor=epoch/5 if epoch<=5 else .1+.9*.5*(1+math.cos(math.pi*min((epoch-5)/(horizon-5),1)))
    for g in optimizer.param_groups:g['lr']=1e-4*factor
    return dict(type='original_epoch_warmup_cosine',epoch=epoch,warmup_epochs=5,horizon_epochs=horizon,factor=factor,base_lr=1e-4)
