"""Resolve all five recipes from immutable official split locks before updates."""
import sys,json,math,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_joint.common import write,read
from bcep.contracts import digest,canonical_hash

def main(p):
 a=Path(p['assets']);old=read(Path(p['root'])/'RSJG_EBJD_assets/experiments/bcep_sixlayer_joint_20261117_100323/identity/RESOLVED_CONFIG.json');lock=read(a/'JMM_SPLIT_LOCK.json')
 # Preserve the actual provider optimizer/scheduler recipe, not GDTS defaults.
 cfgs=list((Path(p['root'])/'RSJG_BCEP_Confirmation_20261116_223645').glob('docs/research/bcep_v1/*/*CONFIG*.json'))
 identity_source=read(Path(p['root'])/'RSJG_EBJD_assets/experiments/bcep_sixlayer_joint_20261117_100323/RUN_PLAN.json')['runs'][0]['provider']['path']
 import torch
 prior=torch.load(identity_source,map_location='cpu',weights_only=False)
 lr_gamma=prior['scheduler']['gamma'];assert prior['optimizer']['param_groups'][0]['betas']==(.9,.999) and prior['optimizer']['param_groups'][0]['weight_decay']==0
 # Auxiliary objective is fixed in the original runner; no weight reuse.
 plan=dict(schema='bcep-jmm-fivefold-plan-v1',assets=str(a),worktree=p['worktree'],docs=p['docs'],split_lock=str(a/'JMM_SPLIT_LOCK.json'),split_sha256=digest(a/'JMM_SPLIT_LOCK.json'),fold_order=['zara2','zara1','hotel','univ','eth'],provider_seed=3102,downstream_seed=3201,inference_seeds=list(range(2035,2040)),authorization=read(a/'identity/AUTHORIZATION.json'),stages=['provider','banks_train_val','joint','test_banks','frozen_evaluation'],folds={},requires_gpu_preflight=True,allow_real_training=False)
 for fold,row in lock['folds'].items():
  s=read(row['path']);folder=a/'folds'/fold
  provider=dict(device='cuda',seed=3102,max_epochs=100,min_completed_epochs=20,patience=15,lr=1e-4,lr_gamma=lr_gamma,lambda_aux=.25,support=256,candidates=20,max_updates=len(s['train_indices'])*100,selection_metric='dev_nll_final_agent_micro',session_seconds=None,batched_horizons=False,optimizer='Adam',optimizer_betas=[.9,.999],optimizer_eps=1e-8,weight_decay=0,recipe_source_checkpoint=identity_source,recipe_source_sha256=digest(identity_source),learned_weights_loaded=False)
  c={k:v for k,v in old.items() if k not in ['budget_seconds','source_resolved','max_updates','optimizer_updates_per_epoch']};c.update(scales=s['bridge_scales'],max_epochs=100,min_completed_epochs=30,patience=15,scheduler_horizon=100,session_seconds=None,grouped_training=False,group_max_slots=32,selection_mode='H',allow_real_training=False)
  c['optimizer_updates_per_epoch']=math.ceil(math.ceil(len(s['train_indices'])/4)/4);c['max_updates']=100*c['optimizer_updates_per_epoch'];c['tail_flush']='original: final partial accumulation divides by4, not by pending microbatch count'
  write(folder/'PROVIDER_CONFIG.json',provider);write(folder/'JOINT_CONFIG.json',c)
  curve=[dict(epoch=e,factor=e/5 if e<=5 else .1+.9*.5*(1+math.cos(math.pi*(e-5)/95)),lr=1e-4*(e/5 if e<=5 else .1+.9*.5*(1+math.cos(math.pi*(e-5)/95)))) for e in range(1,101)];write(folder/'SCHEDULER_CURVE.json',curve)
  plan['folds'][fold]=dict(split=row,provider_config=str(folder/'PROVIDER_CONFIG.json'),provider_config_sha256=digest(folder/'PROVIDER_CONFIG.json'),joint_config=str(folder/'JOINT_CONFIG.json'),joint_config_sha256=digest(folder/'JOINT_CONFIG.json'),output=str(folder))
 write(a/'RUN_PLAN.json',plan);docs=Path(p['docs']);write(docs/'RESOLVED_PLAN.json',plan);write(docs/'SCHEDULER_CURVE.json',curve)
 write(a/'PIPELINE_STATE.json',dict(status='PREPARED_GPU_GATE_PENDING',stage='GPU_PREFLIGHT',formal_optimizer_updates=0))
 print('CONFIGURED',list(plan['folds']), 'provider gamma',lr_gamma)
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--paths',required=True);main(json.loads(Path(pa.parse_args().paths).read_text()))
