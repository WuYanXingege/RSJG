"""Real-input CPU interface regression; no GPU and no formal updates."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,copy,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from bcep_formal.data import FormalData
from bcep_formal.optimizations import train_logical_grouped
from bcep_joint.runner import train_logical
from bcep_joint.model import build,optimizer_for,set_lr
from bcep_joint.common import read,write,seed_all,rng_state,restore_rng
from bcep_joint.inference import generate
from ebjd.export_ethucy import semantic_crops
from ebjd.representation import EndpointBridgeDecomposition,cv_baseline

def main(paths):
 started=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1);root=Path(paths['assets']);lock=read(root/'JMM_SPLIT_LOCK.json');result=dict(status='RUNNING',device='cpu',formal_optimizer_updates=0,gpu_seconds=0,thresholds=dict(loss_rtol=2e-4,loss_atol=2e-5,gradient_global_relative=2e-3,H_atol=1e-6,H_rtol=1e-6))
 try:
  ids=[];configs={}
  for fold,row in lock['folds'].items():
   split=read(row['path']);d=FormalData(split);counts={}
   for purpose in ['train','selection_dev','heldout_confirmation']:
    j=split[purpose+'_indices'][0]
    if purpose=='heldout_confirmation':
     try:d.item(j,purpose);raise AssertionError('test leaked')
     except ValueError:pass
     x=FormalData(split,'export',verify=False).item(j,purpose,False);assert not x['future'].any()
    else:x=d.item(j,purpose)
    counts[purpose]=dict(first_scene=x['scene_id'],agents=len(x['agent_ids']))
   configs[fold]=counts
  split=read(lock['folds']['zara2']['path']);d=FormalData(split,verify=False);i=next(i for i in split['train_indices'] if len(split['members'][i]['agent_ids'])==1);j=next(i for i in split['train_indices'] if len(split['members'][i]['agent_ids'])==3);batch=d.batch([i,j],'train')
  # Input cache cannot be mutated by a consumer or compound coordinate normalization.
  x=d.item(i,'train');ref=x['observed'].clone();x['observed'].zero_();assert torch.equal(d.item(i,'train')['observed'],ref)
  from ebjd.export_ethucy import SEQUENCE_TO_SCENE
  for idx in [i,j]:
   x=d.item(idx,'train');mp=lock['maps'][SEQUENCE_TO_SCENE[x['source_sequence']]];original=semantic_crops(x['observed'][:,-1].numpy(),np.asarray(Image.open(mp['semantic_path']),dtype=np.uint8),np.loadtxt(mp['homography_path']),SEQUENCE_TO_SCENE[x['source_sequence']]);assert np.array_equal(original,x['semantic_maps'].numpy())
  rep=EndpointBridgeDecomposition();base,_=cv_baseline(batch.observed);z=rep._encode_unscaled(batch.future.double(),base.double());rebuilt,_=rep.decode(z,base.double());torch.testing.assert_close(rebuilt,batch.future,atol=1e-6,rtol=1e-6)
  c=read(root/'folds/zara2/JOINT_CONFIG.json');seed_all(3201);m=build(c);opt=optimizer_for(m,c);initial=copy.deepcopy(m.state_dict());g=batch.observed[:,None,:,-1].expand(-1,20,-1,-1).clone()+torch.linspace(-1,1,20)[None,:,None,None]
  gen=torch.Generator().manual_seed(3201);rng=rng_state(gen)
  a=train_logical(m,opt,batch,g,gen,'cpu',4);ga={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None}
  m.load_state_dict(initial);m.zero_grad();restore_rng(rng,gen);b=train_logical_grouped(m,opt,batch,g,gen,'cpu',4,32);gb={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None}
  assert ga.keys()==gb.keys();diff=sum((ga[k].double()-gb[k].double()).square().sum() for k in ga).sqrt();base=sum(ga[k].double().square().sum() for k in ga).sqrt().clamp_min(1e-8);relative=float(diff/base)
  result['grouped_real_input']=dict(loss_reference=a[0],loss_candidate=b[0],gradient_relative=relative,passed=abs(a[0]-b[0])<=2e-5+2e-4*abs(a[0]) and relative<=2e-3,agent_counts=[1,3],candidate_points='synthetic interface only; not formal bank')
  # Optional grouping failure is reported and left OFF; H-only identity is mandatory.
  m.load_state_dict(initial);m.zero_grad();_,h,pi,_=generate(m,batch.observed,batch.semantic_maps,batch.valid,batch.scene_ids,batch.agent_ids,g,2035)
  _,only,pi2,_=generate(m,batch.observed,batch.semantic_maps,batch.valid,batch.scene_ids,batch.agent_ids,g,2035,mode='H');torch.testing.assert_close(h,only,atol=1e-6,rtol=1e-6);assert torch.equal(pi,pi2)
  result.update(status='PASS',split_access_checks=configs,map_storage_exact=True,cache_mutation_isolated=True,representation_reconstruction=True,H_only_max_abs=float((h-only).abs().max()),H_only_passed=True)
 except BaseException as e:result.update(status='FAIL',reason=repr(e));raise
 finally:result['cpu_seconds']=time.monotonic()-started;write(root/'receipts/CPU_PREFLIGHT.json',result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);main(read(p.parse_args().paths))
