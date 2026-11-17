import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from bcep.provider import BCEP
from bcep_formal.optimizations import provider_loss_batched,finite_tensors
from bcep_joint.checks import equal,synthetic_batch
from bcep_joint.model import set_lr

def test_horizon_batch_loss_and_gradients():
 torch.set_num_threads(2);torch.manual_seed(41);m=BCEP()
 with torch.no_grad():
  for net in m.flow.nets:net[-1].weight.add_(torch.randn_like(net[-1].weight)*.001)
 obs=torch.randn(3,8,2)*.1;fut=torch.randn(3,12,2)*.1;maps=torch.rand(3,6,16,16);valid=torch.ones(3,dtype=torch.bool)
 a=m.loss_components(obs,maps,valid,fut);a['loss_total'].backward();grad={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None};m.zero_grad()
 b=provider_loss_batched(m,obs,maps,valid,fut);b['loss_total'].backward()
 torch.testing.assert_close(a['loss_total'],b['loss_total'],rtol=2e-5,atol=2e-5)
 for n,p in m.named_parameters():
  if n in grad:torch.testing.assert_close(grad[n],p.grad,rtol=2e-4,atol=2e-5)

def test_finite_reduction_rejects_nan():
 finite_tensors([torch.ones(3),None],'test')
 with pytest.raises(FloatingPointError):finite_tensors([torch.tensor([float('nan')])],'test')

def test_scheduler_reads_100_horizon():
 m=torch.nn.Linear(2,2);o=torch.optim.AdamW(m.parameters(),lr=1e-4)
 assert set_lr(o,5,100)['factor']==1
 assert set_lr(o,30,100)['factor']>.8
 assert set_lr(o,100,100)['factor']==.1

def test_provider_complete_state_resume(tmp_path):
 from bcep_formal.provider_runner import run_provider
 from bcep.contracts import canonical_hash
 torch.set_num_threads(2);gen=torch.Generator().manual_seed(3);items=[];members=[]
 for i in range(4):
  member=dict(sequence='s',scene_id=str(i),agent_ids=['1','2'],frame_ids=list(range(20)));members.append(member)
  items.append(dict(source_sequence='s',scene_id=str(i),agent_ids=['1','2'],frame_ids=list(range(20)),observed=torch.randn(2,8,2,generator=gen)*.1,future=torch.randn(2,12,2,generator=gen)*.1,semantic_maps=torch.ones(2,6,8,8)))
 split=dict(train_indices=[0,1,2],dev_indices=[3],pred_length=12,dt=.4,unit='m',scales=np.ones((12,2)).tolist(),members=members)
 cfg=dict(device='cpu',max_epochs=100,min_completed_epochs=20,max_updates=300,seed=3102,lr=1e-4,lr_gamma=1.,lambda_aux=.25,patience=15,session_seconds=None,batched_horizons=True)
 ident={'split':canonical_hash(split)}
 a=run_provider(items,split,cfg,ident,tmp_path/'a',stop_after_updates=4)
 run_provider(items,split,cfg,ident,tmp_path/'b',stop_after_updates=2)
 b=run_provider(items,split,cfg,ident,tmp_path/'b',resume=True,stop_after_updates=2)
 for k in ['model','optimizer','scheduler','rng','epoch','completed_epoch','update','train_cursor','dev_cursor','order','best','stale']:equal(a[k],b[k])

def test_joint_accumulation_resume(tmp_path,monkeypatch):
 from bcep_formal import joint_runner as r
 from ebjd.data import collate_scenes,augment_batch_with_points
 class Tiny(torch.nn.Module):
  def __init__(self,c):
   super().__init__();self.bridge=torch.nn.Module();self.bridge.encoder=torch.nn.Module();self.bridge.encoder.map_unet=torch.nn.Linear(2,4);self.bridge.denoiser=torch.nn.Linear(4,2);self.coupling=torch.nn.Linear(4,2)
 class Data:
  def batch(self,ids,purpose):
   b=synthetic_batch(2,8)
   return collate_scenes([dict(observed=b.observed[0]+i*.01,future=b.future[0]+i*.01,semantic_maps=b.semantic_maps[0],metadata={},scene_id=str(i),agent_ids=['0','1'],frame_ids=list(range(20)),timestamps=[j*.4 for j in range(20)],source_sequence='synthetic') for i in ids])
 class Bank:
  def batch(self,b):return b.observed[:,None,:,-1,:].expand(-1,20,-1,-1).clone()
 def logical(m,opt,b,g,gen,device,acc):
  b,g=augment_batch_with_points(b,g,gen);h=m.bridge.encoder.map_unet(b.observed[:,:,-1]);loss=((m.bridge.denoiser(h)+m.coupling(h))-b.future[:,:,-1]+torch.randn_like(b.future[:,:,-1])*.01).square().mean();(loss/acc).backward();return [float(loss.detach())]*3
 def generate(m,obs,maps,valid,sids,aids,goals,seed,mode='H'):
  h=m.bridge.encoder.map_unet(obs[:,:,-1]);path=goals[:,:,:,None].expand(-1,-1,-1,12,-1)+(m.bridge.denoiser(h)+m.coupling(h))[:,None,:,None];return None,path.detach(),None,{}
 monkeypatch.setattr(r,'train_logical',logical);monkeypatch.setattr(r,'generate',generate)
 cfg=dict(downstream_seed=3201,batch_scenes=2,gradient_accumulation=4,max_epochs=2,min_completed_epochs=30,patience=15,scheduler_horizon=100,weight_decay=.0001,betas=[.9,.999],eps=1e-8,session_seconds=None)
 split=dict(train_indices=list(range(7)),selection_dev_indices=[7,8]);args=(cfg,split,Data(),Bank(),Bank(),{'test':1})
 a=r.run_joint(*args,tmp_path/'a',device='cpu',model_factory=Tiny)
 part=r.run_joint(*args,tmp_path/'b',device='cpu',model_factory=Tiny,stop_after_microbatches=3);assert part['pending']==3
 b=r.run_joint(*args,tmp_path/'b',device='cpu',model_factory=Tiny,resume=True)
 for k in ['model','optimizer','scheduler','rng','pending_gradients','epoch','completed_epoch','update','order','cursor','pending','stale']:equal(a[k],b[k])
 equal(a['selection']['best'],b['selection']['best'])

def test_fivefold_supervisor_serial_and_restart(tmp_path,monkeypatch):
 import argparse,json
 from bcep_joint.common import write
 from bcep.contracts import canonical_hash,digest
 from bcep_formal import provider_runner,joint_runner,banks,evaluate,summary
 from bcep_joint import runtime,data as joint_data
 import run_bcep_jmm as runner
 a=tmp_path/'assets';a.mkdir();(a/'identity').mkdir();(a/'receipts').mkdir();calls=[]
 monkeypatch.setattr(runner,'sources',lambda:{'f':'hash'})
 monkeypatch.setattr(runtime,'assert_gpu_idle',lambda:None);monkeypatch.setattr(runtime,'configure',lambda:None);monkeypatch.setattr(runner,'install',lambda:None)
 for name in ['reset_peak_memory_stats','empty_cache']:monkeypatch.setattr(torch.cuda,name,lambda:None)
 for name in ['max_memory_allocated','max_memory_reserved']:monkeypatch.setattr(torch.cuda,name,lambda:0)
 class Data:
  def __init__(self,*args,**kw):pass
 class Bank:
  def __init__(self,*args,**kw):self.sha='fake'
 import bcep_formal.data
 monkeypatch.setattr(bcep_formal.data,'FormalData',Data);monkeypatch.setattr(joint_data,'CandidateBank',Bank)
 def provider(data,s,c,ident,out,resume=False):
  calls.append((s['fold'],'provider'));out.mkdir(exist_ok=True);(out/'best.pt').write_bytes(b'fake');return dict(phase='complete',selected_epoch=20)
 def export(s,cp,purpose,seed,out):calls.append((s['fold'],purpose,seed));return dict(path=str(out),provider_sha256=digest(cp))
 def joint(c,s,data,tr,dev,ident,out,resume=False):
  calls.append((s['fold'],'joint'));out.mkdir(exist_ok=True);(out/'best.pt').write_bytes(b'fake');write(out/'STATUS.json',dict(status='COMPLETE',selection=dict(best=dict(epoch=30))));return dict(phase='complete')
 def evaluation(s,*args):calls.append((s['fold'],'evaluation'));return dict(status='COMPLETE')
 monkeypatch.setattr(provider_runner,'run_provider',provider);monkeypatch.setattr(banks,'export_bank',export);monkeypatch.setattr(joint_runner,'run_joint',joint);monkeypatch.setattr(evaluate,'evaluate',evaluation);monkeypatch.setattr(summary,'summarize',lambda plan:None)
 order=['zara2','zara1','hotel','univ','eth'];plan=dict(assets=str(a),worktree=str(Path(__file__).resolve().parents[1]),fold_order=order,authorization={'explicit_fivefold_authorization':True},folds={})
 lock=a/'SPLIT.json';write(lock,{});plan.update(split_lock=str(lock),split_sha256=digest(lock))
 for f in order:
  folder=a/f;folder.mkdir();s=folder/'split.json';p=folder/'p.json';j=folder/'j.json';write(s,{'fold':f});write(p,{});write(j,{})
  plan['folds'][f]=dict(split=dict(path=str(s),sha256=digest(s)),provider_config=str(p),provider_config_sha256=digest(p),joint_config=str(j),joint_config_sha256=digest(j),output=str(folder))
 file=a/'PLAN.json';write(file,plan);write(a/'PIPELINE_STATE.json',{'status':'PREPARED'});write(a/'receipts/CPU_PREFLIGHT.json',{'status':'PASS'});h=canonical_hash(runner.sources());write(a/'receipts/GPU_PREFLIGHT.json',dict(status='PASS',source_hash=h));write(a/'receipts/RUNTIME_FROZEN.json',dict(source_hash=h,plan_sha256=digest(file)))
 args=argparse.Namespace(plan=str(file),execute_authorized_fivefold=True);runner.main(args)
 assert [f for f,phase,*rest in calls if phase=='provider']==order
 assert [f for f,phase,*rest in calls if phase=='joint']==order
 assert [f for f,phase,*rest in calls if phase=='evaluation']==order
 n=len(calls);runner.main(args);assert len(calls)==n
 assert json.loads((a/'PIPELINE_STATE.json').read_text())['status']=='COMPLETE'
