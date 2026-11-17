"""Temporary-state preflight, called only with idle GPU. Never starts training."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,sys,json,time,copy,resource,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from bcep_joint.common import read,write,canonical_hash,seed_all,rng_state,restore_rng
from bcep.contracts import digest
from bcep.provider import BCEP
from bcep_formal.data import FormalData
from bcep_formal.optimizations import provider_loss_batched,finite_tensors,train_logical_grouped,commit_step
from bcep_joint.model import build,optimizer_for
from bcep_joint.runner import train_logical,commit_step as old_commit
from bcep_joint.inference import generate
from run_bcep_jmm import sources

class Events:
 def __init__(self):self.events={}
 def wrap(self,obj,name,label):
  original=getattr(obj,name)
  def call(*args,**kwargs):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();out=original(*args,**kwargs);b.record();self.events.setdefault(label,[]).append((a,b));return out
  setattr(obj,name,call)
 def result(self):return {k:sum(a.elapsed_time(b) for a,b in pairs)/1000 for k,pairs in self.events.items()}

def grads(m):return {k:p.grad.detach().cpu().clone() for k,p in m.named_parameters() if p.grad is not None}
def compare_grad(a,b):
 assert a.keys()==b.keys()
 diff=sum((a[k].double()-b[k].double()).square().sum() for k in a).sqrt();base=sum(a[k].double().square().sum() for k in a).sqrt().clamp_min(1e-8)
 return float(diff/base)

def main(args):
 p=read(args.plan);root=Path(p['assets']);receipt=root/'receipts/GPU_PREFLIGHT.json';started=time.monotonic();result=dict(status='RUNNING',source_hash=canonical_hash(sources()),thresholds=dict(loss_rtol=2e-4,loss_atol=2e-5,gradient_global_relative=2e-3,H_atol=1e-6,H_rtol=1e-6),formal_optimizer_updates=0)
 from bcep_joint.runtime import assert_gpu_idle,configure
 # Must not even create a CUDA context while the previous experiment owns GPU.
 try:assert_gpu_idle()
 except Exception as e:
  result.update(status='BLOCKED_RESOURCE',reason=str(e),gpu_seconds=0);write(receipt,result);return
 configure();write(receipt,result)
 try:
  fold=p['fold_order'][0];row=p['folds'][fold];split=read(row['split']['path']);c=read(row['joint_config']);data=FormalData(split)
  ids=sorted(split['train_indices'],key=lambda i:len(split['members'][i]['agent_ids']));sample_ids=[ids[0],ids[len(ids)//2],ids[-1]]
  result['representatives']=[dict(index=i,agents=len(split['members'][i]['agent_ids']),scene_id=split['members'][i]['scene_id'],map_hw=[256,256]) for i in sample_ids]
  # Real frozen inputs, temporary candidate interface only: no formal bank is generated.
  small=data.batch(ids[:2],'train');goals=small.observed[:,None,:,-1].expand(-1,20,-1,-1).clone()+torch.linspace(-1,1,20)[None,:,None,None]
  seed_all(3201);m=build(c).cuda();opt=optimizer_for(m,c);initial=copy.deepcopy(m.state_dict());gen=torch.Generator().manual_seed(3201);rng=rng_state(gen)
  train_logical(m,opt,small,goals,gen,'cuda',4);baseline=grads(m);old_loss=None
  m.load_state_dict(initial);m.zero_grad();restore_rng(rng,gen)
  # Evaluate both objective values with identical randomness; first call above only
  # captures reference gradients (model has no running normalization buffers).
  old_loss=train_logical(m,opt,small,goals,gen,'cuda',4)[0];baseline=grads(m)
  m.load_state_dict(initial);m.zero_grad();restore_rng(rng,gen)
  new_loss=train_logical_grouped(m,opt,small,goals,gen,'cuda',4,32)[0];candidate=grads(m)
  relative=compare_grad(baseline,candidate);loss_ok=abs(old_loss-new_loss)<=2e-5+2e-4*abs(old_loss)
  result['grouped_correctness']=dict(loss_reference=old_loss,loss_candidate=new_loss,gradient_relative=relative,passed=loss_ok and relative<=2e-3)
  m.load_state_dict(initial);m.zero_grad();b=small.to('cuda');gg=goals.cuda()
  _,h,pi,_=generate(m,b.observed,b.semantic_maps,b.valid,b.scene_ids,b.agent_ids,gg,2035)
  _,only,pi2,_=generate(m,b.observed,b.semantic_maps,b.valid,b.scene_ids,b.agent_ids,gg,2035,mode='H');torch.testing.assert_close(h,only,atol=1e-6,rtol=1e-6);assert torch.equal(pi,pi2)
  result['H_only']=dict(passed=True,max_abs=float((h-only).abs().max()))
  del m,opt,initial,baseline,candidate,b,h,only,gg;torch.cuda.empty_cache()
  # Provider identity and density Jacobian regression on real inputs.
  x=data.item(sample_ids[1],'train');obs=x['observed'].cuda();maps=x['semantic_maps'].cuda();future=x['future'].cuda();valid=torch.ones(len(obs),device='cuda',dtype=torch.bool)
  seed_all(3102);m=BCEP().cuda();m.scales.copy_(torch.tensor(split['scales'],device='cuda'));m.scales_fitted.fill_(True)
  a=m.loss_components(obs,maps,valid,future);a['loss_total'].backward();ga=grads(m);m.zero_grad();bb=provider_loss_batched(m,obs,maps,valid,future);bb['loss_total'].backward();gb=grads(m)
  rel=compare_grad(ga,gb);result['provider_correctness']=dict(loss_reference=float(a['loss_total']),loss_candidate=float(bb['loss_total']),gradient_relative=rel,passed=rel<=2e-3 and torch.isclose(a['loss_total'],bb['loss_total'],atol=2e-5,rtol=2e-4).item())
  del m,a,bb,ga,gb,obs,maps,future,valid;torch.cuda.empty_cache()
  # Representative wall timing, 1 cold+1 warmup then two paired passes per stage.
  timings=[]
  for stage in ['provider','joint']:
   for rep in range(2):
    for mode in ['baseline','candidate']:
     seed_all(3102 if stage=='provider' else 3201);gen=torch.Generator().manual_seed(3201)
     m=(BCEP() if stage=='provider' else build(c)).cuda();opt=(torch.optim.Adam(m.parameters(),lr=1e-4) if stage=='provider' else optimizer_for(m,c))
     if stage=='provider':m.scales.copy_(torch.tensor(split['scales'],device='cuda'));m.scales_fitted.fill_(True)
     trace=Events()
     if stage=='provider':trace.wrap(m.encoder,'forward','encoder')
     else:
      trace.wrap(m.bridge.encoder,'forward','encoder_map_UNet');trace.wrap(m.bridge.denoiser,'forward','denoiser_CTF');trace.wrap(m.coupling,'assign','coupling_assignment')
     durations=[];loads=[];h2d=[];agents=0;start_all=time.monotonic();torch.cuda.reset_peak_memory_stats()
     # Joint logical updates retain 4-scene microbatch /4 accumulation. Selected
     # scenes repeat ONLY in this interface timing, never in the formal recipe.
     for step in range(5):
      torch.cuda.synchronize();t=time.monotonic();wait=0.;transfer=0.;opt.zero_grad(set_to_none=True)
      micros=1 if stage=='provider' else 4
      for mi in range(micros):
       q=time.monotonic();i=sample_ids[(step+mi)%3]
       if stage=='provider':
        x=data.item(i,'train');wait+=time.monotonic()-q;q=time.monotonic();o=x['observed'].cuda();mps=x['semantic_maps'].cuda();y=x['future'].cuda();v=torch.ones(len(o),device='cuda',dtype=torch.bool);torch.cuda.synchronize();transfer+=time.monotonic()-q
        comp=(m.loss_components(o,mps,v,y) if mode=='baseline' else provider_loss_batched(m,o,mps,v,y));comp['loss_total'].backward();finite_tensors([p.grad for p in m.parameters()],'gradients');opt.step();finite_tensors(m.parameters(),'parameters');agents+=len(o)
       else:
        # Four complete scenes, incl. max-N; grouping has a slot cap and never
        # splits agents of the same scene. Existing weights retained.
        batch=data.batch([i]*4,'train');prop=batch.observed[:,None,:,-1].expand(-1,20,-1,-1).clone()+torch.linspace(-1,1,20)[None,:,None,None];wait+=time.monotonic()-q
        (train_logical if mode=='baseline' else train_logical_grouped)(m,opt,batch,prop,gen,'cuda',4);agents+=int(batch.valid.sum())
      if stage=='joint':(old_commit if mode=='baseline' else commit_step)(m,opt)
      torch.cuda.synchronize();durations.append(time.monotonic()-t);loads.append(wait);h2d.append(transfer)
     torch.cuda.synchronize();save_start=time.monotonic()
     temporary=root/'receipts/PROFILE_TEMPORARY_STATE.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),temporary_benchmark_only=True),temporary);save_seconds=time.monotonic()-save_start
     timings.append(dict(save_seconds=save_seconds,save_bytes=temporary.stat().st_size,stage=stage,repeat=rep,mode=mode,cold_update_seconds=durations[0],warmup_update_seconds=durations[1],seconds_per_update=sum(durations[2:])/3,steady_update_seconds=durations[2:],data_seconds_per_update=sum(loads[2:])/3,h2d_seconds_per_update=sum(h2d[2:])/3 if stage=='provider' else None,h2d_joint='included in logical update; not independently timed',component_cuda_seconds_including_warmup=trace.result(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,temporary_agents_processed=agents,wall_seconds=time.monotonic()-start_all))
     del m,opt,trace;gc.collect();torch.cuda.empty_cache()
  result['timings']=timings
  # Complete official validation workload, with random TEMPORARY models. These
  # are timing results, not selected model quality, and never enter a selector.
  validation=[]
  for mode in ['provider','GH','H']:
   seed_all(3102 if mode=='provider' else 3201)
   m=(BCEP() if mode=='provider' else build(c)).cuda().eval()
   if mode=='provider':m.scales.copy_(torch.tensor(split['scales'],device='cuda'));m.scales_fitted.fill_(True)
   torch.cuda.synchronize();start=time.monotonic();agents=0
   with torch.no_grad():
    for i in split['selection_dev_indices']:
     batch=data.batch([i],'selection_dev').to('cuda');agents+=int(batch.valid.sum())
     if mode=='provider':
      m.loss_components(batch.observed[0],batch.semantic_maps[0],batch.valid[0],batch.future[0])
     else:
      prop=batch.observed[:,None,:,-1].expand(-1,20,-1,-1).clone()+torch.linspace(-1,1,20,device='cuda')[None,:,None,None]
      generate(m,batch.observed,batch.semantic_maps,batch.valid,batch.scene_ids,batch.agent_ids,prop,2035,mode=mode)
   torch.cuda.synchronize();validation.append(dict(mode=mode,windows=len(split['selection_dev_indices']),agents=agents,seconds=time.monotonic()-start,temporary_candidate_interface=True,not_validation_quality=True));del m;gc.collect();torch.cuda.empty_cache()
  result['complete_validation_timing']=validation
  # Candidate adoption requires both paired passes >=3%; no changes by task quality.
  recommendations={}
  for stage,correct in [('provider',result['provider_correctness']['passed']),('joint',result['grouped_correctness']['passed'])]:
   speeds=[]
   for rep in range(2):
    xs={r['mode']:r['seconds_per_update'] for r in timings if r['stage']==stage and r['repeat']==rep};speeds.append(xs['baseline']/xs['candidate'])
   recommendations[stage]=dict(speedups=speeds,adopt=correct and min(speeds)>=1.03)
  result['recommendations']=recommendations;result['status']='PASS';result['candidate_settings_not_applied_automatically']=True
  # Gate authorizes verified legacy execution + finite reduction + H-only;
  # optional speed candidates require a separately frozen config before first update.
 except BaseException as e:result.update(status='FAIL',reason=repr(e));raise
 finally:
  result['gpu_wall_seconds']=time.monotonic()-started;write(receipt,result);write(root/'reports/SPEED_PROFILE_BEFORE_AFTER.json',result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--gpu',action='store_true',required=True);main(p.parse_args())
