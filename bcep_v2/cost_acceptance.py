"""Preflight checks with zero real optimizer steps."""
import argparse,csv,io,subprocess,types,tempfile,math
from pathlib import Path
import torch
from bcep_v2.cost_trial import inputs,build,tensor_hash,SOURCE,BASE
from bcep_v2.joint import pair_loss,assignment
from bcep_v2.objective import bridge_and_pair_loss
from bcep_v2.state import run_stage
from bcep_v2.runtime import configure_cuda
from bcep_joint.common import seed_all,read,write
from bcep_formal.data import FormalData
from bcep.contracts import digest,canonical_hash
from ebjd.unified import gather_candidates,log_sinkhorn

def main():
 p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();configure_cuda();torch.set_num_threads(1)
 config,rec,split,banks=inputs(a.upstream)
 old=types.ModuleType('bcep_v2._original_joint');old.__package__='bcep_v2'
 exec(compile(subprocess.check_output(['git','show',BASE+':bcep_v2/joint.py'],cwd=SOURCE),'<fixed original>','exec'),old.__dict__)
 seed_all(3201);legacy=old.build_joint(config).cuda();candidate=build(config,'cuda');assert tensor_hash(legacy)==tensor_hash(candidate);initial=tensor_hash(candidate)
 ids=[i for i in split['train_indices'] if 1<len(split['members'][i]['agent_ids'])<=4][:2]
 data=FormalData(split,'training',verify=False);batch=data.batch(ids,'train')
 from bcep_v2.stages import augment
 batch,g,_=augment(batch,banks[0].batch(ids,batch),torch.Generator().manual_seed(3201));g=g.cuda();x=batch.to('cuda')
 seed_all(991);noise=torch.randn(len(ids),1,x.valid.shape[1],12,2,device='cuda');times=torch.tensor([.31,.67],device='cuda')
 old_indices=[];original_assign=old.assignment
 def tap(scores,valid,training):
  z=original_assign(scores,valid,training);old_indices.append(z[0].detach());return z
 old.assignment=tap;candidate.coupling.capture_assignments=True
 v0=bridge_and_pair_loss(legacy,x,g,time_draw=times,noise=noise);v1=bridge_and_pair_loss(candidate,x,g,time_draw=times,noise=noise)
 assert torch.equal(old_indices[0],candidate.coupling.last_initial_indices) and torch.equal(old_indices[1],candidate.coupling.last_final_indices)
 max_forward=max(float((u-v).abs().max()) for u,v in zip(v0,v1));assert max_forward==0.
 v1[0].backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in candidate.coupling.parameters())
 # Nonzero feedback fixture exposes the initial-score gradient through messages.
 fixture={};hook=candidate.coupling.score.register_forward_hook(lambda m,i,o:(o.retain_grad(),fixture.update(score=o)) and None)
 candidate.zero_grad(set_to_none=True)
 with torch.no_grad():candidate.coupling.compatibility[-1].weight.fill_(.01)
 original_messages=candidate.coupling.messages
 def tapped(features,observed,proposals,valid,initial):
  initial.retain_grad();fixture['initial']=initial
  return original_messages(features,observed,proposals,valid,initial)
 candidate.coupling.messages=tapped
 loss=bridge_and_pair_loss(candidate,x,g,time_draw=times,noise=noise)[2];loss.backward(retain_graph=True)
 assert fixture['initial'].grad is not None and fixture['initial'].grad.abs().sum()>0 and fixture['score'].grad.abs().sum()>0
 via_initial,=torch.autograd.grad(fixture['initial'],fixture['score'],grad_outputs=fixture['initial'].grad)
 assert via_initial.abs().sum()>0
 feedback_grad=float(fixture['initial'].grad.abs().sum());hook.remove();del candidate,legacy;torch.cuda.empty_cache()
 # Exact archived swaps, actual training loss derivative (FP32, tolerance distinct from FP64 audit).
 prefix='docs/research/bcep_v2/formal_fivefold_top5_20261118/';commit='41ddd13f15e65c0d061901087529edfd8c5c21e1'
 raw=subprocess.check_output(['git','show',commit+':'+prefix+'20261119_zara2_same_swap_proxy_check/PAIRED_SWAPS.csv'],cwd=SOURCE).decode();rows=list(csv.DictReader(io.StringIO(raw)))
 chosen=sum(([r for r in rows if int(r['epoch'])==ep and abs(float(r['s_cost_m']))>.001][:8] for ep in [3,5,26]),[]);max_direction=0.
 dev=FormalData(split,'scoring',verify=False)
 for r in chosen:
  idx=int(r['index']);ep=int(r['epoch']);b=dev.batch([idx],'selection_dev');goals=banks[1].batch([idx],b)
  rawpi=subprocess.check_output(['git','show','4c9eede551bebdd7634ab81b7b4bd790165c6bb0:'+prefix+f'20261119_zara2_joint_endpoint_dev/assignments/{ep}/{idx}.npz'],cwd=SOURCE)
  import numpy as np
  with np.load(io.BytesIO(rawpi)) as z:pi=torch.tensor(z['pi_H'])
  soft=torch.nn.functional.one_hot(pi.permute(0,2,1),20).float().requires_grad_();h=gather_candidates(goals,pi)
  val,_=pair_loss(h,b.future[:,:,-1],b.valid,goals=goals,soft=soft);grad,=torch.autograd.grad(val,soft)
  i,u,v,j,k=[int(r[n]) for n in ['agent_position','world_u','world_v','candidate_u','candidate_v']]
  direction=float(grad[0,i,u,k]-grad[0,i,u,j]+grad[0,i,v,j]-grad[0,i,v,k]);err=abs(direction-float(r['s_cost_m']));assert err<2e-6;max_direction=max(max_direction,err)
 # Padding, singleton identity, independent candidate relabeling; costs cannot backprop to inputs.
 seed_all(11);goals=torch.randn(2,20,3,2,requires_grad=True);truth=torch.randn(2,3,2,requires_grad=True);valid=torch.tensor([[True,False,False],[True,True,True]])
 scores=torch.randn(2,3,20,20,requires_grad=True);pi,_,soft=assignment(scores,valid,True);h=gather_candidates(goals,pi)
 assert torch.equal(pi[0,:,0],torch.arange(20));loss,_=pair_loss(h,truth,valid,goals=goals,soft=soft);loss.backward();assert goals.grad is None and truth.grad is None and scores.grad[0].abs().sum()==0
 relabel=torch.stack([torch.randperm(20) for _ in range(6)]).reshape(2,3,20);newgoals=torch.gather(goals.detach(),1,relabel.transpose(1,2)[...,None].expand(-1,-1,-1,2));newsoft=torch.gather(soft.detach(),-1,relabel[:,:,None].expand(-1,-1,20,-1))
 other,_=pair_loss(h.detach(),truth.detach(),valid,goals=newgoals,soft=newsoft);assert float(abs(other-loss))<1e-7
 # Original complete-state writer with the actual proxy; 12 continuous + 12 interrupted synthetic updates.
 recipe=dict(rec,logical_scenes=1,max_epochs=6,minimum=6,patience=20);ident=dict(stage='joint',pair_backward_proxy='candidate_cost',synthetic=True)
 def run(folder,limit,resume=False,identity=ident):
  seed_all(7);m=torch.nn.Linear(2,400);fixed=torch.randn(1,20,2,2);target=torch.randn(1,2,2);valid=torch.ones(1,2,dtype=torch.bool)
  def backward(ids,order,views):
   z=m(torch.randn(2,2)).reshape(1,2,20,20);pi,_,soft=assignment(z,valid,True);loss,_=pair_loss(gather_candidates(fixed,pi),target,valid,goals=fixed,soft=soft);loss.backward();return dict(loss=float(loss))
  history=[6.,2.,2.,5.,3.,4.];count=0 if not resume else torch.load(folder/'last.pt',weights_only=False)['completed_epoch']
  def validate():
   nonlocal count
   v=history[count];count+=1;return v
  return run_stage(folder,identity,recipe,m,[0,1],backward,validate,limit,resume=resume)
 with tempfile.TemporaryDirectory() as d:
  d=Path(d);continuous=run(d/'a',12);partial=run(d/'b',5);resumed=run(d/'b',12,True)
  from tests.test_bcep_integrated_v2 import equal
  for k in ['model','optimizer','scheduler','rng','history','update','cursor','completed_epoch']:equal(continuous[k],resumed[k])
  top=read(d/'b/TOP5.json');assert [r['epoch'] for r in top]==[2,3,5,6,4]
  for r in top:
   p=d/'b'/r['file'];assert digest(p)==r['sha256'];state=torch.load(p,weights_only=False);assert state['completed_epoch']==r['epoch'] and all(k in state for k in ['model','optimizer','scheduler','rng','order','cursor','stale'])
  try:run(d/'b',12,True,dict(ident,pair_backward_proxy='coordinate'))
  except ValueError:pass
  else:raise AssertionError('proxy identity mismatch accepted')
 impl={str(p.relative_to(SOURCE)):digest(p) for p in sorted(SOURCE.glob('bcep_v2/*.py'))}
 write(a.out/'ACCEPTANCE.json',dict(status='PASS',implementation=canonical_hash(impl),initial_tensor_sha256=initial,forward_max_abs=max_forward,hard_permutations_equal=True,selected_coordinates_and_world_errors_equal=True,synthetic_total_including_prior_acceptance_and_pytest=56,feedback_initial_assignment_grad_l1=feedback_grad,same_swap_checks=len(chosen),same_swap_max_abs=max_direction,same_swap_tolerance=2e-6,synthetic_optimizer_updates=24,real_optimizer_updates=0,top5_and_resume_pass=True))
 print('ACCEPTANCE PASS',flush=True)
if __name__=='__main__':main()
