"""Final GPU checks with ZERO optimizer updates (including cached-edge gradients)."""
import sys,json,subprocess,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import torch
from bcep_v3.fixtures import joint_config,SyntheticData,SyntheticBank
from bcep_v3.model import build,NEW_PREFIXES
from bcep_v2.runtime import configure_cuda
from bcep_v2.joint import neighbors
from bcep_joint.common import seed_all
from bcep_v3.stage2 import callbacks
from test_bcep_v3 import context,coupling
from audit import OUT,PUBLIC,write
configure_cuda();seed_all(915);torch.set_num_threads(1)
config=joint_config();torch.rand(7,device='cuda');before=torch.cuda.get_rng_state().clone();model,_=build(config)
assert torch.equal(before,torch.cuda.get_rng_state())
# Initial old tensors and CPU RNG sequence exactly match same-seed v2 build.
seed_all(810);old,_=build(config,variant='v2_compatible');cpu_before=torch.get_rng_state().clone()
seed_all(810);new,_=build(config);assert torch.equal(cpu_before,torch.get_rng_state())
assert all(torch.equal(t,new.state_dict()[n]) for n,t in old.state_dict().items())
# Cached-edge implementation has identical graph derivatives to recomputing edges.
_,m=coupling();m.cuda();ctx=context(9)
for k,v in vars(ctx).items():setattr(ctx,k,v.cuda())
g=torch.randn(1,20,9,2,device='cuda');p=torch.eye(20,device='cuda').expand(1,9,20,20).clone().requires_grad_()
a=m.hcra.relations(ctx,g,p,torch.tensor(1.,device='cuda'))
b=m.hcra.relations(ctx,g,p,torch.tensor(1.,device='cuda'),neighbors(ctx.observed,ctx.valid))
assert all(torch.equal(x,y) for x,y in zip(a,b))
ga=torch.autograd.grad(a[0].square().sum()+a[1].square().sum(),p)[0]
gb=torch.autograd.grad(b[0].square().sum()+b[1].square().sum(),p)[0];assert torch.equal(ga,gb)
# The only Stage2 graph is the actual grouped backward; all new tensor gradients finite.
checkpoint=torch.load(OUT/'final_continuous/last.pt',map_location='cpu',weights_only=False)
model.load_state_dict(checkpoint['model'],strict=True);model.cuda().train()
data=SyntheticData([2,3],32);bank=SyntheticBank();c=dict(config,batch_scenes=2,gradient_accumulation=1,group_max_slots=4)
bw,_=callbacks(model,data,'cuda',c,bank,bank);bw([0,1],torch.Generator().manual_seed(7),torch.Generator().manual_seed(8))
grads={n:float(p.grad.abs().sum()) if p.grad is not None else None for n,p in model.named_parameters() if n.startswith(NEW_PREFIXES)}
assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if n.startswith(NEW_PREFIXES))
for prefix in NEW_PREFIXES:assert sum(v for n,v in grads.items() if n.startswith(prefix))>0
# Distance boundary ties: all 57 people coincide, therefore all 3192 directed edges are legal.
ctx=context(57)
for k,v in vars(ctx).items():setattr(ctx,k,v.cuda())
ctx.observed.zero_();ctx.baseline.zero_();g=torch.randn(1,20,57,2,device='cuda')
with torch.no_grad():
 h,pi,s=m.assign(ctx,g,torch.tensor(1.,device='cuda'),straight_through=False)
assert m.last_edge_count==57*56 and torch.isfinite(s).all()
write(PUBLIC/'POSTCHECK.json',dict(status='PASS',real_updates=0,synthetic_optimizer_updates=0,
 global_CPU_and_CUDA_rng_preserved=True,old_random_parameters_equal=True,cached_edge_forward_backward_exact=True,
 tied_N57_edges=3192,tied_N57_finite=True,all_new_tensor_gradients_finite=True,tensor_gradient_l1=grads,
 note='Nonzero gradient proves graph connectivity only; additive softmax-invariant biases can have negligible roundoff gradients.'))
print('PASS final zero-update GPU checks')
