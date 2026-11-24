"""Observed-only production path: one native B decode followed by the head."""
import argparse,os
from pathlib import Path
import torch
import time
from bcep_joint.common import read,save
from joint_flow_dependency.inference import generate
from .features import coverage_statistics,frozen_features
from .model import ResidualHead
from .assets import load_teacher,load_inputs

@torch.no_grad()
def generate_student(model,provider,dependence,head,batch,*,seed=2035,ablation=False,audit_random_inputs=False):
 assert not batch.future.count_nonzero(),'inference entry accepts observations only'
 if batch.observed.is_cuda:torch.cuda.synchronize()
 tick=time.perf_counter();prediction,endpoints,detail=generate(model,provider,dependence,batch,seed=seed)
 if batch.observed.is_cuda:torch.cuda.synchronize()
 teacher_seconds=time.perf_counter()-tick;tick=time.perf_counter()
 features=frozen_features(batch.observed[0].cpu(),detail['flow_context'].cpu(),prediction[0].cpu(),
  coverage_statistics(endpoints['raw'].cpu(),endpoints['raw_indices'].cpu(),batch.valid[0].cpu()),batch.valid[0].cpu())
 # Feature aggregation is deterministic CPU float64 then stored FP32; head is FP32.
 feature_seconds=time.perf_counter()-tick;tick=time.perf_counter();device=next(head.parameters()).device
 pack={k:v.to(device) for k,v in features.items() if torch.is_tensor(v)}
 result=head(prediction[0].to(device),pack['base_features'],pack['difference_features'],pack['valid'],ablation=ablation)
 proof = None
 if audit_random_inputs:
  from residual_frozen_test.random_audit import audit_actual
  proof = audit_actual(batch,endpoints,seed)
 return result,dict(random_inputs=proof,B=prediction[0],raw_ids=endpoints['raw_indices'],features=features,ddim_calls=detail['ddim_calls'],
  teacher_seconds=teacher_seconds,feature_seconds=feature_seconds,head_seconds=time.perf_counter()-tick)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--fold',required=True);parser.add_argument('--checkpoint',type=Path,required=True)
 parser.add_argument('--purpose',choices=('train','selection_dev'),default='selection_dev')
 parser.add_argument('--position',type=int,required=True);parser.add_argument('--seed',type=int,default=2035)
 parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda')
 args=parser.parse_args()
 if args.device.startswith('cuda'):
  from bcep_v2.runtime import configure_cuda
  configure_cuda()
 saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
 os.environ['RSJG_DISABLE_OPTIMIZATIONS']=saved['native_disabled_optimizations']
 inputs,model,provider,dependence,identity=load_teacher(args.fold,args.device)
 assert saved['teacher_identity']==identity
 head=ResidualHead();head.load_state_dict(saved['model'],strict=True);head.eval()
 index=inputs.split[args.purpose+'_indices'][args.position]
 batch=inputs.data.batch([index],args.purpose,truth=False).to(args.device)
 y,detail=generate_student(model,provider,dependence,head,batch,seed=args.seed,ablation=saved['ablation'])
 save(dict(predictions=y.cpu(),raw_ids=detail['raw_ids'].cpu(),seed=args.seed,checkpoint=str(args.checkpoint),ddim_calls=detail['ddim_calls']),args.output)

if __name__=='__main__':main()
