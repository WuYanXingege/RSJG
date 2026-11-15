#!/usr/bin/env python3
import sys,json,time,dataclasses,torch,argparse
from pathlib import Path
paper=Path(__file__).resolve().parents[1];sys.path[:0]=[str(paper),str(paper/'scripts')]
from train_learned_goal_bridge import build_scratch,load_config,loader,atomic_json
from ebjd.finite_k_risk import finite_k_endpoint_risk
parser=argparse.ArgumentParser(description='Frozen density quadrature versus 1000 IID20 repetitions; no optimizer')
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--contract',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();out=args.output
if out.exists():raise FileExistsError(out)

report={'status':'RUNNING','rows':[]};atomic_json(out,report);t=time.perf_counter();torch.set_num_threads(2)
try:
 c=load_config(args.config);k=json.loads(args.contract.read_text());model=build_scratch(c,k).cuda().eval()
 old=args.checkpoint
 payload=torch.load(old,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'],strict=True);del payload
 torch.manual_seed(9913)
 with torch.no_grad():
  for i,b in enumerate(loader(c,'validation',torch.Generator().manual_seed(42))):
   if i>=8:break
   b=b.to('cuda');ctx=model.encode_context(b.observed,b.semantic_maps,b.valid);h=model.goal;d=h(ctx);s=model.bridge.representation.goal_scale
   vals=[finite_k_endpoint_risk(h,ctx,b.future[...,-1,:],s,d,order=o).item()*s.item() for o in [2,4]]
   copies=1000
   ctx2=dataclasses.replace(ctx,valid=ctx.valid.expand(copies,-1),social_h=ctx.social_h.expand(copies,-1,-1),map_center=ctx.map_center.expand(copies,-1,-1))
   dd=type(d)(*(v.expand(copies,*v.shape[1:]) for v in (d.spatial_logits,d.tail_logit,d.side_logits,d.tail_scale)))
   g=h.sample(ctx2,s,distribution=dd)
   errors=torch.linalg.vector_norm(g-b.future[:,None,:,-1,:],dim=-1).min(1).values[:,b.valid[0]].mean(1)
   row={'scene':b.scene_ids,'agents':int(b.valid.sum()),'order2_m':vals[0],'order4_m':vals[1],'MC_mean_m':errors.mean().item(),'MC_SE_m':errors.std().item()/copies**.5}
   report['rows'].append(row);atomic_json(out,report)
 report['status']='COMPLETE'
except BaseException as e:report['status']='FAILED';report['error']=repr(e);raise
finally:
 report['gpu_task_wall_seconds']=time.perf_counter()-t;atomic_json(out,report)
print(json.dumps(report,indent=2))
