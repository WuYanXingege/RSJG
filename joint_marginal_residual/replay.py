"""Selected head: re-run full observed-only dev, then reconcile native scores."""
import time
import torch
from bcep.contracts import digest
from bcep_joint.common import read,write,save
from ebjd.metrics import MetricAccumulator,hard_metric_statistics
from joint_flow_fourfold.report import csvwrite
from .assets import RUN,load_teacher
from .model import ResidualHead
from .inference import generate_student
from .train import load_records,apply_head
from .selection import evaluate_gate

@torch.no_grad()
def replay(fold,ablation=False):
 folder=RUN/fold/('control' if ablation else 'head');result=read(folder/'RESULT.json')
 inputs,model,provider,dep,identity=load_teacher(fold,'cuda')
 assert identity==read(RUN/fold/'cache/IDENTITY.json')
 head=ResidualHead();saved=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
 assert digest(result['checkpoint'])==result['checkpoint_sha256'];head.load_state_dict(saved['model']);head.eval()
 output=folder/'observed_replay';output.mkdir(exist_ok=True)
 cached=load_records(fold,'selection_dev')
 # Labels stay in the scoring record; inference never receives that record.
 rows=[];t0=time.monotonic()
 for pos,index in enumerate(inputs.split['selection_dev_indices']):
  p=output/f'{pos:05d}.pt';side=p.with_suffix('.json')
  if side.exists():
   row=read(side);assert row['sha256']==digest(p)
  else:
   batch=inputs.data.batch([index],'selection_dev',truth=False).to('cuda')
   torch.cuda.synchronize();tick=time.monotonic();student,detail=generate_student(model,provider,dep,head,batch,seed=2035,ablation=ablation)
   torch.cuda.synchronize();seconds=time.monotonic()-tick
   record=cached[pos];f=record['features']
   assert torch.equal(detail['raw_ids'].cpu(),f['raw_ids'])
   assert torch.equal(detail['B'].cpu(),f['B']),'teacher replay must equal the same-device cached forward'
   expected=apply_head(head,record,ablation)
   assert torch.equal(student.cpu(),expected),'same CPU feature/head replay must be exact'
   save(dict(B=detail['B'].cpu(),student=student.cpu()),p)
   row=dict(window=f'dev_{pos+1:05d}',n=int(batch.valid.sum()),sha256=digest(p),seconds=seconds,raw_ids_exact=True,B_exact=True,student_exact=True,
    teacher_seconds=detail['teacher_seconds'],feature_seconds=detail['feature_seconds'],head_seconds=detail['head_seconds'])
   write(side,row)
  rows.append(row)
  if (pos+1)%25==0:write(RUN/fold/'PROGRESS.json',dict(stage='ACTUAL_OBSERVED_REPLAY',windows=pos+1,total=len(cached)))
 write(output/'PREDICTIONS_FROZEN.json',dict(rows=rows,seal_unix=time.time(),checkpoint_sha256=result['checkpoint_sha256']))
 acc={b:{g:MetricAccumulator() for g in ('all','N_ge_9')} for b in ('B','student')};public=[]
 for pos,row in enumerate(rows):
  p=output/f'{pos:05d}.pt';assert digest(p)==row['sha256'];values=torch.load(p,weights_only=False);record=cached[pos];f=record['features']
  out=dict(window=row['window'],n=row['n'])
  for b in values:
   stats=hard_metric_statistics(values[b][None],record['truth'][None],f['valid'][None]);acc[b]['all'].update_statistics(stats)
   if row['n']>=9:acc[b]['N_ge_9'].update_statistics(stats)
   out.update({b+'_JADE':float(stats.scene_jade[0]),b+'_JFDE':float(stats.scene_jfde[0]),
    b+'_minADE_sum':float(stats.agent_minade[stats.agent_valid].double().sum()),b+'_minFDE_sum':float(stats.agent_minfde[stats.agent_valid].double().sum())})
  out.update({f'student_minus_B_{q}_cm':100*(out['student_'+q]-out['B_'+q]) for q in ('JADE','JFDE')})
  for q in ('minADE','minFDE'):out[f'student_minus_B_{q}_cm']=100*(out['student_'+q+'_sum']-out['B_'+q+'_sum'])/row['n']
  public.append(out)
 metrics={b:{g:a.as_dict() if a.scene_count else None for g,a in groups.items()} for b,groups in acc.items()}
 assert metrics['B']==result['baseline'] and metrics['student']==result['metrics']
 gate=evaluate_gate(metrics['B'],metrics['student']);assert gate==result['gate']
 csvwrite(folder/'PER_WINDOW.csv',public)
 receipt=dict(exact_cache_reconciliation=True,real_observed_windows=len(rows),metrics=metrics,gate=gate,wall_seconds=time.monotonic()-t0,
  inference_seconds=sum(r['seconds'] for r in rows),online_extra_mean_seconds=sum(r['feature_seconds']+r['head_seconds'] for r in rows)/len(rows),
  each_online_DDIM_calls=1,upstream_frozen=True,training_cache_not_used_by_inference=True,test_access=0)
 write(folder/'REPLAY.json',receipt);return receipt
