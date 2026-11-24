"""Fixed 30-epoch, logical-window batches; complete atomic recovery snapshots."""
import copy,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from bcep.contracts import digest,canonical_hash
from bcep_joint.common import read,write,save,seed_all
from ebjd.metrics import MetricAccumulator,hard_metric_statistics
from joint_flow_fourfold.inputs import state_hash
from joint_flow_fourfold.report import csvwrite
from .assets import RUN
from .cache import immutable
from .model import ResidualHead
from .objective import metric_sums,paired_objective
from .selection import evaluate_gate,rank_epoch

RECIPE=dict(seed=3201,lr=3e-4,weight_decay=1e-4,logical_scenes=32,clip=1.,epochs=30,precision='FP32',scheduler=None)
Q=('minADE','minFDE','JADE','JFDE')

def load_records(fold,purpose):
 root=RUN/fold;cache=root/'cache'/purpose;rows=read(cache/'COMPLETE.json')['rows'];records=[]
 for pos,row in enumerate(rows):
  p=cache/row['file'];lp=root/'labels'/purpose/f'{pos:05d}.pt';label=read(lp.with_suffix('.json'))
  assert digest(p)==row['sha256'] and digest(lp)==label['sha256'] and label['prediction_sha256']==row['sha256']
  features=torch.load(p,map_location='cpu',weights_only=False)
  truth=torch.load(lp,map_location='cpu',weights_only=False)['truth']
  records.append(dict(features=features,truth=truth,row=row))
 return records

def apply_head(head,record,ablation):
 f=record['features'];return head(f['B'],f['base_features'],f['difference_features'],f['valid'],ablation=ablation)

@torch.no_grad()
def evaluate(head,records,ablation):
 head.eval();acc={g:MetricAccumulator() for g in ('all','N_ge_9')};rows=[]
 for r in records:
  y=apply_head(head,r,ablation);f=r['features'];st=hard_metric_statistics(y[None],r['truth'][None],f['valid'][None])
  acc['all'].update_statistics(st)
  if r['row']['n']>=9:acc['N_ge_9'].update_statistics(st)
  rows.append(dict(window=r['row']['window'],n=r['row']['n'],JADE=float(st.scene_jade[0]),JFDE=float(st.scene_jfde[0]),
   minADE_sum=float(st.agent_minade[st.agent_valid].double().sum()),minFDE_sum=float(st.agent_minfde[st.agent_valid].double().sum())))
 return {g:a.as_dict() if a.scene_count else None for g,a in acc.items()},rows

def rng_state(order_rng):
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),order=order_rng.get_state())

def restore_rng(rng,order_rng):
 random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);order_rng.set_state(rng['order'])

def persist(folder,state,head,opt,order_rng,completed=False):
 state.update(model=head.state_dict(),optimizer=opt.state_dict(),scheduler=None,rng=rng_state(order_rng))
 if completed:
  path=folder/f"epoch_{state['completed_epoch']:03d}.pt";save(state,path)
  for alias,target in [('last.pt',path),('best.pt',folder/min(state['history'],key=rank_epoch)['file'])]:
   tmp=folder/(alias+'.link');tmp.unlink(missing_ok=True);os.link(target,tmp);tmp.replace(folder/alias)
   # POSIX rename is a no-op when both names already reference one inode.
   tmp.unlink(missing_ok=True)
  ranking=sorted(state['history'],key=rank_epoch)[:5]
  write(folder/'TOP5.json',[dict(epoch=r['epoch'],file=r['file'],selection_key=rank_epoch(r),metrics=r['metrics'],gate=r['gate'],sha256=digest(folder/r['file'])) for r in ranking])
  write(folder/'EPOCH_METRICS.json',state['history'])
 else:save(state,folder/'last.pt')

def real_preflight(head,train,scales,folder):
 """Real tiny training batch; disposable clone, original initialization unchanged."""
 chosen=list(dict.fromkeys([min(range(len(train)),key=lambda i:train[i]['row']['n']),
  next(i for i,r in enumerate(train) if r['row']['n']>1),max(range(len(train)),key=lambda i:train[i]['row']['n'])]))
 clone=copy.deepcopy(head);optimizer=torch.optim.AdamW(clone.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
 parts=[];teachers=[]
 for i in chosen:
  r=train[i];f=r['features'];y=apply_head(clone,r,False);assert torch.equal(y,f['B'])
  parts.append(metric_sums(y[None],r['truth'][None],f['valid'][None]));teachers.append(metric_sums(f['B'][None],r['truth'][None],f['valid'][None]))
 paired_objective(parts,teachers,scales)['loss'].backward()
 gradients={k:float(p.grad.norm()) for k,p in clone.named_parameters() if p.grad is not None}
 assert gradients['common_readout.2.weight']>0 and gradients['person_readout.2.weight']>0
 torch.nn.utils.clip_grad_norm_(clone.parameters(),1.);optimizer.step()
 permutation_checks=[]
 with torch.no_grad():
  for i in chosen:
   r=train[i];f=r['features'];reference,detail=clone(f['B'],f['base_features'],f['difference_features'],f['valid'],return_details=True)
   repeat=apply_head(clone,r,False);assert torch.equal(reference,repeat)
   slots=torch.arange(f['B'].shape[0]-1,-1,-1);people=torch.arange(f['B'].shape[1]-1,-1,-1)
   permuted=clone(f['B'][slots][:,people],f['base_features'][slots][:,people],
    f['difference_features'][slots][:,people],f['valid'][people])
   expected=reference[slots][:,people]
   torch.testing.assert_close(permuted,expected,atol=1e-6,rtol=1e-6)
   centered=detail['relative'][:,f['valid']].sum(1)
   torch.testing.assert_close(centered,torch.zeros_like(centered),atol=1e-6,rtol=0)
   permutation_checks.append(dict(window=r['row']['window'],n=r['row']['n'],
    identical_repeat_max_abs_m=float((reference-repeat).abs().max()),
    person_and_slot_permutation_max_abs_m=float((permuted-expected).abs().max()),
    relative_displacement_sum_max_abs_m=float(centered.abs().max())))
 recovery=folder/'preflight_recovery.pt';save(dict(model=clone.state_dict(),optimizer=optimizer.state_dict()),recovery)
 duplicate=copy.deepcopy(head);duplicate_opt=torch.optim.AdamW(duplicate.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
 saved=torch.load(recovery,weights_only=False);duplicate.load_state_dict(saved['model']);duplicate_opt.load_state_dict(saved['optimizer'])
 for h,o in ((clone,optimizer),(duplicate,duplicate_opt)):
  o.zero_grad(set_to_none=True);parts=[]
  for i in chosen:
   r=train[i];f=r['features'];parts.append(metric_sums(apply_head(h,r,False)[None],r['truth'][None],f['valid'][None]))
  paired_objective(parts,teachers,scales)['loss'].backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);o.step()
 assert state_hash(clone)==state_hash(duplicate)
 write(folder/'PREFLIGHT.json',dict(real_train_windows=[train[i]['row']['window'] for i in chosen],
  zero_head_exact_B=True,readout_gradient_norms=gradients,optimizer_resume_next_update_exact=True,
  real_window_permutation_checks=permutation_checks,permutation_atol_m=1e-6,permutation_rtol=1e-6,
  disposable_check_models=True,formal_training_updates=0))

def train(fold,ablation=False):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed_all(3201)
 folder=RUN/fold/('control' if ablation else 'head');folder.mkdir(parents=True,exist_ok=True)
 teacher_identity=read(RUN/fold/'cache/IDENTITY.json');baseline=read(RUN/fold/'cache/BASELINE_METRICS.json')['selection_dev']
 scales=read(RUN/fold/'cache/TRAIN_SCALES.json')
 load_start=time.monotonic();training=load_records(fold,'train');dev=load_records(fold,'selection_dev');load_seconds=time.monotonic()-load_start
 head=ResidualHead();order_rng=torch.Generator().manual_seed(3201)
 config=dict(recipe=RECIPE,fold=fold,ablation=ablation,teacher_identity=teacher_identity,cache_sha256={p:digest(RUN/fold/'cache'/p/'COMPLETE.json') for p in ('train','selection_dev')},
  scales=scales,parameters=head.parameter_count,initial_model_sha256=state_hash(head),selector='passed: minFDE,JFDE,epoch; otherwise maximum six-condition violation,minFDE,epoch',
  retention='all completed epochs plus official Top5, best and last; complete optimizer, RNG, order and counters',device='cpu',input_normalization='none',upstream_frozen=True,
  native_disabled_optimizations=os.environ.get('RSJG_DISABLE_OPTIMIZATIONS',''))
 immutable(folder/'RUN_CONFIG.json',config)
 if ablation:assert read(RUN/fold/'head/RUN_CONFIG.json')['initial_model_sha256']==config['initial_model_sha256']
 if not (folder/'PREFLIGHT.json').exists():real_preflight(head,training,scales,folder)
 opt=torch.optim.AdamW(head.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
 teacher_parts=[metric_sums(r['features']['B'][None],r['truth'][None],r['features']['valid'][None]) for r in training]
 state=dict(schema='residual-complete-resume-v1',recipe=RECIPE,teacher_identity=teacher_identity,config_sha256=digest(folder/'RUN_CONFIG.json'),
  ablation=ablation,native_disabled_optimizations=config['native_disabled_optimizations'],epoch=1,completed_epoch=0,update=0,cursor=0,order=None,history=[],epoch_train_seconds=0.,epoch_loss_sum=0.,epoch_updates=0,wall_seconds=0.)
 if (folder/'last.pt').exists():
  state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
  assert state['config_sha256']==digest(folder/'RUN_CONFIG.json')
  head.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'],order_rng)
 while state['completed_epoch']<30:
  if state['order'] is None:state['order']=torch.randperm(len(training),generator=order_rng).tolist()
  head.train();tick=time.monotonic()
  group=state['order'][state['cursor']:state['cursor']+32];opt.zero_grad(set_to_none=True)
  # Ragged single-window physical microbatches share one graph and whole logical-batch denominator/max.
  parts=[]
  for i in group:
   r=training[i];parts.append(metric_sums(apply_head(head,r,ablation)[None],r['truth'][None],r['features']['valid'][None]))
  objective=paired_objective(parts,[teacher_parts[i] for i in group],scales)
  objective['loss'].backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
  state['cursor']+=len(group);state['update']+=1;state['epoch_updates']+=1
  state['epoch_loss_sum']+=float(objective['loss'].detach());state['epoch_train_seconds']+=time.monotonic()-tick
  if state['cursor']==len(training):
   tick=time.monotonic();metrics,rows=evaluate(head,dev,ablation);validation_seconds=time.monotonic()-tick
   gate=evaluate_gate(baseline,metrics);epoch=state['epoch']
   record=dict(epoch=epoch,file=f'epoch_{epoch:03d}.pt',metrics=metrics,gate=gate,train_seconds=state['epoch_train_seconds'],validation_seconds=validation_seconds,
    train_loss=state['epoch_loss_sum']/state['epoch_updates'],ddim_calls=0)
   state['history'].append(record);state['wall_seconds']+=state['epoch_train_seconds']+validation_seconds
   state.update(epoch=epoch+1,completed_epoch=epoch,cursor=0,order=None,epoch_train_seconds=0.,epoch_loss_sum=0.,epoch_updates=0)
   csvwrite(folder/f'epoch_{epoch:03d}_windows.csv',rows);persist(folder,state,head,opt,order_rng,True)
   write(folder/'PROGRESS.json',dict(epoch=epoch,gate=gate,metrics=metrics,unix=time.time()));print(fold,'control' if ablation else 'head',epoch,gate,flush=True)
  else:persist(folder,state,head,opt,order_rng)
 best=min(state['history'],key=rank_epoch);path=folder/best['file']
 # Validate actual writer retention and complete resumability before claiming completion.
 top=read(folder/'TOP5.json');assert len(top)==5 and len(set(r['epoch'] for r in top))==5
 for row in state['history']:
  s=torch.load(folder/row['file'],map_location='cpu',weights_only=False)
  assert s['completed_epoch']==row['epoch'] and s['cursor']==0 and s['order'] is None
  assert all(k in s for k in ('optimizer','scheduler','rng','history','config_sha256'))
 assert digest(folder/'best.pt')==digest(path)
 result=dict(completed_epochs=30,rank1_epoch=best['epoch'],checkpoint=str(path),checkpoint_sha256=digest(path),metrics=best['metrics'],gate=best['gate'],
  baseline=baseline,parameters=head.parameter_count,training_updates=state['update'],optimizer_only_head=True,
  timing=dict(cache_load_seconds=load_seconds,train_update_seconds=sum(r['train_seconds'] for r in state['history']),validation_seconds=sum(r['validation_seconds'] for r in state['history']),
   measured_compute_seconds=state['wall_seconds'],cold_cache_seconds=read(RUN/fold/'cache/TIMING.json')['wall_seconds']),
  retained_completed_epochs=30,top5_epochs=[r['epoch'] for r in top],test_access=0,ablation=ablation)
 write(folder/'RESULT.json',result);return result
