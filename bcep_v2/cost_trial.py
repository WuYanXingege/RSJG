"""Single authorized ZARA2/joint trial; never starts the five-fold controller."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
import torch
from bcep.contracts import digest,canonical_hash
from bcep_joint.common import read,write,seed_all
from bcep_formal.data import FormalData
from bcep_formal.control import install
from bcep_v2.bank import Bank
from bcep_v2.joint import build_joint
from bcep_v2.workflow import recipe
from bcep_v2.state import run_stage
from bcep_v2.stages import joint_callbacks
from bcep_v2.runtime import configure_cuda
from bcep_v2.formal_manifest import write_csv
from bcep_joint.inference import generate
from ebjd.unified import gather_candidates
from ebjd.metrics import MetricAccumulator

SOURCE=Path(__file__).resolve().parents[1]
BASE='468c5eddde48bdeec47c4bb3c39418352be16bb9'

def inputs(upstream):
 run=read(upstream/'FORMAL_RUN.json');assert run['commit']==BASE and read(upstream/'STATUS.json')['status']=='PAUSED_RESUMABLE'
 cfg=read(upstream/'zara2/CONFIG.json');plan=read(run['input_plan']);desc=plan['folds']['zara2']['split'];assert digest(desc['path'])==desc['sha256'];split=read(desc['path'])
 assert len(split['train_indices'])==2681 and len(split['selection_dev_indices'])==590
 assert sum(len(split['members'][i]['agent_ids'])>=9 for i in split['selection_dev_indices'])==127
 assert sum(len(split['members'][i]['agent_ids']) for i in split['selection_dev_indices'])==4262
 for i in split['train_indices']+split['selection_dev_indices']:assert digest(split['records'][i]['path'])==split['records'][i]['sha256']
 banks=[]
 for purpose in ['train','selection_dev']:
  folder=upstream/'zara2/refined_banks'/f'{purpose}_2035';bid=read(folder/'IDENTITY.json');assert bid['purpose']==purpose and bid['seed']==2035 and bid['unit']=='m' and bid['K']==20
  assert bid['base']==read(upstream/'zara2/density/FROZEN.json') and bid['head']==read(upstream/'zara2/head/FROZEN.json')
  banks.append(Bank(folder,bid,split))
 for stage,ep in [('density',19),('head',9)]:
  rows=read(upstream/f'zara2/{stage}/TOP5.json');assert rows[0]['epoch']==ep and digest(upstream/f'zara2/{stage}'/rows[0]['file'])==rows[0]['sha256']
 config=dict(cfg['joint'],scales=split['bridge_scales'],pair_backward_proxy='candidate_cost')
 rec=recipe(cfg,'joint');assert rec==dict(seed=3201,lr=.0001,weight_decay=.0001,clip=5.,logical_scenes=16,max_epochs=100,minimum=30,patience=15,scheduler_horizon=100)
 return config,rec,split,banks

def tensor_hash(model):
 h=hashlib.sha256()
 for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def build(config,device):
 seed_all(3201);return build_joint(config).to(device)

def validation(model,data,bank,out,start_epoch):
 epoch=start_epoch
 def validate():
  nonlocal epoch
  train_peak=torch.cuda.max_memory_allocated();torch.cuda.reset_peak_memory_stats()
  acc={s:MetricAccumulator() for s in ['all','N_ge_9']};rows=[];started=time.monotonic()
  for i in data.split['selection_dev_indices']:
   batch=data.batch([i],'selection_dev');g=bank.batch([i],batch).cuda();x=batch.to('cuda');n=int(x.valid.sum())
   _,h,pi,_=generate(model,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035,mode='H')
   initial=gather_candidates(g,model.coupling.last_initial_indices)
   def jf(v):return float((v-x.future[:,None,:,-1]).norm(dim=-1).mean(-1).min())
   one=MetricAccumulator();one.update(h,x.future,x.valid);m=one.as_dict()
   row=dict(epoch=epoch,index=i,n_agents=n,G_JFDE=jf(g),initial_JFDE=jf(initial),H_JFDE=m['JFDE'],H_JADE=m['JADE'],minADE=m['minADE'],minFDE=m['minFDE'],oracle=m['scene_weighted_minFDE'])
   rows.append(row);acc['all'].update(h,x.future,x.valid)
   if n>=9:acc['N_ge_9'].update(h,x.future,x.valid)
  groups=[]
  for name,a in acc.items():
   rr=[r for r in rows if name=='all' or r['n_agents']>=9]
   groups.append(dict(epoch=epoch,group=name,**a.as_dict(),G_JFDE=sum(r['G_JFDE'] for r in rr)/len(rr),initial_JFDE=sum(r['initial_JFDE'] for r in rr)/len(rr)))
  validate.metrics=acc['all'].as_dict();assert abs(validate.metrics['minFDE']-.3520422351797007)<1e-6 and abs(validate.metrics['scene_weighted_minFDE']-.33942601943420153)<1e-6
  write_csv(out/'dev'/f'epoch_{epoch:03d}_windows.csv',rows);write(out/'dev'/f'epoch_{epoch:03d}_groups.json',groups)
  write(out/'dev'/f'epoch_{epoch:03d}_cost.json',dict(train_peak_bytes=train_peak,validation_peak_bytes=torch.cuda.max_memory_allocated(),validation_seconds=time.monotonic()-started))
  epoch+=1;torch.cuda.reset_peak_memory_stats();return validate.metrics['JADE']
 return validate

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 configure_cuda();torch.set_num_threads(1)
 config,rec,split,banks=inputs(a.upstream)
 impl={str(p.relative_to(SOURCE)):digest(p) for p in sorted(SOURCE.glob('bcep_v2/*.py'))}
 identity=dict(stage='joint',fold='zara2',experiment='zara2_costproxy_stage2_20261119',base_source=BASE,pair_backward_proxy='candidate_cost',implementation=canonical_hash(impl),config=canonical_hash(config),train_bank=banks[0].hash,dev_bank=banks[1].hash,split=canonical_hash(split))
 frozen=dict(identity=identity,config=config,recipe=rec,implementation_files=impl,absolute_update_cap=16800,retention='all completed epochs plus dev H-JADE Top5 (ties earlier), best and last',selection='dev_H_JADE',authorization='only ZARA2 joint, one seed 3201; no test; no other folds',decision=dict(reference_rank1=[.4084707879413993,.8016974424154072,1.3188941478729248],reference_epoch5=[.4097670185111337,.7841791655426308,1.296843430188697],JADE_context_tolerance_m=.0012962305697344,require_both_JFDE_improve=True,require_exceed_epoch5_tradeoff=True))
 a.out.mkdir(parents=True,exist_ok=True)
 if a.command=='prepare':
  assert not (a.out/'FROZEN_CONFIG.json').exists();write(a.out/'FROZEN_CONFIG.json',frozen);write(a.out/'STATUS.json',dict(status='PREPARED',optimizer_updates=0));return
 assert read(a.out/'FROZEN_CONFIG.json')==frozen and read(a.out/'ACCEPTANCE.json')['status']=='PASS'
 assert read(a.out/'ACCEPTANCE.json')['implementation']==identity['implementation']
 with (a.out/'trial.lock').open('a') as lock,(a.upstream/'controller.lock').open('r') as oldlock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  install();model=build(config,'cuda');model.coupling.capture_assignments=True
  init=tensor_hash(model);assert init==read(a.out/'ACCEPTANCE.json')['initial_tensor_sha256']
  data=FormalData(split,'training',verify=False);backward,_=joint_callbacks(model,data,torch.device('cuda'),config,*banks)
  epoch=1
  if a.resume:epoch=torch.load(a.out/'joint/last.pt',map_location='cpu',weights_only=False)['epoch']
  validate=validation(model,data,banks[1],a.out,epoch);write(a.out/'STATUS.json',dict(status='RUNNING',pid=os.getpid(),initial_tensor_sha256=init,identity=identity))
  try:
   state=run_stage(a.out/'joint',identity,rec,model,split['train_indices'],backward,validate,16800,resume=a.resume,record_progress=True)
   write(a.out/'STATUS.json',dict(status=state['status'],optimizer_updates=state['update'],completed_epoch=state['completed_epoch'],stop_reason=state['stop_reason']))
  except BaseException as e:
   write(a.out/'STATUS.json',dict(status='FAILED',reason=repr(e)));raise
if __name__=='__main__':main()
