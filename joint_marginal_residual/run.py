"""Durable gated dev experiment; benchmark and original training have separate state."""
import argparse,fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
from bcep.contracts import digest
from bcep_joint.common import read,write
from .assets import SOURCE,ROOT,RUN,FOUR,BASE,BASE_SOURCE,load_inputs
from .cache import immutable

BRANCH='experiment/joint-marginal-residual-20261124'
DOC=SOURCE/'docs/research/joint_marginal_residual/20261124'
PACKAGES=('joint_marginal_residual','bcep_v2','bcep_joint','bcep_formal','bcep','ebjd','joint_flow_dependency','joint_flow_fourfold','scene_pair_selector')

def sources():return {str(p.relative_to(SOURCE)):digest(p) for name in PACKAGES for p in sorted((SOURCE/name).rglob('*.py'))}

def protect():
 config=read(FOUR/'RUN_CONFIG.json');original=ROOT/'RSJG_joint_flow_fourfold_20261122'
 for p,sha in config['source_sha256'].items():assert digest(original/p)==sha,'running controller source changed'
 for key,relative in [('old_queue_status','bcep_ebjd_v2_integrated/STATUS.json'),('old_queue_contract','bcep_ebjd_v2_integrated/FORMAL_RUN.json'),
  ('old_queue_authorization','bcep_ebjd_v2_integrated/AUTHORIZED.yaml'),('SSEC','zara2_ssec_energy_stage2_20261121/SUPERVISOR_STATUS.json')]:
  assert digest(ROOT/'RSJG_EBJD_assets/experiments'/relative)==config['protected_sha256'][key]

def prepare():
 RUN.mkdir(parents=True,exist_ok=True);protect()
 plan={}
 for fold in ('zara1','zara2','hotel'):
  inputs,checkpoint,identity=load_inputs(fold)
  counts={p:[len(inputs.split['members'][i]['agent_ids']) for i in inputs.split[p+'_indices']] for p in ('train','selection_dev')}
  # Payload floats per person: B480 + base3080 + dedicated100 + history16, plus masks, raw IDs and file metadata.
  plan[fold]=dict(identity=identity,counts={p:dict(windows=len(ns),agents=sum(ns),dense_windows=sum(n>=9 for n in ns)) for p,ns in counts.items()},
   estimated_cache_payload_bytes=sum(sum(ns)*(480+3080+100+16)*4+len(ns)*4096 for ns in counts.values()))
 config=dict(base_commit=BASE,plan=plan,source_sha256=sources(),train_seed=3201,train_inference_seed=2036,dev_inference_seed=2035,
  M=256,K=20,DDIM=20,eta=0.,prediction_frames=12,precision='FP32',head_width=64,
  recipe=dict(epochs=30,optimizer='AdamW',lr=3e-4,weight_decay=1e-4,logical_windows=32,clip=1,scheduler=None),
  loss='mean_q((student_q-B_q)/train_B_scale_q)+amax_q((student_q-B_q)/train_B_scale_q)',
  gates=dict(all_minFDE_le=-.020,all_JFDE_le=-.005,all_minADE_lt=0.,all_JADE_lt=0.,dense_JFDE_le=0.,dense_minFDE_le=0.),
  selector='one epoch: passing minFDE,JFDE,earlier; none passing max excess meters,minFDE,earlier; exact strict checks',
  expansion='zara1 fail stops model route; pass runs zara2/hotel separately and only zara1 equal-parameter P=J,a=0 control; all3pass then released UNIV/ETH',
  control='same architecture, width, initial seed/state, teacher cache, loss, budget and selector; zero dedicated P-J,a only',
  GPU_policy='user follow-up explicitly authorizes concurrent GPU; one auxiliary worker, allocator cap40%, wait if free memory below6GiB; original controller untouched',
  GPU_authorization_amendment='2026-11-24 user: 直接可以并行吧; supersedes idle-only queue in initial execution file',
  retain='all completed epoch snapshots+official Top5+best+last, complete RNG/optimizer/order/counters, atomic writes, no deletions',
  seen_test_baselines='ZARA2/ZARA1/HOTEL test baseline reports were previously viewed; this run uses only own-fold train/selection_dev, zero test access',
  benchmark='real ZARA1 native four-stage fixed logical fixture; >=3 paired warm medians, same precision/device; full-old-training speed not measured',
  original_controller_config_sha256=digest(FOUR/'RUN_CONFIG.json'))
 immutable(RUN/'RUN_CONFIG.json',config);return config

def status(value,**kwargs):write(RUN/'STATUS.json',dict(status=value,unix=time.time(),**kwargs));print(value,kwargs,flush=True)

def wait_gpu():
 while True:
  protect()
  available=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
  if available>=6144:return
  status('WAITING_FOR_SHARED_GPU_HEADROOM',available_MiB=available,minimum_MiB=6144)
  time.sleep(45)

def worker(stage,fold,ablation=False):
 import torch
 torch.set_num_threads(2)
 if stage in ('cache','replay'):
  from bcep_v2.runtime import configure_cuda
  configure_cuda()
  torch.cuda.set_per_process_memory_fraction(.4)
 if stage=='cache':
  from .cache import build
  build(fold)
 elif stage=='train':
  from .train import train
  train(fold,ablation)
 elif stage=='replay':
  from .replay import replay
  replay(fold,ablation)

def execute(stage,fold='',ablation=False):
 name=f'{fold}_{stage}'+('_control' if ablation else '')
 log=RUN/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
 if stage=='benchmark':command=[sys.executable,'-m','joint_marginal_residual.benchmark','--output',str(RUN/'benchmark'),'--device','cuda','--baseline-source',str(BASE_SOURCE)]
 else:command=[sys.executable,'-m','joint_marginal_residual.run','--worker',stage,'--fold',fold]+(['--ablation'] if ablation else [])
 for attempt in range(1,4):
  if stage in ('benchmark','cache','replay'):wait_gpu()
  assert sources()==read(RUN/'RUN_CONFIG.json')['source_sha256'],'own bound source changed'
  status('RUNNING',stage=stage,fold=fold,ablation=ablation,attempt=attempt)
  started=time.time()
  env=os.environ.copy()
  if stage!='benchmark':
   resolved=read(RUN/'benchmark/RESULT.json')['disabled_optimizations']
   env['RSJG_DISABLE_OPTIMIZATIONS']=','.join(resolved)
  with log.open('a',buffering=1) as handle:
   process=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=handle,stderr=subprocess.STDOUT)
   write(RUN/'ACTIVE_CHILD.json',dict(pid=process.pid,stage=stage,fold=fold,ablation=ablation))
   code=process.wait()
  timing_path=RUN/'STAGE_WALL_TIMES.json';timings=read(timing_path) if timing_path.exists() else []
  timings.append(dict(stage=stage,fold=fold,ablation=ablation,attempt=attempt,start_unix=started,end_unix=time.time(),wall_seconds=time.time()-started,exit_code=code))
  write(timing_path,timings)
  if code==0:return
  if (RUN/'benchmark/REJECTED.json').exists():raise RuntimeError('native optimization rejected; preserve fixed tolerance and inspect named operation before cache')
  if attempt==3:raise RuntimeError(name+' failed same-run recovery attempts')
  time.sleep(15)

def run_fold(fold,ablation=False):
 folder=RUN/fold/('control' if ablation else 'head')
 if not (RUN/fold/'cache/TIMING.json').exists():execute('cache',fold)
 if not (folder/'RESULT.json').exists():execute('train',fold,ablation)
 if not (folder/'REPLAY.json').exists():execute('replay',fold,ablation)
 return read(folder/'RESULT.json')

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',choices=('cache','train','replay'))
 p.add_argument('--fold');p.add_argument('--ablation',action='store_true');p.add_argument('--push-on-complete',action='store_true');args=p.parse_args()
 if args.worker:worker(args.worker,args.fold,args.ablation);return
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);config=prepare()
  if args.prepare_only:status('PREPARED',plan=config['plan']);return
  if not (RUN/'benchmark/RESULT.json').exists():execute('benchmark')
  results={'zara1':run_fold('zara1')}
  if results['zara1']['gate']['passed']:
   results['zara1_control']=run_fold('zara1',True)
   for fold in ('zara2','hotel'):results[fold]=run_fold(fold)
   if all(results[f]['gate']['passed'] for f in ('zara1','zara2','hotel')):
    for fold in ('univ','eth'):
     while not (FOUR/fold/'B_FROZEN.json').exists():status('WAITING_FOR_READONLY_UPSTREAM',fold=fold);time.sleep(45)
     results[fold]=run_fold(fold)
  from .report import build_report
  build_report(results,config);protect()
  if args.push_on_complete:
   assert subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip()==BRANCH
   subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],cwd=SOURCE,check=True)
   subprocess.run(['git','diff','--cached','--check'],cwd=SOURCE,check=True)
   if subprocess.call(['git','diff','--cached','--quiet'],cwd=SOURCE):subprocess.run(['git','commit','-m','Archive fixed dual-assignment residual experiment and native timings'],cwd=SOURCE,check=True)
   subprocess.run(['git','push','-u','origin',BRANCH],cwd=SOURCE,check=True)
   head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
   remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=SOURCE,text=True).split()[0];assert head==remote
   write(RUN/'DELIVERY.json',dict(sha=head,report=f'https://github.com/WuYanXingege/RSJG/blob/{head}/{DOC.relative_to(SOURCE)}/README.md'))
  status('COMPLETE',model_zara1_pass=results['zara1']['gate']['passed'],trained=list(results))

if __name__=='__main__':
 try:main()
 except BaseException:status('INTERRUPTED_RESUME_SAME_TASK',traceback=traceback.format_exc());raise
