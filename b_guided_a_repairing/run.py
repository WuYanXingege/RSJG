"""Frozen two-fold B-guided permutation of native A/H0 full trajectories."""
import csv,fcntl,json,os,subprocess,time,traceback
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
import numpy as np
import scipy
import torch
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import read,write
from bcep_joint.inference import generate as generate_a
from bcep_formal.data import FormalData
from ebjd.metrics import BatchMetricStatistics,MetricAccumulator,hard_metric_statistics
from whole_scene_swap.run import assets
from joint_flow_fourfold.inputs import state_hash
from joint_flow_fourfold.controller import check_protection
from .matching import repair_trajectories,joint_criterion

SOURCE=Path(__file__).resolve().parents[1];ROOT=SOURCE.parent
RUN=ROOT/'RSJG_EBJD_assets/experiments/zara1_zara2_b_guided_a_repairing_20261123'
DOC=SOURCE/'docs/research/b_guided_a_repairing/zara1_zara2_20261123'
SWAP=ROOT/'RSJG_EBJD_assets/experiments/zara1_zara2_whole_scene_swap_20261123'
FOUR=ROOT/'RSJG_EBJD_assets/experiments/joint_flow_fourfold_confirmation_20261122'
BASE='0165436a36440d457186e3f62da2e6f4afa8ff70'
FOLDS=('zara1','zara2');GROUPS=('all','N_ge_9');BRANCHES=('A','B','R')
METRICS=('JADE','JFDE','minADE','minFDE')
EXPECTED={
 'zara1':dict(A_epoch=15,A_sha256='bf85c688ec1fb96bbcea3de1384f54e9f5abd40c91be38294fe3cd11c0eb210f',B_epoch=6,B_sha256='fc7d9cca99769b12f2366eda20396130f547534248c82451e1041d640ff8389d',windows=671,dense_windows=181),
 'zara2':dict(A_epoch=5,A_sha256='9281673fda5c439bdd463b0bfc9af1a65a3a0fea30941ad3c956cbb617943317',B_epoch=35,B_sha256='ea13cd7fd99a02c44c3c9af9a14a0cc9b5166b629e5c338694cca97f1c8c502a',windows=590,dense_windows=127)}


def immutable(path,value):
 if path.exists():assert read(path)==value,'immutable identity changed: '+str(path)
 else:write(path,value)


def csvwrite(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 tmp.replace(path)


def archive_json(relative,commit=BASE):
 return json.loads(subprocess.check_output(['git','show',commit+':'+relative],cwd=SOURCE))


def source_hashes():
 return {str(p.relative_to(SOURCE)):digest(p) for folder in ('b_guided_a_repairing','whole_scene_swap','joint_flow_dependency','scene_pair_selector','joint_flow_fourfold','bcep_v2','bcep_joint','bcep_formal','bcep','ebjd','bcep_confirmation','selector_control') for p in sorted((SOURCE/folder).rglob('*.py'))}


def fold_assets(fold):
 inputs,model,provider,dep,prior_identity,result=assets(fold)
 expected=EXPECTED[fold]
 assert prior_identity['checkpoint_sha256']==expected['B_sha256'] and prior_identity['epoch']==expected['B_epoch']
 assert prior_identity['upstream']['checkpoint_sha256']==expected['A_sha256']
 fixed=archive_json(f'docs/research/whole_scene_swap/zara1_zara2_20261123/{fold}/IDENTITY.json')
 assert prior_identity==fixed==read(SWAP/fold/'IDENTITY.json')
 prior=read(SWAP/fold/'PREDICTIONS_FROZEN.json')
 assert prior['identity']==prior_identity and len(prior['rows'])==expected['windows']
 oldcfg=read(SWAP/'RUN_CONFIG.json')
 assert oldcfg['seed']==2035 and oldcfg['M']==256 and oldcfg['K']==20 and oldcfg['DDIM']==20 and not oldcfg['calibration']
 assert oldcfg['training_updates']==oldcfg['test_access']==0
 assert read(SWAP/'STATUS.json')['status']=='COMPLETE'
 # Only the original B array is reused; the former swap output is never used.
 identity=dict(fold=fold,models=expected,original_B_identity=prior_identity,
  original_B_seal_sha256=digest(SWAP/fold/'PREDICTIONS_FROZEN.json'),
  original_B_config_sha256=digest(SWAP/'RUN_CONFIG.json'),A_dev_bank_sha256=inputs.banks['selection_dev'].hash,
  member_sequence_sha256=canonical_hash([inputs.split['members'][i] for i in inputs.split['selection_dev_indices']]))
 inputs.output=RUN/fold
 del provider,dep,result
 return inputs,model,identity,prior


@contextmanager
def observations_only():
 original=FormalData.item
 def guarded(self,index,purpose,truth=True):
  if purpose!='selection_dev' or truth:raise RuntimeError('predictions/matching may only read dev observations')
  return original(self,index,purpose,truth=False)
 FormalData.item=guarded
 try:yield
 finally:FormalData.item=original


def save_npz(path,**arrays):
 tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)


@torch.no_grad()
def predict_fold(fold,config):
 inputs,model,identity,prior=fold_assets(fold);folder=RUN/fold;folder.mkdir(exist_ok=True)
 immutable(folder/'IDENTITY.json',identity)
 ids=inputs.split['selection_dev_indices'];pred=folder/'predictions';pred.mkdir(exist_ok=True)
 acache=folder/'native_A';acache.mkdir(exist_ok=True)
 rows=[];started=time.time();cfgsha=digest(RUN/'RUN_CONFIG.json');idsha=canonical_hash(identity)
 with observations_only():
  for pos,index in enumerate(ids):
   window=f'dev_{pos+1:04d}';old=prior['rows'][pos]
   assert old['window']==window and old['identity_sha256']==canonical_hash(prior['identity'])
   assert old['config_sha256']==identity['original_B_config_sha256']
   bp=SWAP/fold/'predictions'/old['file'];assert digest(bp)==old['sha256']
   with np.load(bp,allow_pickle=False) as z:b=z['B'].copy()
   assert b.dtype==np.float32 and b.shape==(20,old['n'],12,2) and np.isfinite(b).all()
   # B ordinal alignment is tied to the same complete split/member hash and seed.
   member=inputs.split['members'][index];assert old['n']==len(member['agent_ids'])
   alignment=canonical_hash(dict(fold=fold,member=member,seed=2035,valid=[True]*old['n']))
   output=pred/f'{pos:04d}.npz';side=output.with_suffix('.json')
   if side.exists():
    row=read(side)
    assert row['alignment_sha256']==alignment and row['identity_sha256']==idsha and row['config_sha256']==cfgsha
    assert row['sha256']==digest(output) and row['B_source_sha256']==old['sha256']
   else:
    ap=acache/f'{pos:04d}.npz';apmeta=ap.with_suffix('.json')
    if apmeta.exists():
     ainfo=read(apmeta)
     assert ainfo['alignment_sha256']==alignment and ainfo['identity_sha256']==idsha and ainfo['sha256']==digest(ap)
     with np.load(ap,allow_pickle=False) as z:a=z['A'].copy();valid=z['valid'].copy()
    else:
     tick=time.perf_counter();batch=inputs.data.batch([index],'selection_dev',truth=False)
     assert batch.valid.all() and not batch.future.count_nonzero()
     goals=inputs.banks['selection_dev'].batch([index],batch)
     _,path,hpi,proof=generate_a(model,batch.observed,batch.semantic_maps,batch.valid,batch.scene_ids,batch.agent_ids,goals,2035,mode='H')
     assert proof['exact_candidate_gather'] and proof['exact_noise_gather'] and proof['per_agent_bijection']
     a=path[0].cpu().numpy();valid=batch.valid[0].cpu().numpy()
     save_npz(ap,A=a,valid=valid,H0_permutation=hpi.cpu().numpy())
     ainfo=dict(alignment_sha256=alignment,identity_sha256=idsha,sha256=digest(ap),
      A_checkpoint_sha256=EXPECTED[fold]['A_sha256'],seed=2035,mode='H',DDIM=20,eta=0.,
      native_inference_seconds=time.perf_counter()-tick,origin='this_run_missing_native_A_inferred_once',proof=proof)
     write(apmeta,ainfo)
    assert a.dtype==np.float32 and a.shape==b.shape and valid.shape==(old['n'],)
    assert valid.all() and np.isfinite(a).all()
    r,permutation,matching=repair_trajectories(a,b[:,:,-1,:],valid)
    save_npz(output,A=a,B=b,R=r,perm=permutation,valid=valid)
    row=dict(window=window,file=output.name,n=int(valid.sum()),sha256=digest(output),
     identity_sha256=idsha,config_sha256=cfgsha,alignment_sha256=alignment,
     A_source_sha256=ainfo['sha256'],B_source_sha256=old['sha256'],
     A_origin=ainfo['origin'],B_origin='original_B_from_sealed_whole_scene_swap_cache',
     native_A_inference_seconds=ainfo['native_inference_seconds'],matching=matching,
     R_additional_decode_calls=0,calibration=False,whole_scene_swap=False)
    write(side,row)
   rows.append(row)
   write(RUN/'STATUS.json',dict(status='PREDICTING_AND_MATCHING_WITHOUT_TRUTH',fold=fold,completed=pos+1,total=len(ids),seconds=time.time()-started))
   print(fold,pos+1,'match_s',row['matching']['matching_seconds'],'elapsed',time.time()-started,flush=True)
 assert state_hash(model)==identity['original_B_identity']['frozen_states']['backbone']
 assert not any(p.requires_grad for p in model.parameters())
 sealpath=folder/'PREDICTIONS_FROZEN.json'
 if sealpath.exists():
  seal=read(sealpath);assert seal['rows']==rows and seal['identity']==identity and seal['config_sha256']==cfgsha
 else:
  seal=dict(identity=identity,config_sha256=cfgsha,rows=rows,seal_unix=time.time());write(sealpath,seal)
 return seal


def verify_all_seals(seals):
 for fold,seal in seals.items():
  assert read(RUN/fold/'PREDICTIONS_FROZEN.json')==seal
  assert len(seal['rows'])==EXPECTED[fold]['windows']
  for pos,row in enumerate(seal['rows']):
   assert row['window']==f'dev_{pos+1:04d}'
   path=RUN/fold/'predictions'/row['file'];assert digest(path)==row['sha256']
   with np.load(path,allow_pickle=False) as z:
    assert set(z.files)=={'A','B','R','perm','valid'}
    a,b,r,p,valid=(z[key] for key in ('A','B','R','perm','valid'))
    assert all(x.dtype==np.float32 and x.shape==(20,row['n'],12,2) and np.isfinite(x).all() for x in (a,b,r))
    for i in np.flatnonzero(valid):
     assert np.array_equal(np.sort(p[:,i]),np.arange(20))
     assert r[np.argsort(p[:,i]),i].tobytes()==a[:,i].tobytes()
 return dict(status='BOTH_FOLDS_ALL_ABR_VERIFIED_BEFORE_TRUTH',
  manifests={f:digest(RUN/f/'PREDICTIONS_FROZEN.json') for f in FOLDS},
  windows=sum(EXPECTED[f]['windows'] for f in FOLDS),verification_unix=time.time(),test_access=0)


def pack_stats(st):
 return {field.name:getattr(st,field.name).cpu().tolist() for field in fields(BatchMetricStatistics)}


def unpack_stats(value):
 return BatchMetricStatistics(**{key:torch.tensor(item,dtype=torch.bool if key=='agent_valid' else torch.float32) for key,item in value.items()})


def marginal_invariants(stats,n):
 a,r=stats['A'],stats['R'];valid=a.agent_valid
 assert torch.equal(valid,r.agent_valid)
 ade=float((r.agent_minade[valid]-a.agent_minade[valid]).abs().max())
 fde=float((r.agent_minfde[valid]-a.agent_minfde[valid]).abs().max())
 assert ade==fde==0.,'R must preserve each valid agent marginal exactly'
 singleton=True
 if n==1:
  singleton=all(torch.equal(getattr(a,k),getattr(r,k)) for k in ('agent_minade','agent_minfde','scene_jade','scene_jfde'))
  assert singleton,'singleton four metrics changed'
 return dict(agent_minADE_max_abs_m=ade,agent_minFDE_max_abs_m=fde,singleton_metrics_equal=singleton)


def score_fold(fold,seal,global_verification):
 # Only this entry reads future; all two-fold predictions were already verified.
 assert set(global_verification['manifests'])==set(FOLDS)
 for f,sha in global_verification['manifests'].items():assert digest(RUN/f/'PREDICTIONS_FROZEN.json')==sha
 inputs,model,identity,prior=fold_assets(fold);del model,prior
 assert identity==seal['identity']
 folder=RUN/fold;out=DOC/fold;out.mkdir(parents=True,exist_ok=True)
 scored=folder/'scored_windows';scored.mkdir(exist_ok=True)
 acc={g:{b:MetricAccumulator() for b in BRANCHES} for g in GROUPS};rows=[];new_reads=0;read_times=[]
 for pos,index in enumerate(inputs.split['selection_dev_indices']):
  row=seal['rows'][pos];path=folder/'predictions'/row['file'];assert digest(path)==row['sha256']
  sp=scored/f'{pos:04d}.json'
  if sp.exists():
   cached=read(sp);assert cached['prediction_sha256']==row['sha256'] and cached['global_manifests']==global_verification['manifests']
   stats={b:unpack_stats(cached['statistics'][b]) for b in BRANCHES}
  else:
   with np.load(path,allow_pickle=False) as z:arrays={b:z[b].copy() for b in BRANCHES};valid=z['valid'].copy()
   first=time.time();assert first>global_verification['verification_unix']>max(read(RUN/f/'PREDICTIONS_FROZEN.json')['seal_unix'] for f in FOLDS)
   batch=inputs.data.batch([index],'selection_dev',truth=True);new_reads+=1
   assert np.array_equal(batch.valid[0].numpy(),valid) and int(batch.valid.sum())==row['n']
   stats={b:hard_metric_statistics(torch.from_numpy(arrays[b])[None],batch.future,batch.valid) for b in BRANCHES}
   marginal_invariants(stats,row['n'])
   cached=dict(prediction_sha256=row['sha256'],global_manifests=global_verification['manifests'],truth_read_unix=first,statistics={b:pack_stats(st) for b,st in stats.items()})
   write(sp,cached)
  read_times.append(cached['truth_read_unix']);invariants=marginal_invariants(stats,row['n'])
  record=dict(fold=fold,window=row['window'],n=row['n'],inverse_restore_failures=row['matching']['inverse_restore_failures'],changed_agents=row['matching']['changed_agents'],changed_window=row['matching']['changed_window'],matching_seconds=row['matching']['matching_seconds'],hungarian_seconds=row['matching']['hungarian_seconds'],**invariants)
  for b,st in stats.items():
   acc['all'][b].update_statistics(st)
   if row['n']>=9:acc['N_ge_9'][b].update_statistics(st)
   record.update({b+'_JADE':float(st.scene_jade[0]),b+'_JFDE':float(st.scene_jfde[0]),
    b+'_minADE':float(st.agent_minade[st.agent_valid].double().mean()),b+'_minFDE':float(st.agent_minfde[st.agent_valid].double().mean()),
    b+'_minADE_sum':float(st.agent_minade[st.agent_valid].double().sum()),b+'_minFDE_sum':float(st.agent_minfde[st.agent_valid].double().sum())})
  for baseline in ('A','B'):
   for metric in METRICS:record[f'R_minus_{baseline}_{metric}_cm']=100*(record['R_'+metric]-record[baseline+'_'+metric])
  rows.append(record)
 metrics={g:{b:a.as_dict() for b,a in branches.items()} for g,branches in acc.items()}
 assert metrics['all']['A']['scene_count']==EXPECTED[fold]['windows'] and metrics['N_ge_9']['A']['scene_count']==EXPECTED[fold]['dense_windows']
 reference=archive_json(f'docs/research/whole_scene_swap/zara1_zara2_20261123/{fold}/NATIVE_METRICS.json')
 for group in GROUPS:
  assert metrics[group]['B']==reference[group]['B'],'sealed B native metrics must replay exactly'
  assert all(metrics[group]['R'][m]==metrics[group]['A'][m] for m in ('minADE','minFDE'))
 if fold=='zara1':
  history=archive_json('docs/research/continuous_joint_flow/fourfold_confirmation_20261122/zara1_training_status/upstream/joint/EPOCH_METRICS.json')
  a_ref={'all':next(x['metrics'] for x in history if x['epoch']==15)}
 else:
  a_ref=archive_json('docs/research/continuous_joint_flow/zara2_20261122/FINAL_METRICS.json')['A']
 replay={g:{m:metrics[g]['A'][m]-v[m] for m in METRICS} for g,v in a_ref.items()}
 assert all(abs(x)<=1e-4 for g in replay.values() for x in g.values()),'native A replay mismatch requires investigation'
 checks=[dict(fold=fold,group=g,metric=m,**joint_criterion(metrics[g]['A'][m],metrics[g]['B'][m],metrics[g]['R'][m])) for g,m in [('all','JADE'),('all','JFDE'),('N_ge_9','JFDE')]]
 diagnostics=dict(windows=len(rows),agents=metrics['all']['A']['agent_count'],
  inverse_restore_failures=sum(r['inverse_restore_failures'] for r in rows),
  agent_minADE_max_abs_m=max(r['agent_minADE_max_abs_m'] for r in rows),agent_minFDE_max_abs_m=max(r['agent_minFDE_max_abs_m'] for r in rows),
  singleton_windows=sum(r['n']==1 for r in rows),singleton_metric_failure_windows=sum(r['n']==1 and not r['singleton_metrics_equal'] for r in rows),
  changed_windows=sum(r['changed_window'] for r in rows),zero_change_windows=sum(not r['changed_window'] for r in rows),
  matching_seconds=sum(r['matching_seconds'] for r in rows),hungarian_seconds=sum(r['hungarian_seconds'] for r in rows),
  B_reused_prior_windows=len(rows),A_reused_prior_full_trajectory_windows=0,A_missing_windows_inferred=len(rows),B_new_inference_windows=0,R_decode_calls=0,
  A_native_decode_seconds=sum(r['native_A_inference_seconds'] for r in seal['rows']),training_updates=0,test_access=0)
 decision=dict(fold=fold,marginal_identity=True,joint_checks=checks,passed=all(r['passed'] for r in checks))
 summary=[]
 for g in GROUPS:
  for m in METRICS:
   a,b,r=(metrics[g][branch][m] for branch in BRANCHES)
   summary.append(dict(fold=fold,group=g,windows=metrics[g]['A']['scene_count'],agents=metrics[g]['A']['agent_count'],metric=m,A=a,B=b,R=r,R_minus_A_cm=100*(r-a),R_minus_B_cm=100*(r-b)))
 csvwrite(out/'PER_WINDOW.csv',rows);csvwrite(out/'SUMMARY.csv',summary);csvwrite(out/'JOINT_CRITERIA.csv',checks)
 write(out/'NATIVE_METRICS.json',metrics);write(out/'INVARIANTS_AND_RUNTIME.json',diagnostics);write(out/'DECISION.json',decision)
 write(out/'ASSET_IDENTITY.json',identity);write(out/'A_BASELINE_REPLAY.json',dict(absolute_difference_m=replay,tolerance_m=1e-4,B_native_replay_exact=True))
 write(out/'PREDICTIONS_FROZEN_PUBLIC.json',dict(identity=identity,original_manifest_sha256=digest(folder/'PREDICTIONS_FROZEN.json'),seal_unix=seal['seal_unix'],rows=[dict(window=r['window'],n=r['n'],prediction_sha256=r['sha256'],alignment_sha256=r['alignment_sha256']) for r in seal['rows']]))
 receipt=dict(status='COMPLETE',fold=fold,unique_scored_windows=len(rows),new_future_reads_this_invocation=new_reads,
  both_folds_sealed_before_truth=True,global_manifests=global_verification['manifests'],global_verification_unix=global_verification['verification_unix'],all_fold_seal_unix={f:read(RUN/f/'PREDICTIONS_FROZEN.json')['seal_unix'] for f in FOLDS},
  first_truth_unix=min(read_times),last_truth_unix=max(read_times),native_scoring='hard_metric_statistics + MetricAccumulator',test_access=0,training_updates=0,
  output_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name not in ('SCORE_RECEIPT.json','README.md')})
 write(out/'SCORE_RECEIPT.json',receipt)
 return dict(metrics=metrics,diagnostics=diagnostics,decision=decision,summary=summary)


def main():
 RUN.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
 with (RUN/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
  config=dict(base_commit=BASE,folds=list(FOLDS),models=EXPECTED,seed=2035,K=20,unit='m',observation_frames=8,dt_seconds=.4,prediction_frames=12,DDIM=20,eta=0.,training_updates=0,test_access=0,
   matching='independent per-valid-agent endpoint Euclidean float64 Hungarian; rows shared B scene ID; columns native A trajectory ID; full 12-frame gather',
   scipy_version=scipy.__version__,numpy_version=np.__version__,torch_version=str(torch.__version__),solver='scipy.optimize.linear_sum_assignment',
   tie_protocol='original row/column numbering; fixed solver version; no random perturbation',
   B_source='sealed original B array from whole_scene_swap; never swap output',calibration=False,whole_scene_swap=False,
   criterion='marginal exact identity AND all six unrounded joint checks; positive gain retention>=90%; otherwise R<=A',
   model_device='cpu',torch_threads=2,matching_precision='float64',scoring='native CPU FP32 hard metrics and native FP64 accumulator',
   joint_aggregation='equal windows',marginal_aggregation='valid-agent weighted',
   A_archive_replay_absolute_tolerance_m=1e-4,marginal_identity_tolerance_m=0.,source_sha256=source_hashes())
  immutable(RUN/'RUN_CONFIG.json',config);check_protection(read(FOUR/'RUN_CONFIG.json'))
  seals={f:predict_fold(f,config) for f in FOLDS}
  verification=verify_all_seals(seals)
  if (RUN/'ALL_PREDICTIONS_VERIFIED.json').exists():
   previous=read(RUN/'ALL_PREDICTIONS_VERIFIED.json');assert previous['manifests']==verification['manifests'];verification=previous
  else:write(RUN/'ALL_PREDICTIONS_VERIFIED.json',verification)
  write(RUN/'STATUS.json',dict(status='SCORING_SEALED_PREDICTIONS',unix=time.time()))
  result={f:score_fold(f,seals[f],verification) for f in FOLDS}
  from .report import build_report
  build_report(result,config,DOC)
  assert source_hashes()==config['source_sha256'];check_protection(read(FOUR/'RUN_CONFIG.json'))
  for p in DOC.rglob('*'):
   if p.is_file():assert p.suffix in ('.json','.csv','.md') and b'/media/ee615' not in p.read_bytes()
  subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],cwd=SOURCE,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=SOURCE,check=True)
  if subprocess.call(['git','diff','--cached','--quiet'],cwd=SOURCE)!=0:subprocess.run(['git','commit','-m','Archive two-fold B-guided A trajectory repairing results'],cwd=SOURCE,check=True)
  branch=subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip();subprocess.run(['git','push','-u','origin',branch],cwd=SOURCE,check=True)
  sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=SOURCE,text=True).split()[0];assert sha==remote
  write(RUN/'DELIVERY.json',dict(sha=sha,report=f'https://github.com/WuYanXingege/RSJG/blob/{sha}/{DOC.relative_to(SOURCE)}/README.md',summary=f'https://github.com/WuYanXingege/RSJG/blob/{sha}/{DOC.relative_to(SOURCE)}/SUMMARY.csv'))
  write(RUN/'STATUS.json',dict(status='COMPLETE',unix=time.time(),decision=all(result[f]['decision']['passed'] for f in FOLDS)))
if __name__=='__main__':
 try:main()
 except BaseException:
  RUN.mkdir(parents=True,exist_ok=True);write(RUN/'STATUS.json',dict(status='INTERRUPTED_RESUME_SAME_RUN',error=traceback.format_exc()));raise
