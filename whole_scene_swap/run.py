"""Two frozen dev folds, output-only whole-scene swaps, sealed paired scoring."""
import csv,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import read,write
from bcep_formal.data import FormalData
from bcep_v2.provider import GEOMETRY
from ebjd.metrics import MetricAccumulator,hard_metric_statistics
from joint_flow_dependency.model import EdgeDependence,undirected_edges,sample_latents
from joint_flow_dependency.inference import encode_observed,flow_world,decode
from joint_flow_dependency.sampling import base_noise,bridge_noise,select_joint_samples
from joint_flow_fourfold.inputs import Inputs as FoldInputs,state_hash
from scene_pair_selector.inputs import Inputs as Zara2Inputs
from whole_scene_swap.selector import select_swaps
SOURCE=Path(__file__).resolve().parents[1];ROOT=SOURCE.parent
RUN=ROOT/'RSJG_EBJD_assets/experiments/zara1_zara2_whole_scene_swap_20261123'
DOC=SOURCE/'docs/research/whole_scene_swap/zara1_zara2_20261123'
FOUR=ROOT/'RSJG_EBJD_assets/experiments/joint_flow_fourfold_confirmation_20261122'
Z2=ROOT/'RSJG_EBJD_assets/experiments/zara2_joint_flow_dependency_20261122'
CACHE=ROOT/'RSJG_EBJD_assets/experiments/zara2_endpoint_calibration_1cm_20261123'
BASE='d3e2aaf791cb9c7ff0603711dfa07e49ad718217'
Z2_COMMIT='d4fbe6d8e3b142645b9e801116aac308fcb19c9d'
METRICS=('JADE','JFDE','minADE','minFDE')
FOLDS=('zara1','zara2')

def immutable(path,value):
 if path.exists():assert read(path)==value,str(path)
 else:write(path,value)

def csvwrite(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def sources():
 return {str(p.relative_to(SOURCE)):digest(p) for folder in ('whole_scene_swap','joint_flow_dependency','scene_pair_selector','joint_flow_fourfold','bcep','bcep_v2','bcep_joint','bcep_formal','ebjd','bcep_confirmation','selector_control') for p in sorted((SOURCE/folder).rglob('*.py'))}

def assets(fold):
 if fold=='zara1':
  upstream=read(FOUR/fold/'UPSTREAM_FROZEN.json');result=read(FOUR/fold/'B_FROZEN.json')
  # Check the fixed archived identity, not a newly changing selector.
  fixed=json.loads(subprocess.check_output(['git','show',BASE+':docs/research/continuous_joint_flow/fourfold_confirmation_20261122/zara1_training_status/B_FROZEN.json'],cwd=SOURCE))
  assert result['rank1_sha256']==fixed['rank1_sha256']=='fc7d9cca99769b12f2366eda20396130f547534248c82451e1041d640ff8389d'
  inputs=FoldInputs(upstream,RUN/fold,'cpu');path=Path(result['rank1_checkpoint']);expected=671;dense=181;epoch=6
 else:
  inputs=Zara2Inputs();result=read(Z2/'TRAINING_RESULT.json');path=Z2/'dependence/epoch_035.pt';expected=590;dense=127;epoch=35
  assert result['rank1_sha256']=='ea13cd7fd99a02c44c3c9af9a14a0cc9b5166b629e5c338694cca97f1c8c502a'
  fixed=json.loads(subprocess.check_output(['git','show',Z2_COMMIT+':docs/research/continuous_joint_flow/zara2_20261122/TRAINING_RESULT.json'],cwd=SOURCE));assert result==fixed
 assert digest(path)==result['rank1_sha256'] and result['rank1_epoch']==epoch
 ids=inputs.split['selection_dev_indices'];assert len(ids)==expected and sum(len(inputs.split['members'][i]['agent_ids'])>=9 for i in ids)==dense
 model,provider=inputs.load_frozen('cpu');dep=EdgeDependence()
 saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['completed_epoch']==epoch
 dep.load_state_dict(saved['model'],strict=True)
 for m in (model,provider,dep):m.cpu().eval().requires_grad_(False)
 identity=dict(fold=fold,upstream=inputs.identity,checkpoint_sha256=digest(path),epoch=epoch,dev_windows=expected,dense_windows=dense,
  frozen_states={k:state_hash(m) for k,m in dict(backbone=model,provider=provider,dependence=dep).items()})
 return inputs,model,provider,dep,identity,result

@contextmanager
def observed_only():
 original=FormalData.item
 def guarded(self,index,purpose,truth=True):
  if purpose!='selection_dev' or truth:raise RuntimeError('prediction stage may only read selection_dev observations')
  return original(self,index,purpose,truth=False)
 FormalData.item=guarded
 try:yield
 finally:FormalData.item=original

@torch.no_grad()
def predict_fold(fold,config):
 inputs,model,provider,dep,identity,result=assets(fold)
 folder=RUN/fold;folder.mkdir(exist_ok=True);immutable(folder/'IDENTITY.json',identity)
 prediction=folder/'predictions';prediction.mkdir(exist_ok=True);ids=inputs.split['selection_dev_indices'];rows=[]
 started=time.time();cache_identity=None
 if fold=='zara2' and (CACHE/'RUN_CONFIG.json').exists():
  cache_identity=read(CACHE/'RUN_CONFIG.json')
  assert cache_identity['checkpoint_sha256']==identity['checkpoint_sha256'] and cache_identity['seed']==2035
  assert read(CACHE/'STATUS.json')['status']=='COMPLETE'
 with observed_only():
  for pos,index in enumerate(ids):
   path=prediction/f'{pos:04d}.npz';side=path.with_suffix('.json')
   if side.exists():
    row=read(side);assert row['config_sha256']==digest(RUN/'RUN_CONFIG.json') and row['identity_sha256']==canonical_hash(identity) and row['sha256']==digest(path)
   else:
    begin=time.perf_counter();batch=inputs.data.batch([index],'selection_dev',truth=False)
    assert batch.valid.all() and not batch.future.count_nonzero()
    context,features=encode_observed(provider,batch)
    rawpath=prediction/f'{pos:04d}_raw.npz';rawside=rawpath.with_suffix('.json');cache_hit=False
    if rawside.exists():
     proof=read(rawside);assert proof['identity_sha256']==canonical_hash(identity) and proof['sha256']==digest(rawpath)
     with np.load(rawpath) as z:raw=torch.from_numpy(z['raw'].copy())
     cache_hit=True
    elif fold=='zara2' and cache_identity is not None and (CACHE/'predictions'/f'{pos:04d}_base.npz').exists():
     src=CACHE/'predictions'/f'{pos:04d}_base.npz';proof=read(src.with_suffix('.json'))
     assert proof['config']==cache_identity and proof['sha256']==digest(src)
     with np.load(src) as z:raw=torch.from_numpy(z['raw'].copy())
     cache_hit=True
    else:
     edges=undirected_edges(batch.observed[0]);eps,xi=base_noise(batch.scene_ids[0],batch.agent_ids[0],edges,seed=2035)
     factor=dep(batch.observed[0],context,edges)
     raw=flow_world(provider,batch.observed[0],context,sample_latents(factor,eps,xi))
    assert raw.shape==(256,len(batch.agent_ids[0]),2) and raw.dtype==torch.float32
    if not rawside.exists():
     tmp=rawpath.with_suffix('.tmp.npz');np.savez_compressed(tmp,raw=raw.numpy());tmp.replace(rawpath)
     write(rawside,dict(identity_sha256=canonical_hash(identity),sha256=digest(rawpath),original_cache_hit=cache_hit))
    # Both paths share observation encoding and a single raw bank. Both end-to-end
    # timings include this same measured common part; model startup/I/O excluded below.
    bridge_context=model.encode_context(batch.observed,batch.semantic_maps,batch.valid)
    common_seconds=time.perf_counter()-begin
    t=time.perf_counter();original_ids,_=select_joint_samples(raw);base_selection_seconds=time.perf_counter()-t
    t=time.perf_counter();new_ids,selection=select_swaps(raw.numpy(),original_ids.numpy());swap_seconds=time.perf_counter()-t
    predictions={};noise_map={};branch_seconds={}
    for branch,chosen in [('B',original_ids),('swap',torch.from_numpy(new_ids))]:
     t=time.perf_counter()
     corrected,shift=provider.head(provider.base,batch.observed[0],batch.semantic_maps[0],batch.valid[0],raw[chosen],context,features,dict(layout=GEOMETRY))
     assert shift.norm(dim=-1).max()<.5
     noise=bridge_noise(batch.scene_ids[0],batch.agent_ids[0],chosen.tolist(),seed=2035)
     predictions[branch]=decode(model,bridge_context,corrected[None],noise)[0].numpy()
     branch_seconds[branch]=time.perf_counter()-t
     noise_map[branch]={int(i):noise[0,k] for k,i in enumerate(chosen)}
    for i in noise_map['B'].keys()&noise_map['swap'].keys():assert torch.equal(noise_map['B'][i],noise_map['swap'][i])
    if selection['swaps']==0:assert np.array_equal(predictions['B'],predictions['swap'])
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**predictions,S0=original_ids.numpy(),S=new_ids);tmp.replace(path)
    row=dict(window=f'dev_{pos+1:04d}',file=path.name,n=int(batch.valid.sum()),sha256=digest(path),identity_sha256=canonical_hash(identity),config_sha256=digest(RUN/'RUN_CONFIG.json'),selection=selection,
     timings=dict(common_seconds=common_seconds,original_selection_seconds=base_selection_seconds,swap_added_seconds=swap_seconds,B_head_bridge_seconds=branch_seconds['B'],new_head_bridge_seconds=branch_seconds['swap'],
      B_end_to_end_seconds=common_seconds+base_selection_seconds+branch_seconds['B'],new_end_to_end_seconds=common_seconds+base_selection_seconds+swap_seconds+branch_seconds['swap']),
     raw_cache_hit=cache_hit,full_set_correction_calls_per_branch=1,ddim_calls_per_branch=1,identical_raw_id_noise_verified=True,calibration=False)
    write(side,row)
   rows.append(row)
   write(RUN/'STATUS.json',dict(status='PREDICTING_WITHOUT_TRUTH',fold=fold,completed=pos+1,total=len(ids),seconds=time.time()-started))
   print(fold,pos+1,'swaps',row['selection']['swaps'],'elapsed',time.time()-started,flush=True)
 for name,module in dict(backbone=model,provider=provider,dependence=dep).items():
  assert state_hash(module)==identity['frozen_states'][name]
  assert not any(p.requires_grad for p in module.parameters())
 sealpath=folder/'PREDICTIONS_FROZEN.json'
 if sealpath.exists():seal=read(sealpath);assert seal['rows']==rows and seal['identity']==identity
 else:seal=dict(identity=identity,rows=rows,unix=time.time());write(sealpath,seal)
 return seal


def score_fold(fold,seal):
 folder=RUN/fold;inputs,model,provider,dep,identity,result=assets(fold)
 del model,provider,dep
 assert seal['identity']==identity
 ids=inputs.split['selection_dev_indices'];assert len(seal['rows'])==len(ids)
 for row in seal['rows']:assert digest(folder/'predictions'/row['file'])==row['sha256']
 verified=time.time();acc={g:{b:MetricAccumulator() for b in ('B','swap')} for g in ('all','N_ge_9')};records=[]
 first=time.time()
 for pos,index in enumerate(ids):
  detail=seal['rows'][pos];path=folder/'predictions'/detail['file'];assert digest(path)==detail['sha256']
  with np.load(path) as z:arrays={b:z[b].copy() for b in ('B','swap')}
  batch=inputs.data.batch([index],'selection_dev',truth=True);n=int(batch.valid.sum());assert n==detail['n']
  sel=detail['selection'];row=dict(fold=fold,window=detail['window'],n=n,swaps=sel['swaps'],J0=sel['J0'],J=sel['J'],P0=sel['P0'],P=sel['P'],constraint_violations=sel['constraint_violations'],**detail['timings'])
  for b in ('B','swap'):
   st=hard_metric_statistics(torch.from_numpy(arrays[b])[None],batch.future,batch.valid)
   acc['all'][b].update_statistics(st)
   if n>=9:acc['N_ge_9'][b].update_statistics(st)
   row.update({b+'_JADE':float(st.scene_jade[0]),b+'_JFDE':float(st.scene_jfde[0]),b+'_minADE':float(st.agent_minade[st.agent_valid].double().mean()),b+'_minFDE':float(st.agent_minfde[st.agent_valid].double().mean()),b+'_minADE_sum':float(st.agent_minade[st.agent_valid].double().sum()),b+'_minFDE_sum':float(st.agent_minfde[st.agent_valid].double().sum())})
  for m in METRICS:row['difference_'+m+'_cm']=100*(row['swap_'+m]-row['B_'+m])
  records.append(row)
 metrics={g:{b:a.as_dict() for b,a in v.items()} for g,v in acc.items()}
 reference=result['rank1_metrics'];replay={}
 for group in acc:
  replay[group]={m:metrics[group]['B'][m]-reference[group][m] for m in METRICS}
  assert abs(replay[group]['minFDE'])<=1e-5,(fold,group,replay[group])
 out=DOC/fold;out.mkdir(parents=True,exist_ok=True);csvwrite(out/'PER_WINDOW.csv',records);write(out/'NATIVE_METRICS.json',metrics);write(out/'BASELINE_REPLAY.json',dict(reference=reference,replay_difference=replay,device='CPU',reference_device='CUDA',minFDE_absolute_tolerance_m=1e-5))
 write(out/'SCORE_RECEIPT.json',dict(seal_unix=seal['unix'],verification_unix=verified,first_truth_unix=first,future_reads=len(ids),prediction_manifest_sha256=digest(folder/'PREDICTIONS_FROZEN.json'),test_access=0,training_updates=0))
 summary=[];diagnostics={}
 for group in acc:
  subset=[r for r in records if group=='all' or r['n']>=9];a,b=metrics[group]['B'],metrics[group]['swap']
  for m in METRICS:summary.append(dict(fold=fold,group=group,windows=a['scene_count'],agents=a['agent_count'],metric=m,B=a[m],new=b[m],difference_cm=100*(b[m]-a[m])))
  diagnostics[group]=dict(windows=len(subset),changed_windows=sum(r['swaps']>0 for r in subset),zero_swap_windows=sum(r['swaps']==0 for r in subset),total_swaps=sum(r['swaps'] for r in subset),max_swaps=max(r['swaps'] for r in subset),constraint_violations=sum(r['constraint_violations'] for r in subset),
   **{k+'_mean':float(np.mean([r[k] for r in subset])) for k in ('J0','J','P0','P')},
   **{k+'_sum':sum(r[k] for r in subset) for k in seal['rows'][0]['timings']})
 csvwrite(out/'SUMMARY.csv',summary);write(out/'SELECTION_DIAGNOSTICS.json',diagnostics)
 write(out/'IDENTITY.json',identity)
 return summary,metrics,diagnostics


def report(results,config):
 allrows=[]
 intro=['# ZARA1 / ZARA2 开发集：原 B 与整组交换筛选','固定 seed=2035，零训练、零测试集访问，两路均关闭1 cm校准。所有两折两路预测均在读取开发集真值前封存。以下是开发集配对实验，2 cm仅为研发解读目标。']
 for fold,(summary,metrics,diagnostics) in results.items():
  allrows+=summary;lines=[f'# {fold.upper()} 完整开发集配对结果']
  for group in ('all','N_ge_9'):
   a,b=metrics[group]['B'],metrics[group]['swap'];gain=100*(a['minFDE']-b['minFDE']);ade=100*(b['minADE']-a['minADE']);ja=100*(b['JADE']-a['JADE']);jf=100*(b['JFDE']-a['JFDE'])
   sentence=f'{fold.upper()} / {group}：minFDE降低 {gain:+.9f} cm，'+('达到' if gain>=2 else '未达到')+f'2 cm研发目标；边缘 minADE 变化 {ade:+.9f} cm（'+('改善' if ade<0 else '持平' if ade==0 else '恶化')+f'）；真实 JADE 变化 {ja:+.9f} cm，JFDE 变化 {jf:+.9f} cm。'
   lines.append(sentence);intro.append(sentence)
  lines+=['| 分组 | 窗数 | 人次 | 指标 | B（m） | 新筛选（m） | 新−B（cm） |','|---|---:|---:|---|---:|---:|---:|']
  for r in summary:lines.append(f"| {r['group']} | {r['windows']} | {r['agents']} | {r['metric']} | {r['B']:.12f} | {r['new']:.12f} | {r['difference_cm']:+.9f} |")
  for g,v in diagnostics.items():lines.append(f"{g}：换组 {v['changed_windows']}/{v['windows']} 窗，零交换 {v['zero_swap_windows']} 窗；共 {v['total_swaps']} 次交换；约束违反 {v['constraint_violations']} 次。J均值 {v['J0_mean']:.12f}→{v['J_mean']:.12f} m，P均值 {v['P0_mean']:.12f}→{v['P_mean']:.12f} m。附加筛选耗时 {v['swap_added_seconds_sum']:.3f} s；同设备B端到端累计 {v['B_end_to_end_seconds_sum']:.3f} s，新筛选端到端累计 {v['new_end_to_end_seconds_sum']:.3f} s。")
  lines+=['两路在同一CPU、相同线程配置上运行。端到端值包括各自的观测/地图及共享原始候选准备时间、原筛选时间、整组修正和桥推理；新路另加交换筛选时间。共同部分实际计算一次，分别计入两路独立路径的累计耗时。缓存命中时计入读取时间，不冒充无缓存Flow采样耗时；模型加载、归档和评分不计入。原B的DDIM在同一CPU重放，以获得同设备真实耗时。','每次采用交换后直接重算J/P；精确J≤最初J0，不放宽约束。无合格交换才终止。每条分支整组修正一次、20步DDIM一次；相同原始编号桥噪声逐项核验一致。J/P仅约束候选覆盖，真实JADE/JFDE以上表实测为准。','[汇总CSV](SUMMARY.csv) · [匿名逐窗CSV](PER_WINDOW.csv) · [J/P与耗时](SELECTION_DIAGNOSTICS.json) · [原B回放差](BASELINE_REPLAY.json) · [权重身份](IDENTITY.json)']
  (DOC/fold/'README.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n')
 intro+=['[ZARA1完整报告](zara1/README.md) · [ZARA2完整报告](zara2/README.md) · [两折汇总CSV](SUMMARY.csv) · [固定配置](RUN_CONFIG.json)','联合指标按窗口等权，边缘指标按有效行人人次加权；新筛选减原B为统一差值方向，厘米。候选、权重、场景编号和原始轨迹只保留在训练主机。']
 (DOC/'README.md').write_text('\n\n'.join(intro)+'\n');csvwrite(DOC/'SUMMARY.csv',allrows);write(DOC/'RUN_CONFIG.json',config)


def main():
 RUN.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
 with (RUN/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
  config=dict(base_commit=BASE,folds=list(FOLDS),seed=2035,M=256,K=20,DDIM=20,eta=0,training_updates=0,test_access=0,calibration=False,device='cpu',torch_threads=2,
   swap_rule='best P among all one-swap proposals with J<=initial J0 and P decrease >1e-12; ties outgoing raw ID then incoming raw ID; replace in original slot; no iteration cap',
   coverage_precision='CPU float64',model_and_metrics_precision='FP32; native FP64 accumulation',source_sha256=sources())
  from joint_flow_fourfold.controller import check_protection
  check_protection(read(FOUR/'RUN_CONFIG.json'))
  immutable(RUN/'RUN_CONFIG.json',config)
  # Both folds are sealed before the first dev-future read.
  seals={fold:predict_fold(fold,config) for fold in FOLDS}
  for fold,seal in seals.items():
   for r in seal['rows']:assert digest(RUN/fold/'predictions'/r['file'])==r['sha256']
  write(RUN/'ALL_PREDICTIONS_FROZEN.json',dict(folds={f:digest(RUN/f/'PREDICTIONS_FROZEN.json') for f in FOLDS},unix=time.time()))
  results={fold:score_fold(fold,seals[fold]) for fold in FOLDS}
  report(results,config);assert sources()==config['source_sha256']
  check_protection(read(FOUR/'RUN_CONFIG.json'))
  write(RUN/'STATUS.json',dict(status='SCORED',unix=time.time()))
  subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],cwd=SOURCE,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=SOURCE,check=True)
  subprocess.run(['git','commit','-m','Archive two-fold whole-scene swap development experiment'],cwd=SOURCE,check=True)
  branch=subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip();subprocess.run(['git','push','-u','origin',branch],cwd=SOURCE,check=True)
  sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=SOURCE,text=True).split()[0];assert sha==remote
  write(RUN/'DELIVERY.json',dict(sha=sha,report=f'https://github.com/WuYanXingege/RSJG/blob/{sha}/{DOC.relative_to(SOURCE)}/README.md'));write(RUN/'STATUS.json',dict(status='COMPLETE',unix=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  RUN.mkdir(parents=True,exist_ok=True);write(RUN/'STATUS.json',dict(status='INTERRUPTED_RESUME_SAME_RUN',error=traceback.format_exc()));raise
