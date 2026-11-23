"""Frozen seed2035 test: replay endpoints, reuse sealed native B paths, calibrate."""
import csv,json,time,subprocess,fcntl,traceback,sys
from pathlib import Path
import numpy as np
import torch
from bcep.contracts import digest
from bcep_joint.common import read,write
from bcep_formal.data import FormalData
from bcep_v2.runtime import configure_cuda
from joint_flow_dependency.inputs import Inputs
from joint_flow_dependency.model import EdgeDependence
from joint_flow_dependency.inference import encode_observed,sample_endpoints,generate
from ebjd.metrics import MetricAccumulator,hard_metric_statistics
from endpoint_calibration.run import calibrate,CONFIG as CALIBRATION,table,SHA,WEIGHT
S=Path(__file__).resolve().parents[1];ROOT=S.parent
RUN=ROOT/'RSJG_EBJD_assets/experiments/zara2_calibration_frozen_test_2035_20261123'
OLD=ROOT/'RSJG_EBJD_assets/experiments/zara2_joint_flow_frozen_test_20261122'
DOC=S/'docs/research/continuous_joint_flow/zara2_calibration_frozen_test_2035_20261123'
BASE='171bcff3b6b2dbb6aad0becb04e8d7c086c8ffe3'
CAL='e0ce0fbbafa7bb671c2febcbe984ed7aafcc45a5'
METRICS=('JADE','JFDE','minADE','minFDE')
def main():
 RUN.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
 with (RUN/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(1)
  # Calibrator bytes must be identical to the prespecified development experiment.
  native=subprocess.check_output(['git','show',CAL+':endpoint_calibration/run.py'],cwd=S)
  assert native==(S/'endpoint_calibration/run.py').read_bytes()
  assert digest(WEIGHT)==SHA
  oldseal=read(OLD/'PREDICTIONS_FROZEN.json');assert oldseal['identity']['B_checkpoint_sha256']==SHA
  oldrows={r['index']:r for r in oldseal['rows'] if r['seed']==2035}
  inputs=Inputs();split=inputs.split;ids=split['heldout_confirmation_indices']
  assert len(ids)==998 and len(oldrows)==998
  assert sum(len(split['members'][i]['agent_ids'])>=9 for i in ids)==214
  config=dict(base_commit=BASE,calibration_commit=CAL,calibration=CALIBRATION,checkpoint_sha256=SHA,
   purpose='heldout_confirmation',seed=2035,windows=998,dense_windows=214,updates=0,
   native_prediction_manifest_sha256=digest(OLD/'PREDICTIONS_FROZEN.json'),
   endpoint_replay='original GPU FP32 seed2035 Flow/head; compare corrected endpoints to sealed B final frames atol1e-6 rtol0',
   scoring='native CPU FP32 hard metrics, FP64 accumulation; joint window macro, marginal agent micro',
   source_sha256=digest(__file__))
  if (RUN/'RUN_CONFIG.json').exists():assert read(RUN/'RUN_CONFIG.json')==config
  else:write(RUN/'RUN_CONFIG.json',config)
  folder=RUN/'predictions';folder.mkdir(exist_ok=True);rows=[];start=time.time()
  data=FormalData(split,phase='export',verify=True)
  if '--export-only' in sys.argv:
   configure_cuda()
   model,provider=inputs.load_frozen('cuda');model.cpu();provider.to('cuda');dep=EdgeDependence().cuda()
   saved=torch.load(WEIGHT,map_location='cpu',weights_only=False);dep.load_state_dict(saved['model'],strict=True);dep.eval().requires_grad_(False)
   for pos,index in enumerate(ids):
    cache=folder/f'{pos:04d}_raw.npz'
    if cache.exists():
     proof=read(cache.with_suffix('.json'));assert proof['sha256']==digest(cache) and proof['config_sha256']==digest(RUN/'RUN_CONFIG.json')
     continue
    batch=data.batch([index],'heldout_confirmation',truth=False).to('cuda')
    with torch.no_grad():
     ctx,features=encode_observed(provider,batch);e=sample_endpoints(provider,dep,batch,ctx,features)
    raw=e['raw'].cpu().numpy();g=e['corrected'][0].cpu().numpy();samples=e['raw_indices'].cpu().numpy()
    op=OLD/oldrows[index]['file'];assert digest(op)==oldrows[index]['sha256']
    with np.load(op) as z:assert np.abs(z['B'][:,:,-1]-g).max()<=1e-6
    temp=cache.with_suffix('.tmp.npz');np.savez_compressed(temp,raw=raw,g=g,ids=samples,noise=e['noise'].cpu().numpy());temp.replace(cache)
    write(cache.with_suffix('.json'),dict(sha256=digest(cache),config_sha256=digest(RUN/'RUN_CONFIG.json')))
    write(RUN/'STATUS.json',dict(status='ENDPOINT_REPLAY_WITHOUT_TRUTH',completed=pos+1,total=998,seconds=time.time()-start))
   return
  assert all((folder/f'{pos:04d}_raw.npz').exists() for pos in range(998)), 'run --export-only first'
  original=FormalData.item
  def guarded(self,index,purpose,truth=True):
   if purpose=='heldout_confirmation' and truth:raise RuntimeError('truth forbidden before global seal')
   return original(self,index,purpose,truth)
  FormalData.item=guarded
  try:
   for pos,index in enumerate(ids):
    path=folder/f'{pos:04d}.npz';side=path.with_suffix('.json');old=oldrows[index];op=OLD/old['file']
    assert digest(op)==old['sha256']
    if side.exists():
     row=read(side);assert row['sha256']==digest(path) and row['config_sha256']==digest(RUN/'RUN_CONFIG.json')
    else:
     cache=folder/f'{pos:04d}_raw.npz'
     if cache.exists():
      proof=read(cache.with_suffix('.json'));assert proof['sha256']==digest(cache) and proof['config_sha256']==digest(RUN/'RUN_CONFIG.json')
      with np.load(cache) as z:raw,g,samples=z['raw'],z['g'],z['ids']
     else:
      batch=data.batch([index],'heldout_confirmation',truth=False).to('cuda')
      with torch.no_grad():
       ctx,features=encode_observed(provider,batch);e=sample_endpoints(provider,dep,batch,ctx,features)
      raw=e['raw'].cpu().numpy();g=e['corrected'][0].cpu().numpy();samples=e['raw_indices'].cpu().numpy()
      temp=cache.with_suffix('.tmp.npz');np.savez_compressed(temp,raw=raw,g=g,ids=samples,noise=e['noise'].cpu().numpy());temp.replace(cache)
      write(cache.with_suffix('.json'),dict(sha256=digest(cache),config_sha256=digest(RUN/'RUN_CONFIG.json')))
     with np.load(op) as z:y=z['B'].copy()
     assert y.shape==(20,len(split['members'][index]['agent_ids']),12,2)
     replay=float(np.abs(y[:,:,-1]-g).max());assert replay<=1e-6,(pos,replay)
     d,diag=calibrate(raw,g,samples)
     yp=(y.astype(np.float64)+np.arange(1,13)[None,None,:,None]/12*d[:,:,None]).astype(np.float32)
     endpoint_error=float(np.abs(yp[:,:,-1]-(g+d)).max());assert endpoint_error<=1e-5
     tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,B=y,Bp=yp,delta=d);tmp.replace(path)
     row=dict(window=f'test_{pos+1:04d}',file=path.name,sha256=digest(path),n=y.shape[1],
      config_sha256=digest(RUN/'RUN_CONFIG.json'),original_B_sha256=old['sha256'],endpoint_replay_max_m=replay,
      Bp_endpoint_max_error_m=endpoint_error,diagnostics=diag,original_B_trajectory_reused_exactly=True)
     write(side,row)
    rows.append(row)
    write(RUN/'STATUS.json',dict(status='CALIBRATING_WITHOUT_TRUTH',completed=pos+1,total=998,seconds=time.time()-start))
    print('window',pos+1,time.time()-start,flush=True)
  finally:FormalData.item=original
  seal=dict(rows=rows,config_sha256=digest(RUN/'RUN_CONFIG.json'),unix=time.time())
  if (RUN/'PREDICTIONS_FROZEN.json').exists():seal=read(RUN/'PREDICTIONS_FROZEN.json');assert seal['rows']==rows
  else:write(RUN/'PREDICTIONS_FROZEN.json',seal)
  for row in rows:assert digest(folder/row['file'])==row['sha256']
  verified=time.time()
  write(RUN/'STATUS.json',dict(status='SCORING',unix=time.time()))
  scoring=FormalData(split,'scoring');acc={g:{b:MetricAccumulator() for b in ('B','Bp')} for g in ('all','N_ge_9')};windows=[]
  first=time.time()
  for pos,index in enumerate(ids):
   row=rows[pos];path=folder/row['file'];assert digest(path)==row['sha256']
   with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
   batch=scoring.batch([index],'heldout_confirmation',truth=True);n=int(batch.valid.sum())
   r=dict(window=row['window'],n=n,shift_max_m=float(np.linalg.norm(arrays['delta'],axis=-1).mean(1).max()))
   for b in ('B','Bp'):
    st=hard_metric_statistics(torch.from_numpy(arrays[b])[None],batch.future,batch.valid)
    acc['all'][b].update_statistics(st)
    if n>=9:acc['N_ge_9'][b].update_statistics(st)
    r.update({b+'_JADE':float(st.scene_jade[0]),b+'_JFDE':float(st.scene_jfde[0]),b+'_minADE_sum':float(st.agent_minade[st.agent_valid].double().sum()),b+'_minFDE_sum':float(st.agent_minfde[st.agent_valid].double().sum())})
   for m in ('JADE','JFDE'):r['delta_'+m]=r['Bp_'+m]-r['B_'+m]
   windows.append(r)
  metrics={g:{b:a.as_dict() for b,a in v.items()} for g,v in acc.items()}
  csvtext=subprocess.check_output(['git','show',BASE+':docs/research/continuous_joint_flow/zara2_frozen_test_20261122/PER_SEED.csv'],cwd=S,text=True)
  reference={r['group']:r for r in csv.DictReader(csvtext.splitlines()) if r['seed']=='2035'}
  bounds={}
  for key,limit in [('shift_max_m',.010),('delta_JFDE',.010),('delta_JADE',13/24*.010)]:
   vals=[abs(r[key]) for r in windows];bounds[key]=dict(theory_m=limit,maximum_m=max(vals),exceed=sum(v>limit for v in vals),exceed_fp32_1e_5=sum(v>limit+1e-5 for v in vals));assert not bounds[key]['exceed_fp32_1e_5']
  summary=[];lines=['# ZARA2 单折、单推理种子：1 cm 校准冻结测试','所有结果限于 ZARA2 单折、单推理种子 seed=2035，零训练。'];md=['| 分组 | 指标 | B（米） | B′（米） | B′−B（厘米） |','|---|---|---:|---:|---:|']
  for g in acc:
   a,b=metrics[g]['B'],metrics[g]['Bp']
   for m in METRICS:
    assert a[m]==float(reference[g]['B_'+m]),(g,m,a[m],reference[g]['B_'+m])
    change=b[m]-a[m];summary.append(dict(group=g,metric=m,windows=a['scene_count'],agents=a['agent_count'],B=a[m],Bp=b[m],difference_cm=100*change));md.append(f'| {g} | {m} | {a[m]:.12f} | {b[m]:.12f} | {100*change:+.9f} |')
   recovered=100*(a['minFDE']-b['minFDE'])/(a['minFDE']-float(reference[g]['A_minFDE']))
   lines.append(f"{g}：边缘 minFDE 改善 {100*(a['minFDE']-b['minFDE']):+.9f} cm，追回同种子 H0 差距 {recovered:.6f}%；JADE 改变 {100*(b['JADE']-a['JADE']):+.9f} cm，JFDE 改变 {100*(b['JFDE']-a['JFDE']):+.9f} cm。")
  lines.append('\n'.join(md))
  for key,v in bounds.items():lines.append(f"{key}：理论界 {v['theory_m']:.12f} m，最大绝对值 {v['maximum_m']:.12f} m，超理论界 {v['exceed']} 窗，超 FP32 绝对容差 1e−5 m 为 {v['exceed_fp32_1e_5']} 窗。")
  diagnostics=[d for r in rows for d in r['diagnostics']];fail=sum(not d['success'] for d in diagnostics)
  lines.append(f'固定算法/参数来自 {CAL}，测试协议来自 {BASE}。原 B 的998组轨迹逐文件SHA核验并原样复用；按相同冻结权重/2035随机流重放原始256候选和修正后20终点，与缓存B第12帧逐坐标核对 atol=1e-6 m。B′使用同一轨迹残差及场景/人员顺序，无新扩散采样。全部预测封存并核验后才读取测试未来。联合指标按窗口等权，边缘指标按有效行人人次加权。求解器未达收敛条件 {fail}/{len(diagnostics)} 次，所有返回解均可行且覆盖目标不增加，不宣称全局最优。')
  lines.append('[汇总CSV](SUMMARY.csv) · [匿名逐窗CSV](PER_WINDOW.csv) · [界核验](BOUNDS.json) · [配置](RUN_CONFIG.json)')
  table(DOC/'SUMMARY.csv',summary);table(DOC/'PER_WINDOW.csv',windows)
  for name,v in [('METRICS.json',metrics),('BOUNDS.json',bounds),('RUN_CONFIG.json',config),('SCORE_RECEIPT.json',dict(seal_unix=seal['unix'],verification_unix=verified,first_truth_unix=first,truth_reads=998,original_B_metrics_exact=True,prediction_manifest_sha256=digest(RUN/'PREDICTIONS_FROZEN.json'))),('SOLVER_DIAGNOSTICS.json',dict(calls=len(diagnostics),nonconverged=fail))]:write(DOC/name,v)
  (DOC/'README.md').write_text('\n\n'.join(lines)+'\n')
  subprocess.run(['git','add',str(DOC.relative_to(S))],cwd=S,check=True);subprocess.run(['git','commit','-m','Archive ZARA2 single-seed frozen calibration test'],cwd=S,check=True)
  branch=subprocess.check_output(['git','branch','--show-current'],cwd=S,text=True).strip();subprocess.run(['git','push','-u','origin',branch],cwd=S,check=True)
  sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=S,text=True).split()[0];assert sha==remote
  write(RUN/'DELIVERY.json',dict(sha=sha,report=f'https://github.com/WuYanXingege/RSJG/blob/{sha}/{DOC.relative_to(S)}/README.md'));write(RUN/'STATUS.json',dict(status='COMPLETE',unix=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  RUN.mkdir(parents=True,exist_ok=True);write(RUN/'STATUS.json',dict(status='INTERRUPTED_RESUME_SAME_RUN',error=traceback.format_exc()));raise
