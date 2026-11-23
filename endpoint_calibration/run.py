"""Fixed 1cm output-only calibration; CPU inference leaves the training GPU alone."""
import os, json, csv, time, subprocess, fcntl, traceback
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from bcep.contracts import digest
from bcep_joint.common import read, write
from joint_flow_dependency.inputs import Inputs
from joint_flow_dependency.model import EdgeDependence
from joint_flow_dependency.inference import generate
from ebjd.metrics import hard_metric_statistics, MetricAccumulator
SOURCE=Path(__file__).resolve().parents[1]
ROOT=SOURCE.parent
RUN=ROOT/'RSJG_EBJD_assets/experiments/zara2_endpoint_calibration_1cm_20261123'
DOC=SOURCE/'docs/research/continuous_joint_flow/zara2_calibration_1cm_20261123'
WEIGHT=ROOT/'RSJG_EBJD_assets/experiments/zara2_joint_flow_dependency_20261122/dependence/epoch_035.pt'
SHA='ea13cd7fd99a02c44c3c9af9a14a0cc9b5166b629e5c338694cca97f1c8c502a'
CONFIG=dict(seed=2035,M=256,K=20,DDIM=20,eta=0,updates=0,budget_m=.010,solver='scipy SLSQP float64, analytic gradients',ftol=1e-10,maxiter=300,fp32_atol_m=1e-5,device='cpu',threads=2,checkpoint_sha256=SHA,assignment='fixed nearest corrected endpoint; tie smaller original sample ID',fallback='feasible zero if objective increases; radial budget correction',base_commit='d4fbe6d8e3b142645b9e801116aac308fcb19c9d')
def table(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def calibrate(raw,goals,ids):
 raw,goals=raw.astype(np.float64),goals.astype(np.float64)
 n=goals.shape[1];order=np.argsort(ids,kind='stable')
 labels=order[np.linalg.norm(raw[:,None]-goals[order][None],axis=-1).argmin(axis=1)]
 delta=np.zeros_like(goals);diagnostics=[]
 for k in range(20):
  clusters=[raw[labels[:,i]==k,i]-goals[k,i] for i in range(n)]
  active=[i for i,c in enumerate(clusters) if len(c)]
  def fun(x):
   x=x.reshape(-1,2);value=0.;grad=np.zeros_like(x)
   for j,i in enumerate(active):
    diff=x[j]-clusters[i];norm=np.linalg.norm(diff,axis=1)
    value+=norm.sum();grad[j]=(diff/np.maximum(norm[:,None],1e-15)).sum(0)
   return value,grad.ravel()
  budget=n*.010
  def constraint(x):return budget-np.linalg.norm(x.reshape(-1,2),axis=1).sum()
  def jac(x):
   a=x.reshape(-1,2);return (-a/np.maximum(np.linalg.norm(a,axis=1)[:,None],1e-15)).ravel()
  if not active:continue
  x0=np.zeros(2*len(active));initial=fun(x0)[0]
  result=minimize(fun,x0,jac=True,method='SLSQP',constraints=[dict(type='ineq',fun=constraint,jac=jac)],options=dict(ftol=CONFIG['ftol'],maxiter=CONFIG['maxiter']))
  x=result.x.reshape(-1,2);mass=np.linalg.norm(x,axis=1).sum()
  if not np.isfinite(x).all():x[:]=0
  elif mass>budget:x*=budget*(1-1e-12)/mass
  if fun(x.ravel())[0]>initial:x[:]=0
  delta[k,active]=x
  diagnostics.append(dict(initial=float(initial),final=float(fun(x.ravel())[0]),success=bool(result.success),iterations=int(result.nit)))
 assert np.max(np.linalg.norm(delta,axis=-1).mean(1))<=.010+1e-12
 return delta,diagnostics

def main():
 RUN.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
 with (RUN/'run.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
  config=dict(CONFIG,source_sha256=digest(__file__))
  if (RUN/'RUN_CONFIG.json').exists():assert read(RUN/'RUN_CONFIG.json')==config
  else:write(RUN/'RUN_CONFIG.json',config)
  assert digest(WEIGHT)==SHA
  inputs=Inputs();model,provider=inputs.load_frozen('cpu');dep=EdgeDependence()
  saved=torch.load(WEIGHT,map_location='cpu',weights_only=False);assert saved['completed_epoch']==35
  dep.load_state_dict(saved['model'],strict=True)
  for module in (model,provider,dep):module.eval().requires_grad_(False)
  indices=inputs.split['selection_dev_indices'];assert len(indices)==590
  folder=RUN/'predictions';folder.mkdir(exist_ok=True);manifest=[];started=time.time()
  # The observation reader is guarded for the entire generation/calibration pass.
  original=inputs.data.item
  def guard(index,purpose,truth=True):
   assert purpose=='selection_dev' and not truth
   return original(index,purpose,truth=False)
  inputs.data.item=guard
  for pos,index in enumerate(indices):
   path=folder/f'{pos:04d}.npz';receipt=path.with_suffix('.json')
   if receipt.exists():
    row=read(receipt);assert digest(path)==row['sha256'] and row['config']==config
   else:
    rawpath=folder/f'{pos:04d}_base.npz'
    if rawpath.exists():
     meta=read(rawpath.with_suffix('.json'));assert meta['config']==config and meta['sha256']==digest(rawpath)
     with np.load(rawpath) as z:base={k:z[k].copy() for k in z.files}
    else:
     batch=inputs.data.batch([index],'selection_dev',truth=False)
     with torch.no_grad():y,e,_=generate(model,provider,dep,batch)
     base=dict(Y=y[0].numpy(),raw=e['raw'].numpy(),g=e['corrected'][0].numpy(),ids=e['raw_indices'].numpy(),noise=e['noise'].numpy())
     tmp=rawpath.with_suffix('.tmp.npz');np.savez_compressed(tmp,**base);tmp.replace(rawpath)
     write(rawpath.with_suffix('.json'),dict(config=config,sha256=digest(rawpath)))
    d,diag=calibrate(base['raw'],base['g'],base['ids'])
    yp=(base['Y'].astype(np.float64)+np.arange(1,13)[None,None,:,None]/12*d[:,:,None]).astype(np.float32)
    assert np.max(np.abs(yp[:,:,-1]-(base['g']+d)))<=1e-5
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,B=base['Y'],Bp=yp,delta=d);tmp.replace(path)
    row=dict(window=f'dev_{pos+1:04d}',sha256=digest(path),config=config,diagnostics=diag)
    write(receipt,row)
   manifest.append(dict(window=row['window'],file=path.name,sha256=row['sha256']))
   elapsed=time.time()-started
   write(RUN/'STATUS.json',dict(status='PREDICTING_WITHOUT_TRUTH',completed=pos+1,total=590,seconds=elapsed,estimated_remaining_seconds=elapsed/(pos+1)*(590-pos-1)))
   print('prediction',pos+1,elapsed,flush=True)
  inputs.data.item=original
  write(RUN/'PREDICTIONS_FROZEN.json',dict(rows=manifest,unix=time.time(),config=config))
  for row in manifest:assert digest(folder/row['file'])==row['sha256']
  acc={g:{b:MetricAccumulator() for b in ('B','Bp')} for g in ('all','N_ge_9')};rows=[]
  for pos,index in enumerate(indices):
   path=folder/manifest[pos]['file'];assert digest(path)==manifest[pos]['sha256']
   with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
   batch=inputs.data.batch([index],'selection_dev',truth=True);n=int(batch.valid.sum());r=dict(window=f'dev_{pos+1:04d}',n=n,mean_endpoint_shift_max=float(np.linalg.norm(arrays['delta'],axis=-1).mean(1).max()))
   for b in ('B','Bp'):
    s=hard_metric_statistics(torch.from_numpy(arrays[b])[None],batch.future,batch.valid)
    acc['all'][b].update_statistics(s)
    if n>=9:acc['N_ge_9'][b].update_statistics(s)
    r.update({b+'_JADE':float(s.scene_jade[0]),b+'_JFDE':float(s.scene_jfde[0]),b+'_minADE':float(s.agent_minade[s.agent_valid].double().mean()),b+'_minFDE':float(s.agent_minfde[s.agent_valid].double().mean())})
   for m in ('JADE','JFDE','minADE','minFDE'):r['delta_'+m]=r['Bp_'+m]-r['B_'+m]
   rows.append(r)
  metrics={g:{b:a.as_dict() for b,a in v.items()} for g,v in acc.items()}
  assert metrics['all']['B']['scene_count']==590 and metrics['N_ge_9']['B']['scene_count']==127
  for g,reference in [('all',.457311081213),('N_ge_9',.493647037634)]:assert abs(metrics[g]['B']['minFDE']-reference)<1e-6,(g,metrics[g]['B']['minFDE'])
  bounds={}
  for key,limit in [('mean_endpoint_shift_max',.010),('delta_JFDE',.010),('delta_JADE',13/24*.010)]:
   values=[abs(r[key]) for r in rows];bounds[key]=dict(theoretical_m=limit,maximum_m=max(values),exceed=sum(v>limit for v in values),exceed_fp32_tolerance=sum(v>limit+1e-5 for v in values));assert not bounds[key]['exceed_fp32_tolerance']
  summary=[];answers=[]
  for g,h0 in [('all',.352042235180),('N_ge_9',.382158567245)]:
   a,b=metrics[g]['B'],metrics[g]['Bp'];gain=a['minFDE']-b['minFDE'];recovered=100*gain/(a['minFDE']-h0)
   answers.append(f'{g}：真实边缘 minFDE 改善 {gain*100:+.8f} cm；相对 H0 的差距追回 {recovered:.6f}%。')
   for m in ('JADE','JFDE','minADE','minFDE'):summary.append(dict(group=g,metric=m,B=a[m],Bp=b[m],difference=b[m]-a[m],windows=a['scene_count'],agents=a['agent_count']))
  table(DOC/'SUMMARY.csv',summary);table(DOC/'PER_WINDOW.csv',rows)
  for name,value in [('RUN_CONFIG.json',config),('METRICS.json',metrics),('BOUNDS.json',bounds)]:write(DOC/name,value)
  diagnostics=[d for p in sorted(folder.glob('[0-9][0-9][0-9][0-9].json')) for d in read(p)['diagnostics']]
  write(DOC/'SOLVER_DIAGNOSTICS.json',dict(calls=len(diagnostics),converged=sum(d['success'] for d in diagnostics),initial_objective=sum(d['initial'] for d in diagnostics),final_objective=sum(d['final'] for d in diagnostics)))
  text='\n\n'.join(answers)+f"\n\n最大窗口 |ΔJFDE|={bounds['delta_JFDE']['maximum_m']:.10f} m，|ΔJADE|={bounds['delta_JADE']['maximum_m']:.10f} m。精确界/容差超界计数见 BOUNDS.json。\n\n训练更新为零；全部590窗B/B′预测封存后才读取开发集真值。CPU原生指标，与归档B边缘minFDE偏差小于1e-6m。每窗只生成一次原B轨迹，候选、原始场景编号和桥噪声保存在主机缓存。SLSQP float64从零初值求解，固定分配；非收敛状态单列诊断，不宣称全局最优，接受解均可行且覆盖目标不增。候选覆盖目标仅为诊断，实际收益以上述真实指标为准。未访问测试集，未更改四折任务。\n\n[汇总表](SUMMARY.csv) · [匿名逐窗](PER_WINDOW.csv) · [界核验](BOUNDS.json) · [固定配置](RUN_CONFIG.json)\n"
  (DOC/'README.md').write_text(text)
  write(RUN/'STATUS.json',dict(status='COMPLETE',unix=time.time()))
  subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],cwd=SOURCE,check=True)
  subprocess.run(['git','commit','-m','Archive fixed 1cm ZARA2 calibration results'],cwd=SOURCE,check=True)
  branch=subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip()
  subprocess.run(['git','push','-u','origin',branch],cwd=SOURCE,check=True)
  sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=SOURCE,text=True).split()[0];assert sha==remote
  write(RUN/'DELIVERY.json',dict(sha=sha,report=f'https://github.com/WuYanXingege/RSJG/blob/{sha}/{DOC.relative_to(SOURCE)}/README.md'))
if __name__=='__main__':
 try:main()
 except BaseException:
  RUN.mkdir(parents=True,exist_ok=True);write(RUN/'STATUS.json',dict(status='INTERRUPTED_RESUME_SAME_RUN',error=traceback.format_exc()));raise
