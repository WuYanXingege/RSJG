"""Post-completion dev-only comparison. No training or automatic other-fold launch."""
import argparse,csv,io,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from bcep_v2.cost_trial import inputs,build,SOURCE
from bcep_v2.runtime import configure_cuda
from bcep_formal.data import FormalData
from bcep_joint.common import read,write
from bcep_v2.formal_manifest import write_csv
from bcep.contracts import digest
from ebjd.unified import gather_candidates
P='docs/research/bcep_v2/formal_fivefold_top5_20261118/'
def archived(commit,name):return subprocess.check_output(['git','show',commit+':'+P+name],cwd=SOURCE).decode()
def loadcsv(path):
 with path.open() as f:return list(csv.DictReader(f))
def main(a):
 assert read(a.run/'STATUS.json')['status']=='COMPLETE','Do not report unfinished training as complete'
 configure_cuda();torch.set_num_threads(1);cfg,rec,split,banks=inputs(a.upstream)
 history=read(a.run/'joint/EPOCH_METRICS.json');top=read(a.run/'joint/TOP5.json');best=top[0];assert best==sorted(top,key=lambda r:(r['validation'],r['epoch']))[0]
 for r in top:assert digest(a.run/'joint'/r['file'])==r['sha256']
 old=list(csv.DictReader(io.StringIO(archived('f44652730d0d7660c0eefed2cefa6bf939f1037c','zara2/joint/EPOCH_METRICS.csv'))))
 old_windows=list(csv.DictReader(io.StringIO(archived('4c9eede551bebdd7634ab81b7b4bd790165c6bb0','20261119_zara2_joint_endpoint_dev/WINDOW_METRICS.csv'))))
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 groups=[];windows=[]
 for r in history:
  ep=r['epoch'];groups.extend(read(a.run/'dev'/f'epoch_{ep:03d}_groups.json'));windows.extend(loadcsv(a.run/'dev'/f'epoch_{ep:03d}_windows.csv'))
 write_csv(out/'PER_EPOCH_GROUPS.csv',groups);write_csv(out/'PER_WINDOW.csv',windows);write(out/'TOP5.json',top);write_csv(out/'TOP5.csv',[dict(epoch=r['epoch'],update=r['update'],dev_H_JADE=r['validation'],file=r['file'],sha256=r['sha256']) for r in top])
 # The historical full-scene curve is archived. Recompute only endpoints for all
 # 41 historical checkpoints on the 127 high-count dev windows, never DDIM.
 data=FormalData(split,'scoring',verify=False);dense=[i for i in split['selection_dev_indices'] if len(split['members'][i]['agent_ids'])>=9]
 old_dense=[];cfg=dict(cfg,pair_backward_proxy='coordinate');model=build(cfg,'cuda').eval();model.coupling.capture_assignments=True
 cache=a.run/'old_dense_endpoint_audit';cache.mkdir(exist_ok=True)
 with torch.no_grad():
  for r in old:
   ep=int(r['epoch']);path=cache/f'epoch_{ep:03d}.json'
   if not path.exists():
    state=torch.load(a.upstream/f'zara2/joint/epoch_{ep:03d}.pt',map_location='cpu',weights_only=False);assert state['completed_epoch']==ep and state['identity']['dev_bank']==banks[1].hash
    model.load_state_dict(state['model']);del state;rr=[]
    for i in dense:
     x=data.batch([i],'selection_dev');g=banks[1].batch([i],x).cuda();x=x.to('cuda');context=model.encode_context(x.observed,x.semantic_maps,x.valid)
     h,pi,_=model.coupling.assign(context,g,model.bridge.representation.goal_scale,straight_through=False);initial=gather_candidates(g,model.coupling.last_initial_indices)
     jf=lambda z:float((z-x.future[:,None,:,-1]).norm(dim=-1).mean(-1).min())
     rr.append(dict(index=i,G_JFDE=jf(g),initial_JFDE=jf(initial),H_JFDE=jf(h)))
    write(path,dict(epoch=ep,windows=len(rr),checkpoint_sha256=digest(a.upstream/f'zara2/joint/epoch_{ep:03d}.pt'),H_JFDE=sum(x['H_JFDE'] for x in rr)/len(rr),initial_JFDE=sum(x['initial_JFDE'] for x in rr)/len(rr),rows=rr))
   old_dense.append(read(path))
 write_csv(out/'OLD_HIGH_COUNT_CURVE.csv',[{k:v for k,v in r.items() if k!='rows'} for r in old_dense])
 oldmetrics={int(r['epoch']):json.loads(r['metrics']) for r in old};od={r['epoch']:r for r in old_dense}
 new={(r['epoch'],r['group']):r for r in groups};ep=best['epoch'];n=new[ep,'all'];nh=new[ep,'N_ge_9'];comparison=[]
 for oe in [26,5,3]:
  for metric,nv,ov in [('H_JADE',n['JADE'],oldmetrics[oe]['JADE']),('H_JFDE',n['JFDE'],oldmetrics[oe]['JFDE']),('N_ge_9_H_JFDE',nh['JFDE'],od[oe]['H_JFDE'])]:comparison.append(dict(new_rank1_epoch=ep,old_epoch=oe,metric=metric,new_m=nv,old_m=ov,delta_m=nv-ov,delta_cm=100*(nv-ov)))
 write_csv(out/'RANK1_COMPARISON.csv',comparison)
 matched=[]
 for e in [3,5,26]:
  if (e,'all') not in new:continue
  for group in ['all','N_ge_9']:
   z=new[e,group];matched.append(dict(epoch=e,group=group,new_H_JADE=z['JADE'],new_H_JFDE=z['JFDE'],old_H_JADE=oldmetrics[e]['JADE'] if group=='all' else None,old_H_JFDE=oldmetrics[e]['JFDE'] if group=='all' else od[e]['H_JFDE']))
 write_csv(out/'MATCHED_EPOCHS.csv',matched)
 per=[]
 for r in windows:
  if int(r['epoch'])!=ep:continue
  idx=int(r['index']);oldh=next(float(v['JFDE_m']) for v in old_windows if int(v['epoch'])==26 and int(v['index'])==idx and v['method']=='H');oldi=next(float(v['JFDE_m']) for v in old_windows if int(v['epoch'])==26 and int(v['index'])==idx and v['method']=='initial_Hungarian')
  per.append(dict(index=idx,n_agents=int(r['n_agents']),new_H_JFDE=float(r['H_JFDE']),old_H_JFDE=oldh,delta_H_JFDE=float(r['H_JFDE'])-oldh,new_initial_JFDE=float(r['initial_JFDE']),old_initial_JFDE=oldi))
 write_csv(out/'RANK1_PAIRED_WINDOWS.csv',per)
 paired=[]
 for group in ['all','N_ge_9']:
  rr=[r for r in per if group=='all' or r['n_agents']>=9];d=np.array([r['delta_H_JFDE'] for r in rr]);paired.append(dict(group=group,windows=len(rr),threshold_m=1e-6,improved=int((d< -1e-6).sum()),worsened=int((d>1e-6).sum()),near_zero=int((abs(d)<=1e-6).sum()),mean_delta_m=float(d.mean()),initial_delta_m=float(np.mean([r['new_initial_JFDE']-r['old_initial_JFDE'] for r in rr])),feedback_delta_m=float(np.mean([(r['new_H_JFDE']-r['new_initial_JFDE'])-(r['old_H_JFDE']-r['old_initial_JFDE']) for r in rr]))))
 write_csv(out/'PAIRED_GROUPS.csv',paired)
 decision='本折证据有限'
 if nh['JFDE']>=od[26]['H_JFDE'] or n['JADE']>oldmetrics[26]['JADE']+.0012962305697344:decision='不建议扩折：高人数退步或路径代价明显'
 elif n['JFDE']<oldmetrics[26]['JFDE'] and nh['JFDE']<od[26]['H_JFDE'] and n['JFDE']<oldmetrics[5]['JFDE'] and nh['JFDE']<od[5]['H_JFDE'] and n['JADE']<=oldmetrics[5]['JADE']:decision='建议下一步扩折验证；本轮不自动扩折'
 costs=dict(train_seconds=sum(r['train_seconds'] for r in history),validation_seconds=sum(r['validation_seconds'] for r in history),epochs=len(history),updates=history[-1]['update'],stop_reason=read(a.run/'STATUS.json')['stop_reason'])
 write(out/'FINAL_RESULT.json',dict(decision=decision,new_rank1_epoch=ep,cost=costs,comparison=comparison,paired=paired))
 for filename in ['FROZEN_CONFIG.json','ACCEPTANCE.json']:write(out/filename,read(a.run/filename))
 lines=['# ZARA2 第二阶段候选代价代理受控训练','',f'**判断：{decision}。**','',f'新模型按开发集H-JADE选中第{ep}轮。真实训练{costs["updates"]}次更新、{costs["epochs"]}轮；停止原因{costs["stop_reason"]}。训练耗时{costs["train_seconds"]/3600:.2f}小时，验证{costs["validation_seconds"]/3600:.2f}小时。','', '| 新rank1对照 | 指标 | 新值 m | 旧值 m | 新减旧 cm |','|---|---|---:|---:|---:|']
 lines += [f"| 旧第{r['old_epoch']}轮 | {r['metric']} | {r['new_m']:.8f} | {r['old_m']:.8f} | {r['delta_cm']:+.4f} |" for r in comparison]
 lines+=['','初次配对与反馈贡献（负值表示新模型改善）：','']
 for r in paired:lines.append(f"- {r['group']}：初次配对变化{r['initial_delta_m']:+.6f} m，反馈增益变化{r['feedback_delta_m']:+.6f} m；最终配对改善/恶化/近零窗口为{r['improved']}/{r['worsened']}/{r['near_zero']}（阈值1e-6 m）。")
 lines+=['','该分解是各自训练后模型前向的描述，不能独立证明反馈模块的训练因果。相邻窗口可能重叠；只有一个训练种子，结论只支持本折受控比较。没有用测试集开发或选模，没有启动其他折。','', '旧五折队列仍为PAUSED_RESUMABLE。验收合成步骤单列于ACCEPTANCE.json，真实更新数只取本试验计数。','', '[逐轮分组CSV](PER_EPOCH_GROUPS.csv) · [Top5](TOP5.csv) · [逐窗比较](RANK1_PAIRED_WINDOWS.csv) · [同轮比较](MATCHED_EPOCHS.csv)']
 (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(decision=decision,cost=costs),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
