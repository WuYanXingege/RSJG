"""Summarize the single preregistered comparison; no further model execution."""
import argparse,csv,io,json,hashlib,subprocess
from pathlib import Path

def read(p):return json.loads(p.read_text())
def csvread(p):
 with p.open() as f:return list(csv.DictReader(f))
def csvwrite(p,rr):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
def table(h,rr):return ['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rr]+['']
def main(a):
 out=Path(__file__).resolve().parent;qa=read(out/'QA.json');assert qa['status']=='PASS';stop=read(out/'STOP_RECEIPT.json');groups=csvread(out/'GROUP_SUMMARY.csv');pairs=csvread(out/'PAIRED_WINDOWS.csv')
 lookup={(r['split'],int(r['epoch']),r['group']):r for r in groups};delta=lambda s,e,g:float(lookup[s,e,g]['delta_initial'])
 reversal=stop['rank1_epoch']!=1 # Must be inspected before any automatic A/B classification.
 if reversal:decision='C';conclusion='最新rank1发生变化，需依据新指标保留最窄结论；本诊断不追加算力。'
 elif all(delta(s,e,'all')>.01 for s in ['train','selection_dev'] for e in [3,5]) and all(delta(s,e,'N_ge_9')>=-.01 for s in ['train','selection_dev'] for e in [3,5]):decision='A';conclusion='新代理在训练样本上也出现初次硬配对退步。'
 elif all(delta('selection_dev',e,'all')>.01 and delta('train',e,'all')<=.01 and delta('selection_dev',e,'N_ge_9')>=-.01 and delta('train',e,'N_ge_9')<=.01 for e in [3,5]):decision='B';conclusion='开发窗口的退步未在固定训练窗口中同样出现。'
 else:decision='C';conclusion='预设两轮及分组未形成一致的训练侧/开发侧初次配对退步证据。'
 nextstep={'A':'恢复旧坐标代理作为后续研究基线','B':'沿用旧v2基线，仅在既有结果内按录制和人数定位贡献','C':'保留现有基线和两套资产，结束这条验证'}[decision]
 prefix='docs/research/bcep_v2/formal_fivefold_top5_20261118/'
 def gittext(commit,path):return subprocess.check_output(['git','show',commit+':'+prefix+path],cwd=a.repo,text=True)
 old=list(csv.DictReader(io.StringIO(gittext('f44652730d0d7660c0eefed2cefa6bf939f1037c','zara2/joint/EPOCH_METRICS.csv'))));om={int(r['epoch']):json.loads(r['metrics']) for r in old}
 oldw=list(csv.DictReader(io.StringIO(gittext('4c9eede551bebdd7634ab81b7b4bd790165c6bb0','20261119_zara2_joint_endpoint_dev/WINDOW_METRICS.csv'))))
 current=[dict(model='new_current_rank1',epoch=stop['rank1_epoch'],H_JADE=stop['rank1_H_JADE'],H_JFDE=stop['rank1_H_JFDE'],dense_H_JFDE=stop['rank1_dense_H_JFDE'])]
 for e in [26,3,5]:
  rr=[r for r in oldw if int(r['epoch'])==e and r['method']=='H' and int(r['n_agents'])>=9]
  current.append(dict(model='old_rank1' if e==26 else 'old_reference',epoch=e,H_JADE=om[e]['JADE'],H_JFDE=om[e]['JFDE'],dense_H_JFDE=sum(float(r['JFDE_m']) for r in rr)/len(rr)))
 csvwrite(out/'LATEST_RANK1_COMPARISON.csv',current)
 trial=a.workspace/'RSJG_EBJD_assets/experiments/zara2_costproxy_stage2_20261119';latest=[]
 for ep in range(1,stop['completed_epoch']+1):
  for g in read(trial/'dev'/f'epoch_{ep:03d}_groups.json'):latest.append(dict(epoch=ep,group=g['group'],H_JADE=g['JADE'],H_JFDE=g['JFDE'],initial_JFDE=g['initial_JFDE']))
 csvwrite(out/'LATEST_COMPLETED_EPOCHS.csv',latest)
 # Distribution details only on the B branch; no new scoring or sample selection.
 if decision=='B':
  upstream=a.workspace/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated';formal=read(upstream/'FORMAL_RUN.json');plan=read(Path(formal['input_plan']));split=read(Path(plan['folds']['zara2']['split']['path']));names=sorted({str(m['sequence']) for m in split['members']});bins={}
  for r in pairs:
   key=(r['split'],int(r['epoch']),names.index(str(split['members'][int(r['index'])]['sequence']))+1,'N_ge_9' if int(r['n_agents'])>=9 else 'N_lt_9');bins.setdefault(key,[]).append(r)
  csvwrite(out/'RECORDING_CONTRIBUTIONS.csv',[dict(split=k[0],epoch=k[1],anonymous_recording=k[2],people_group=k[3],windows=len(rr),delta_initial_sum_m=sum(float(r['delta_initial']) for r in rr),delta_initial_mean_m=sum(float(r['delta_initial']) for r in rr)/len(rr)) for k,rr in bins.items()])
 L=['# ZARA2 可恢复暂停与固定训练/开发窗口诊断','',f'**停在update {stop["saved_update"]}（完整{stop["completed_epoch"]}轮，第{stop["current_epoch"]}轮游标{stop["cursor"]}）；{conclusion}下一步只建议：{nextstep}。**','',f'预先规定的决策分支为 **{decision}**，工程量级为0.01 m，无统计显著性含义。本轮完成这一诊断后结束，未实施网络改动或开启任何后续训练。','',
'## 最新现场与停机收据','',
'15:18以后又完成了第11轮，但H-JADE rank1仍为第1轮，未出现实质反转。第11轮和此前所有完整轮次见LATEST_COMPLETED_EPOCHS.csv；没有把第12轮的部分更新当作完成轮次。','',
f'从实时运行清单读取PID并验证命令行、实际工作目录和试验根目录，再向该试验原生信号处理器发送一次SIGTERM；没有使用旧报告PID或按名称批量杀进程。首次观察update {stop["first_observed_update_at_request"]}，发信号前最后一条周期进度{stop["observed_update_before_signal"]}，最终保存{stop["saved_update"]}。周期日志不等于精确发信号时刻，不能把差值解释为信号后更新数。**原试验真实更新累计1903，本次诊断新增训练更新0。**','',
'现场实际发生了按研究决策提前暂停，而非自然早停。保留原生SESSION_BUDGET_PAUSED状态，在分析收据中登记PAUSED_RESUMABLE_BY_RESEARCH_DECISION。恢复快照包含model、optimizer、scheduler、RNG、epoch/update、order/cursor、stale及轮内统计。last、best、11个完整epoch和Top5 [1,7,11,10,9]全部保留并校验哈希。','',
'训练进程退出，试验锁与旧控制器锁释放；本试验原有完成后处理等待进程已核实身份后取消，因此不会自动追加旧高人数曲线计算或报告任务。旧五折队列继续保持PAUSED_RESUMABLE，ZARA1/head的7097恢复点及其Top5未改动。','',
'## 最新新旧rank1与历史参照','']
 L+=table(['权重','epoch','H-JADE m','H-JFDE m','≥9人 H-JFDE m'],[[r['model'],r['epoch'],f"{r['H_JADE']:.6f}",f"{r['H_JFDE']:.6f}",f"{r['dense_H_JFDE']:.6f}"] for r in current])
 L+=['最新新rank1相对旧正式rank1的终点误差虽较低，但H-JADE明显更差；旧第3轮在这三项指标上都优于新rank1。没有依据新数据推翻此前“尚未超过已有折中”的判断。本轮不再追加轮次来改变这一结论。','',
'## 一次固定成对比较的输入与计分','',
'严格加载旧coordinate与新candidate_cost的实际epoch 1、3、5权重，六份均存在，完成轮、恢复状态、train/dev bank身份一致，哈希见CHECKPOINTS.json。没有使用相邻轮次替代或补训。','',
'使用原train 2681窗及selection_dev 590窗（127窗≥9人），seed 2035，K=20，米制绝对终点。训练窗口采用与bank对应的固定原始视图，关闭增广；窗口索引、帧、有效行人顺序由Bank.batch和记录哈希核对。新旧在同一split内逐窗配对，train与dev分开汇总。','',
'旧dev epoch3/5复用4c9eede的已核准结果，新dev三个轮次复用该试验完整轮次验证记录；其余缺项在eval/inference_mode下通过实际assign与gather_candidates计算。初次与最终配对来自同次前向。真值仅供计分；没有轨迹桥、DDIM、optimizer.step、候选重导、超参数扫描或测试集读取。','',
'JFDE=min_k mean_i ||g[k,i]−y[i]||，然后对窗口等权。反馈收益=I−H（正值表示改善）；初次差值=I_new−I_old、最终差值=H_new−H_old（正值表示新版退步）。','',
'## 全部原始分组结果','']
 L+=table(['split','epoch','人数','窗数','旧G/I/H m','新G/I/H m','旧/新反馈收益 m','Δ初次 m','Δ最终 m'],[[r['split'],r['epoch'],r['group'],r['windows'],' / '.join(f"{float(r['old_'+k]):.6f}" for k in ['G_JFDE','I_JFDE','H_JFDE']),' / '.join(f"{float(r['new_'+k]):.6f}" for k in ['G_JFDE','I_JFDE','H_JFDE']),f"{float(r['old_feedback_gain']):+.6f} / {float(r['new_feedback_gain']):+.6f}",f"{float(r['delta_initial']):+.6f}",f"{float(r['delta_final']):+.6f}"] for r in groups])
 L+=['epoch1用于交代起点；以下只用事先指定的epoch3、5进行主判断，不挑选窗口或调整阈值。','']
 L+=table(['epoch','train Δ初次 m','dev Δ初次 m','train≥9 Δ初次 m','dev≥9 Δ初次 m'],[[e]+[f'{delta(s,e,g):+.8f}' for s,g in [('train','all'),('selection_dev','all'),('train','N_ge_9'),('selection_dev','N_ge_9')]] for e in [3,5]])
 L+=['第5轮，新版反馈收益相对旧版在train增加0.039646 m、在dev增加0.016914 m，抵消了部分初次退步，但最终H仍分别退步0.045305 m和0.054866 m。因此直接证据定位到初次硬配对，不支持把全部退步简单归因于邻居反馈失效。','', '没有将train与dev的绝对误差相减来判定过拟合。train分数是固定原始视图上的冻结硬配对表现，不等同于训练时随机增广视图的softmin损失；这里回答的是同一固定窗口内新旧模型的初次硬配对差异。','',
'## 数值核验与结论边界','']
 L+=table(['核验','最大偏差 m'],[['单人窗G=I=H',f"{qa['single_person_max_difference_m']:.3e}"],['缓存G/逐人下界与当前bank计分',f"{qa['cached_vs_bank_max_difference_m']:.3e}"],['Δ最终=Δ初次−(新收益−旧收益)',f"{qa['decomposition_max_residual_m']:.3e}"]])
 L+=['容差为1e−6 m。同一候选的G与逐人最小误差跨权重不变；新计算最终排列通过逐人双射检查。全部输入资产的文件清单、大小及修改时间在计分前后完全一致。发布逐窗表只含窗口索引、人数及误差，不含行人身份、坐标、权重、bank或内部绝对路径。','']
 if decision=='A':L+=['**唯一建议：恢复旧坐标代理作为后续研究基线。** 第3、5轮中train和dev的全部窗口Δ初次都超过+0.01 m，高人数子组没有实质反向，因此训练样本中也能观察到初次硬配对退步。这定位了退步发生的环节，不证明梯度代理是所有误差的唯一原因，也不是某个新监督设计的效果验证。','',
'若项目之后决定再做模型改动，唯一候选是给第一次Hungarian输出增加直接的场景级联合终点监督，沿用旧版可微代理，保留最终配对监督与邻居反馈。本轮不实施、不试跑，也不启动旧基线训练。']
 elif decision=='B':L+=['**唯一建议：以旧v2为基线，在已有逐窗结果中按录制及人数解释贡献。** RECORDING_CONTRIBUTIONS.csv已给出匿名录制分组；本轮不认定过拟合、不增加网络层数、不追加计算。']
 else:L+=['**唯一建议：保留现有基线和两套资产，结束这条验证。** 预设A/B条件没有完整成立或最新rank1需要优先修正判断，不追加轮次、窗口或阈值以形成结论。']
 L+=['','## 交付','', '[逐窗指标](WINDOW_METRICS.csv) · [逐窗成对差](PAIRED_WINDOWS.csv) · [分组CSV](GROUP_SUMMARY.csv) · [停机收据](STOP_RECEIPT.json) · [最新rank1对照](LATEST_RANK1_COMPARISON.csv) · [核验](QA.json) · [输入身份](IDENTITY.json)','',
'复算命令（参数只指向训练主机已有资产；不含恢复训练指令）：','', '```bash','python diagnose.py --workspace "$RSJG_WORKSPACE" --repo "$REPORT_CHECKOUT" --out "$DIAGNOSTIC_OUTPUT"','python report.py --workspace "$RSJG_WORKSPACE" --repo "$REPORT_CHECKOUT"','```','',
'复算时将两个脚本放在同一输出目录。当前计划已经执行完毕，不需要再运行；所有模型资产留在训练主机。**本轮诊断新增更新=0，原五折队列保持暂停，完成即结束。**']
 (out/'REPORT_ZH.md').write_text('\n'.join(L)+'\n');(out/'DECISION.json').write_text(json.dumps(dict(branch=decision,conclusion=conclusion,next_step=nextstep,threshold_m=.01,rank1_reversal=reversal,diagnostic_optimizer_updates=0),ensure_ascii=False,indent=2)+'\n');print(decision,conclusion);print([(e,delta('train',e,'all'),delta('selection_dev',e,'all')) for e in [3,5]])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);main(p.parse_args())
