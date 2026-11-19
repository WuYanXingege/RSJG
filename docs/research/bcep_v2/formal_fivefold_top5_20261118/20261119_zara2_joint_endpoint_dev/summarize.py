import csv,json,math,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
O=Path(__file__).resolve().parent
R=json.loads((O/'RESULT.json').read_text());receipt=json.loads((O/'STOP_RECEIPT.json').read_text())
def rows(name):
 with (O/name).open() as f:return list(csv.DictReader(f))
def save(name,rr):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
w=rows('WINDOW_METRICS.csv');inv=rows('PERMUTATION_CHECKS.csv');summary=R['summary'];effects=[];random=[]
for epoch in [26,3,5]:
 for subset in ['all','single','multi']:
  selected=[r for r in w if int(r['epoch'])==epoch and (subset=='all' or r['subset']==subset)]
  by={}
  for r in selected:by.setdefault(r['index'],{})[r['method']]=float(r['JFDE_m'])
  for a,b in [('G','initial_Hungarian'),('initial_Hungarian','H'),('G','H')]:
   d=np.array([r[b]-r[a] for r in by.values()])
   effects.append(dict(epoch=epoch,subset=subset,comparison=b+' minus '+a,windows=len(d),mean_delta_m=float(d.mean()),improved=int((d< -1e-7).sum()),worsened=int((d>1e-7).sum()),unchanged=int((abs(d)<=1e-7).sum())))
  means=[np.mean([r[f'shuffle_repeat_{j:02d}'] for r in by.values()]) for j in range(20)]
  random.append(dict(epoch=epoch,subset=subset,repetitions=20,mean_m=float(np.mean(means)),replicate_sd_m=float(np.std(means,ddof=1)),min_m=float(min(means)),max_m=float(max(means))))
save('PAIRED_EFFECTS.csv',effects);save('SHUFFLE_SUMMARY.csv',random)
get=lambda e,s,m:next(r['JFDE_m'] for r in summary if (r['epoch'],r['subset'],r['method'])==(e,s,m))
lines=['ZARA2 联合终点只读诊断与可恢复停机报告',f'完成时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}','',
'正式五折队列保持 PAUSED_RESUMABLE；未恢复队列，未修改网络或正式训练源码。本轮诊断的 optimizer.step 次数为 0。',
'注意“0 更新”的边界：接到请求时原训练仍在运行。发信号前 status 的周期进度为 update 7090，原进程在安全边界保存为 7097。首次status检查为update 7000；检查至退出期间原训练确实仍推进。这 7 个编号的差不能精确解释为信号后更新数，因为进度日志有间隔；不能声称从首次检查到退出的原运行更新数为 0。没有为诊断启动、续训或补跑任何训练。','',
'停机证据',
'实验 bcep_ebjd_v2_formal_top5_20261118；源码 commit 468c5eddde48bdeec47c4bb3c39418352be16bb9。FORMAL_RUN.json、STATUS.json 和正式 status 命令交叉核对。',
'控制器 1922379 的 /proc/cmdline 明确包含正式运行根目录；仅向该控制器发送一次 SIGTERM。控制器按原实现通知子进程 1947217。两者已退出，controller.lock、zara1/.lock、zara2/.lock 均可重新获得并已立即释放。',
'暂停于 ZARA1 / Stage 1 Provider / head。last.pt：update 7097，当前 epoch 10，completed_epoch 9，cursor 2360/2889，状态 SESSION_BUDGET_PAUSED。根状态为 PAUSED_RESUMABLE；本阶段是训练阶段，无候选导出游标。',
'快照实际包含 model、optimizer、scheduler、epoch/update、python/numpy/cpu/cuda/order/view RNG、order/cursor、history/stale 与轮内累计状态。仅检查恢复结构及身份，未执行恢复测试，因为恢复训练会违背本轮 0 诊断训练更新。',
'五个已产生阶段的 Top5 共 25 个文件均存在、哈希吻合，按原 validation 值及较早 epoch 排序。保留所有已完成 epoch、best、last、bank 和既有 ZARA2 结果。正式运行目录 '+str(R['protected_file_count'])+' 个文件的大小和 mtime 在诊断前后完全一致；被使用 checkpoint、bank、开发集记录另有哈希检查。',
'现行 selector 如实沿用：density 为 dev_nll_final_agent_micro，head 为 dev_minfde20_agent_micro，joint 为 dev_H_JADE；本轮不重排官方选择，不以 JFDE 改选权重。','',
'数据、单位与复现',
'官方 selection_dev 590 窗口、4262 次有效行人出现；不是 4262 个去重身份。每窗完整有效行人，无填充行参与分母。seed 2035 refined bank，K=20，原始 kmeans 中心顺序；窗口、帧、行人身份逐项对齐，并核验所有相关 SHA256。',
'终点和真值均为米制绝对坐标，没有再次乘除训练尺度。仅使用未来第 12 帧作计分；模型配对前向只接收观测、地图及既有候选。',
'JFDE(w)=min_k mean_i ||G[w,pi_i(k),i]-y[w,i]||，再对窗口等权；逐人下界 L(w)=mean_i min_j ||G[w,j,i]-y[w,i]||。它允许真值为每人选不同候选，只是覆盖下界，不是可部署模型。',
'第 3、5、26 轮完整权重均存在，已实际加载；未重建历史权重。最终采用GPU FP32前向、CPU计分及梯度检查；所有参数冻结。复现如下：',
'epoch | archive H-JFDE | reproduced H-JFDE | abs delta (m)']
for c in R['checkpoints']:lines.append(f"{c['epoch']} | {c['archive_JFDE']:.10f} | {c['reproduced_H_JFDE']:.10f} | {c['archive_abs_delta']:.3g}")
lines+=['第26轮逐人下界复算 '+f'{get(26,"all","oracle"):.10f}'+' m，归档值 0.33942601943420153 m。','',
'同候选场景等权结果（m）',
'原始 G 是 bank 原编号组成的世界；初次 Hungarian 是同一训练后权重、邻居消息加入前的分数配对；H 是加入一轮邻居反馈后的最终配对。初次结果并非独立训练的消融模型。',
'独立打乱对照从原 G 出发，各行人独立均匀排列20个候选，每窗20次，种子2035+window_index。表内先对20次均值，再对窗口等权；SHUFFLE_SUMMARY.csv给出重复之间的标准差与范围，它不是数据集置信区间。',
'epoch | subset | windows | G | 初次Hungarian | H | 独立打乱 | 逐人下界']
for epoch in [26,3,5]:
 for subset in ['all','single','multi']:
  n=next(r['windows'] for r in summary if r['epoch']==epoch and r['subset']==subset)
  lines.append(f'{epoch} | {subset} | {n} | '+' | '.join(f'{get(epoch,subset,m):.6f}' for m in ['G','initial_Hungarian','H','independent_shuffle','oracle']))
lines+=['','每人第1/2/5近候选误差（候选集合对三轮相同；单行明细见AGENT_NEAREST.csv）','subset | rank | windows | agent occurrences | scene macro (m) | agent micro (m)']
for r in R['nearest']:lines.append(f"{r['subset']} | {r['rank']} | {r['windows']} | {r['agents']} | {r['scene_macro_m']:.6f} | {r['agent_micro_m']:.6f}")
lines+=['','世界编号与实际跨人对应关系',
'统一对所有人应用同一个世界行重排，仅改变标签，不改变联合世界集合，JFDE 必须不变。所有窗口均已验证 H 与 H_global_relabel 完全一致；单人窗口的 G、初次、H、打乱、下界也一致（阈值1e-7 m）。',
'为了排除单纯换名，将每个配对表按第一个有效行人的原候选编号规范化后再比较；只在该相对配对表不同的窗口计为跨人关系改变。']
for epoch in [26,3,5]:
 rr=[r for r in inv if int(r['epoch'])==epoch and int(r['n_agents'])>1]
 changed=sum(int(r['initial_vs_H_relative_entries'])>0 for r in rr)
 raw=sum(int(r['initial_vs_H_raw_entries'])>0 for r in rr)
 lines.append(f'第{epoch}轮：多人窗{len(rr)}，初次→H原始表变化{raw}窗，消除统一编号后仍变化{changed}窗。')
lines+=['','成对变化（负值为改善；阈值1e-7 m）','epoch | subset | comparison | mean delta (m) | improved / worsened / unchanged']
for r in effects:
 if r['subset']=='multi':lines.append(f"{r['epoch']} | multi | {r['comparison']} | {r['mean_delta_m']:+.6f} | {r['improved']} / {r['worsened']} / {r['unchanged']}")
lines+=['','只读梯度机制检查',
'按官方开发集多人窗口顺序，在首尾之间等间距取32个位置。每窗用 RNG=1192035+window_index，无放回均匀取2名行人及20对不同世界，逐次只交换一名行人的两个候选，其他人及候选集固定。每轮1280个交换，三轮使用相同的窗口/行人/世界抽样；候选编号会随该轮H配对而不同。',
'真实代价 C=-0.1[logsumexp_k(-mean_i||X_ki-y_i||/0.1)-log20]。真实变化是 C(X_swap)-C(X)。坐标代理方向是 <grad_X C(X), X_swap-X>，等价于现行 hard-forward / soft-coordinate-backward 在最终分配矩阵P上的 <grad_P C, P_swap-P>。负值表示建议降低代价。',
'此检查定位在最终分配矩阵、Sinkhorn Jacobian之前，且只针对 pair softmin 项；没有测参数空间总梯度、初次配对到邻居反馈的反传链路，或 diffusion/map 等其它损失。无需训练模式更新。所有模型参数冻结；只对局部坐标求导。',
'用float64核算有限差与局部导数，并与现行FP32 pair_loss的坐标梯度对照；中心差分eps=1e-5验证局部导数。方向不一致指非近零真实变化与非近零代理方向异号。分母排除任意一侧近零；两侧近零数可能重叠。',
'epoch | tolerance (m) | valid swaps | near-zero true | near-zero proxy | informative | disagreements | rate']
for r in R['gradient']:lines.append(f"{r['epoch']} | {r['threshold_m']:g} | {r['valid_swaps']} | {r['near_zero_true']} | {r['near_zero_proxy']} | {r['informative']} | {r['disagreements']} | {r['disagreement_rate']:.2%}")
lines+=['不一致表明该局部坐标线性化未必指示有限离散交换的代价方向；不能据此称自动微分实现错误。有限步长的非线性、softmin权重迁移及范数曲率都可能贡献。也不能据此预测改网络后的效果。','',
'证据与归因边界（以正式选中的第26轮为主）',
f'候选覆盖：直接测得逐人下界{get(26,"all","oracle"):.6f} m，最终H为{get(26,"all","H"):.6f} m，差额{get(26,"all","H")-get(26,"all","oracle"):.6f} m。候选中有更接近真值的逐人解，现有联合对应没有同时实现它们；这支持存在可重排的终点协调空间，但不证明观测信息足以学到oracle，也不证明候选覆盖已足够或改善覆盖无效。',
f'初次配对：在多人窗，G→初次的均值变化为{get(26,"multi","initial_Hungarian")-get(26,"multi","G"):+.6f} m，初次→H为{get(26,"multi","H")-get(26,"multi","initial_Hungarian"):+.6f} m。这是同一候选和同一权重前向中的直接配对效果，不是初次模块独立训练质量的因果估计。',
'邻居反馈：三轮的初次配对及邻居反馈均在均值上改善JFDE，有直接数据支持；第26轮反馈在501个多人窗中改善259窗、恶化231窗、11窗无明显变化，所以收益并不逐窗一致。无法单凭该检查归因整个训练过程为何学成该状态，也不能等同于去掉反馈后重训的效果。',
'训练梯度：第26轮非近零交换的方向不一致率为39.38%，阈值改为1e-6后为39.43%，直接支持局部代理对有限交换存在方向偏差；不能将全部联合误差归因给梯度，也没有新模型优劣证据。第3/5轮只用于观察已有检查点差异；没有按这些终点结果改选官方权重。','',
'唯一优先修改建议（待用户审阅，尚未实施）',
'优先审查并替换 pair softmin 的坐标混合代理为候选误差代价层面的分配代理：在保持同一硬Hungarian前向、候选集合和官方selector的前提下，让反向直接面对每个候选到真值的终点代价，而不是先混合候选坐标再求距离。此次只提出这一项，后续需经授权再用相同交换诊断及训练验证其收益；不预先承诺指标改善，不同步更改候选生成、邻居结构或网络容量。','',
'交付索引',
'STOP_RECEIPT.json：身份、一次SIGTERM、退出、锁、恢复游标及收尾更新说明。',
'IDENTITY.json / CHECKPOINTS.csv / RETAINED_TOP5_AUDIT.json：输入与权重证据。',
'SUMMARY.csv / WINDOW_METRICS.csv / PAIRED_EFFECTS.csv / SHUFFLE_SUMMARY.csv：总体及逐窗终点结果。',
'AGENT_NEAREST.csv / NEAREST_SUMMARY.csv：每次行人出现的1/2/5近候选误差与两种加权。',
'PERMUTATION_CHECKS.csv / assignments/：全局编号不变性与实际对应表。',
'SWAPS.csv / GRADIENT_SUMMARY.csv：全部交换、局部梯度、近零及方向不一致率。',
'diagnose_gpu.py：实际完成的冻结模型GPU FP32前向、CPU梯度诊断脚本；diagnose.py与diagnose_parallel.py及日志保留早期CPU诊断中断痕迹，非训练中断。',
'RESULT.json：完整机器可读结果；正式队列不自动恢复。']
(O/'REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[lines.index('同候选场景等权结果（m）'):lines.index('世界编号与实际跨人对应关系')]))
