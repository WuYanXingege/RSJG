"""Render the Chinese report from the frozen audit CSV/JSON outputs only."""
import csv,json
from pathlib import Path
O=Path(__file__).resolve().parent
R=json.loads((O/'RESULT.json').read_text());Q=R['qa'];S=R['summary'];T=R['transitions']
def rows(n):
 with (O/n).open() as f:return list(csv.DictReader(f))
def percent(x):return 'NA' if x is None else f'{100*x:.2f}%'
def fraction(n,d):return f'{n}/{d} ({percent(n/d if d else None)})'
def table(headers, rr):
 return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rr]+['']
def select(ep,group,threshold,proxy):return next(r for r in S if r['epoch']==ep and r['subgroup']==group and r['truth_threshold_m']==threshold and r['proxy']==proxy)
L=['# ZARA2 相同交换上的配对代理冻结验证','',
'**判断：值得启动一次受控的 ZARA2 单折训练验证；本轮没有启动训练。** 支持这一判断的依据是：新代理消除了已抽样的有害采用建议，同时保留了大多数原本正确识别的好交换及其误差幅度收益；并非仅因为“零有害建议”。此结论支持投入一次验证，不表示已经改善硬 JFDE、JADE 或测试指标。','',
'正式第 26 轮，在相同的 1115 条真值非近零交换上，有害采用建议从 **391 降为 0**；旧代理正确采用的 335 条有益交换中，新代理保留 **281 条（83.88%）**，保留这些旧正确建议所覆盖误差幅度的 **93.34%**。但相对全部 388 条有益交换，新代理召回率为 **72.42%**（旧代理 86.34%），收益捕获率为 **74.87%**（旧代理 80.21%）：需要承担 13.92 个百分点的召回下降和 5.34 个百分点的收益捕获下降。','',
'在真实代价变化超过 1 cm 的交换中，有害建议从 **83 降为 0**，旧正确建议保留 **49/55（89.09%）**。9 人及以上的 9 个窗口中，有害建议从 **119 降为 0**，旧正确建议保留 **99/109（90.83%）**；其中超过 1 cm 的有益交换仅有 5 条，保留 4 条，样本很小。两类子组方向一致，但不能据此声称高人数场景已泛化改善。','',
'## 输入与运行边界','',
'- 固定来源提交：`4c9eede551bebdd7634ab81b7b4bd790165c6bb0`，正式源码：`468c5eddde48bdeec47c4bb3c39418352be16bb9`。',
'- 直接从 Git 对象读取该提交的 `SWAPS.csv` 原行、最终 `pi_H`、诊断脚本、身份和哈希；没有重新抽样。原始顺序保留在 `sample_row`，从 0 开始。',
'- 实际读取并核验第 3、5、26 轮 checkpoint 的哈希、模型状态、完成轮、update、恢复字段和 dev bank 身份。最终硬排列复用已存 `pi_H`，不重新运行模型。',
'- 相同的 32 个官方开发集多人窗口，40 次交换/窗/轮，1280 次/轮，共 3840 行。按 epoch、window index、agent ID、world u/v、candidate a/b 逐行对齐；允许不同 epoch 的候选编号不同。',
'- seed 2035 refined bank，K=20；候选和真值均为 FP32 原始米制绝对坐标，转 FP64 后计算。只读取这 32 窗的开发集终点，人数按银行和成员身份校验；不读测试集记录。',
'- CPU FP64，仅在坐标/分配矩阵上求诊断导数。模型前向、DDIM、optimizer updates 均为 0。原始权重、bank 和绝对资产路径不包含在本次发布材料中。',
'- 五折状态始终为 `PAUSED_RESUMABLE`；恢复快照保持 ZARA1/head update 7097、当前 epoch 10、completed_epoch 9、cursor 2360。原控制器/子进程不存在，锁已释放。本轮未发送信号、未 start/resume、未改正式源码。','',
'## 定义及先行复现','',
'对固定硬排列，D[i,a]=||g[i,a]−y[i]||₂，X[k,i]=g[i,pi[k,i]]，e[k]=mean_i D[i,pi[k,i]]。每个窗口先对有效行人平均，再计算 C(e)=−τ(logsumexp(−e/τ)−log K)，τ=0.1 m。归一化常数 log K 保留。','',
'交换行人 i 在世界 u、v 的候选 a、b：δ=(D[i,b]−D[i,a])/N，de[u]=δ、de[v]=−δ。真实变化为 C(e+de)−C(e)，重新计算交换后的 softmin；旧代理 s_coord=〈∇_X C, X_swap−X〉；新代理 s_cost=〈softmax(−e/τ), de〉。两个代理均在交换前的最终分配矩阵处、Sinkhorn Jacobian 之前衡量。新代理没有使用真实有限变化代替导数。','',
'96 个 epoch×窗口锚点的每世界误差、hard JFDE 和归一化 softmin，均通过“选中坐标”与“候选误差查表”两路校验。旧坐标函数直接从归档脚本提取；对应数值核验如下。','']
L+=table(['检查','最大绝对误差 / 结果'],[
 ['复现旧 true_delta',f"{Q['maximum_errors']['old_true_abs_error']:.3e} m"],
 ['复现旧 s_coord',f"{Q['maximum_errors']['old_coord_abs_error']:.3e} m"],
 ['坐标式与查表式真实有限变化',f"{Q['maximum_errors']['lookup_vs_coordinate_true_abs_error']:.3e} m"],
 ['新代理对 e+λde 的中心差分（全部3840行）',f"{Q['maximum_errors']['cost_fd_abs_error']:.3e} m"],
 ['新代理对硬分配 P 的独立自动微分（全部3840行）',f"{Q['maximum_errors']['cost_P_autograd_abs_error']:.3e} m"],
 ['true_delta ≤ s_cost 上界检查',f"超容差违反 {Q['concavity_violations']}/3840；最大正越界 {Q['concavity_max_positive_violation_m']:.3e} m"]])
L+=['数值一致性及凹性容差为 **1e−10 m**；中心差分容差为 **1e−8 m**，步长 λ=1e−5/max(1,max|de|)。这两项是实现核验容差，与统计弃权阈值 **1e−6 m** 不同。','',
'旧报告的“双侧非近零”分母只用于复现，复现值为：','']
L+=table(['epoch','异号数 / 双侧非近零分母','实际有害却建议采用','实际有益却建议拒绝'],[[r['epoch'],f"{r['disagreements']}/{r['denominator']}",r['harm_adopted'],r['benefit_rejected']] for r in Q['legacy_reproduction']])
L+=['第26轮的 391+40=431 已复现。主分析不能沿用 1093 作为两个代理各自的有效分母：仅按真值筛选时共有 **1115** 条，旧代理的 22 条弃权必须保留。','',
'## 同一真值分母下的成对比较','',
'主口径只按 |true_delta|>1e−6 m 选入，有益为负、有害为正。每个代理以同一建议阈值分为采用（s<−1e−6）、拒绝（s>+1e−6）、弃权（其余）。有害建议率分母是该代理在同一真值集合内的全部采用建议；有益召回率分母是同一集合的全部真实有益交换。分母为零记 NA。没有删掉代理自身的弃权行来计算“准确率”。','',
'收益捕获率=采用的有益交换 Σ(−true_delta) / 全部有益交换 Σ(−true_delta)。这些和只用于逐条交换诊断；交换可能相互冲突，**不能当成一次实际排列可以同时取得的改进**。','']
C=rows('CONFUSION_2x3.csv')
for group,threshold,title in [('all',1e-6,'全部32窗，主口径'),('all',.01,'全部32窗，|真实变化|>1 cm'),('N_ge_9',1e-6,'9人及以上，主口径'),('N_ge_9',.01,'9人及以上，|真实变化|>1 cm')]:
 L+=['### '+title,'']
 rr=[r for r in S if r['subgroup']==group and r['truth_threshold_m']==threshold]
 base=[r for r in rr if r['proxy']=='coord']
 L+=table(['epoch','原窗口 / 交换数','入选窗口 / 交换数','真实有益 / 有害'],[[r['epoch'],f"{r['windows_total']} / {r['swaps_total']}",f"{r['windows_eligible']} / {r['truth_eligible']}",f"{r['actual_benefit']} / {r['actual_harm']}"] for r in base])
 L+=['下表每一真值列均按“采用 / 拒绝 / 弃权”排列，构成每个代理的完整2×3计数表。大变化子集的代理弃权阈值仍为1e−6 m。','']
 counts=[]
 for r in rr:
  sub=[c for c in C if int(c['epoch'])==r['epoch'] and c['subgroup']==group and float(c['truth_threshold_m'])==threshold and c['proxy']==r['proxy']]
  b=next(c for c in sub if c['truth']=='benefit');h=next(c for c in sub if c['truth']=='harm')
  counts.append([r['epoch'],r['proxy'],' / '.join(b[k] for k in ['adopt','reject','abstain']),' / '.join(h[k] for k in ['adopt','reject','abstain']),r['adopted'],r['abstained']])
 L+=table(['epoch','代理','真实有益：采用/拒绝/弃权','真实有害：采用/拒绝/弃权','总采用','总弃权'],counts)
 L+=table(['epoch','代理','有害建议率：分子/分母','有益召回率：分子/分母','收益捕获：分子/分母（m）'],[[r['epoch'],r['proxy'],fraction(r['harm_adopted'],r['adopted']),fraction(r['benefit_adopted'],r['actual_benefit']),f"{r['captured_benefit_sum_m']:.8f}/{r['total_benefit_sum_m']:.8f} ({percent(r['benefit_gain_capture'])})"] for r in rr])
L+=['## 纠正与漏掉的好交换','',
'以下均为同一条交换的迁移。所有子组中新代理新增的有害采用均为0，也没有新增旧代理未采用的有益交换；完整3×3迁移表含全部弃权变化，见 [TRANSITION_MATRIX.csv](TRANSITION_MATRIX.csv)。','']
L+=table(['epoch','子组 / 真值阈值','纠正旧有害采用','保留旧正确采用','漏掉：转拒绝 / 转弃权','保留旧正确建议的收益幅度'],[[r['epoch'],r['subgroup']+f" / {r['truth_threshold_m']:g}",f"{r['harm_adoptions_removed']}/{r['old_harm_adopted']}",fraction(r['old_benefit_retained'],r['old_benefit_adopted']),f"{r['old_benefit_lost_to_reject']} / {r['old_benefit_lost_to_abstain']}",fraction(round(r['retained_old_benefit_gain_m'],8),round(r['old_benefit_gain_sum_m'],8))] for r in T])
L+=['第26轮主口径：391条原有害采用全部转为拒绝，没有通过弃权掩盖它们。旧代理正确采用的335条中，281条保留、40条转拒绝、14条转弃权；总弃权从22条增至25条。按全部真实有益交换计算，新代理漏掉107/388条；其中54条是本来被旧代理正确采用、现被新代理漏掉的。','',
'代表性遗漏按真实收益幅度从大到小选择，并另列1个转弃权案例；所有遗漏记录见 [MISSED_BENEFICIAL_SWAPS.csv](MISSED_BENEFICIAL_SWAPS.csv)。','']
miss=rows('MISSED_BENEFICIAL_SWAPS.csv');m26=[r for r in miss if r['epoch']=='26'];examples=m26[:6]+[next(r for r in m26 if r['cost_action']=='abstain')]
L+=table(['原CSV行号（0起）','窗口 / N / 行人ID','世界 / 候选','真实Δ（m）','旧坐标代理（m）','新代价代理（m）','新建议'],[[r['sample_row'],f"{r['index']} / {r['n_agents']} / {r['agent_id']}",f"{r['world_u']},{r['world_v']} / {r['candidate_u']},{r['candidate_v']}",f"{float(r['true_delta_m']):+.8f}",f"{float(r['s_coord_m']):+.8f}",f"{float(r['s_cost_m']):+.8f}",r['cost_action']] for r in examples])
L+=['例如窗口2745的一条交换真实改善8.80 cm，新代价代理却为+0.05813 m而拒绝；高人数窗口2992（9人）的一条真实改善1.69 cm也被拒绝。这些不是可以忽略的精度噪声。softmin对世界误差为凹函数，因此正的局部导数仍允许有限交换后代价下降；新代理更保守，并非有限交换的精确分类器。','',
'## 按窗口检查与高人数场景','',
'32窗在三轮中相同；9人及以上共有9窗，每轮360条交换。其人数为9、9、10、13、17、17、26、33、34。第26轮的32窗均至少纠正1条旧有害采用，11窗保留全部旧正确采用。下表为第26轮主口径；三轮及两个真值阈值的逐窗率、分子分母和弃权数均见 [WINDOW_SUMMARY.csv](WINDOW_SUMMARY.csv)、[WINDOW_TRANSITIONS.csv](WINDOW_TRANSITIONS.csv)。','']
WT=rows('WINDOW_TRANSITIONS.csv');ws=[r for r in WT if int(r['epoch'])==26 and float(r['truth_threshold_m'])==1e-6]
L+=table(['窗口','N','真值入选交换数','消除有害采用','旧正确采用保留','漏掉→拒绝/弃权'],[[r['window'],r['n_agents'],r['truth_eligible'],r['harm_adoptions_removed'],f"{r['old_benefit_retained']}/{r['old_benefit_adopted']}",f"{r['old_benefit_lost_to_reject']}/{r['old_benefit_lost_to_abstain']}"] for r in ws])
L+=['同窗交换、同窗跨epoch记录是配对观察，不是3840个独立场景。没有用独立同分布假设给出显著性或泛化置信区间。9人及以上且超过1 cm子集中，第26轮只有5条有益交换，因此4/5的保留率只能作样本内描述。','',
'只读引用已归档的H终点记录，未重新运行590窗推理或完整轨迹：','']
ctx=R['archived_endpoint_context'];ct=[]
for group in ['all_590','all_590_N_ge_9','sample_32','sample_32_N_ge_9']:
 q={r['epoch']:r for r in ctx if r['group']==group}
 ct.append([group,q[3]['windows'],f"{q[3]['archived_H_JFDE_m']:.6f}",f"{q[5]['archived_H_JFDE_m']:.6f}",f"{q[26]['archived_H_JFDE_m']:.6f}",f"{q[26]['archived_H_JFDE_m']-q[3]['archived_H_JFDE_m']:+.6f}"])
L+=table(['已有开发集子组','窗口数','epoch3 H-JFDE','epoch5 H-JFDE','epoch26 H-JFDE','3→26变化（m）'],ct)
L+=['全开发集127个9人及以上窗口由第3轮到第26轮退步约8.07 cm；抽样9窗退步约3.73 cm。这只能重述已发生的历史差异，不能证明坐标代理造成退步，也不能证明代价代理会修复它。官方选择仍是开发集H-JADE选出的第26轮，本轮未按终点诊断改选权重。','',
'## 为什么值得一次验证，以及仍不能得出的结论','',
'**值得做一次ZARA2单折受控训练验证。** 三轮均消除旧有害采用；更关键的是，保留原正确有益建议约83.9%–84.8%，保留它们的误差幅度收益约86.9%–93.3%，没有漏掉多数有价值的原正确建议。在超过1 cm及高人数子组中也保留了大多数原正确建议。这些实际计数足以支持小范围验证其训练作用。','',
'代价明确存在：第26轮全部真实有益交换的收益捕获仍有约25.1%未被新代理覆盖；第5轮新代理相对全部有益交换的收益捕获为70.91%，较旧代理低10.65个百分点。跨轮保持旧收益的程度并不恒定。有限步交换上安全的负方向，经过Sinkhorn Jacobian、网络共享参数、多损失和训练更新后，不必然产生更好的硬排列。','',
'“零有害采用”本身主要是数学预期：C(e+de)−C(e)≤∇C(e)·de=s_cost，因此若s_cost<−阈值，真实变化也为负（容差内）。它不是独立的泛化性能发现。决定是否值得验证的核心是保留了多少好交换，以及损失了多少有益幅度。','',
'若后续授权训练，仅将pair项的坐标代理与候选代价代理作为对照变量，保留候选、硬前向、官方选模指标和其它训练设置。本轮不启动该实验，也不恢复五折队列。没有观察到新模型的硬JFDE、JADE或测试改善，不能作此声明。','',
'## 交付与复算','',
'- [逐交换成对结果 PAIRED_SWAPS.csv](PAIRED_SWAPS.csv)：3840行，含真实变化、两个代理、建议、身份及数值核验。',
'- [汇总 SUMMARY.csv](SUMMARY.csv)、[完整2×3表](CONFUSION_2x3.csv)、[纠正和保留计数](TRANSITIONS.csv)、[全部迁移](TRANSITION_MATRIX.csv)。',
'- [实现核验 QA.json](QA.json)、[旧表复现](LEGACY_REPRODUCTION.csv)、[输入哈希与权重身份](INPUT_IDENTITY.json)、[运行状态](RUN_STATUS.json)、[宿主进程检查](HOST_PROCESS_CHECK.json)。',
'- [计算脚本](check_same_swaps.py)、[报告生成脚本](build_report.py)、[任务原文](TASK_SPEC_ZH.txt)。原始资产须从训练主机按身份和哈希取得；发布包不含权重、bank或资产绝对路径。','',
'在具有原资产布局的机器上执行（以下是占位路径，不是资产位置）：','',
'```bash',
'python check_same_swaps.py --workspace "$RSJG_WORKSPACE" --archive-repo "$RSJG_ARCHIVE_REPO" --output "$AUDIT_OUTPUT"',
'# 将 build_report.py 放入同一输出目录后运行：',
'python "$AUDIT_OUTPUT/build_report.py"',
'```','',
'**最终状态：诊断 optimizer updates=0；五折控制器 PAUSED_RESUMABLE；正式源码修改=否；队列恢复=否。**']
(O/'REPORT_ZH.md').write_text('\n'.join(L)+'\n')
print('Wrote REPORT_ZH.md; decision: worthwhile for one controlled fold, not launched.')
