"""Render the Chinese audit report from published anonymous statistics only."""
import argparse,csv,json,hashlib
from pathlib import Path

def main(d):
    read=lambda name:json.loads((d/name).read_text())
    rows=list(csv.DictReader((d/'GROUPS.csv').open()));dist=list(csv.DictReader((d/'DISTRIBUTIONS.csv').open()));gr=list(csv.DictReader((d/'GRADIENT_OBSERVABILITY.csv').open()))
    q=read('QA.json');f=read('FROZEN_CONFIG.json');code=f['code_commit'];base=f['base_source']
    url='https://github.com/WuYanXingege/RSJG/blob/'
    def link(path,line,commit=code):return f'[{path}:{line}]({url}{commit}/{path}#L{line})'
    lines=[f'''# BCEP–EBJD 共享世界联合终点目标：工程验收与冻结权重审计

工程验收通过；冻结权重上存在新增训练信号，但**没有新训练后的开发集效果证据**。本轮 `RUN_ONE_ZARA2=false`，真实优化器更新 **0**，未启动训练、测试集评测或等待队列。旧五折与 candidate_cost 均保持可恢复暂停。

判断：这项单一目标具备进入一次 ZARA2 受控训练验证的工程条件，尚不支持扩折或宣称改善。若后续明确授权 `RUN_ONE_ZARA2=true`，只检验本目标相对旧 coordinate 基线的整套效果；本次交付结束即退出。特别需要检验“初配直接监督是否足够强”：4 个固定样例的初配直接参数梯度远小于最终项，非零 a 不能证明初配会有效改善。

## 实验身份与暂停资产

- 正式来源：`{base}`；本轮代码提交：`{code}`。报告提交另由 GitHub 固定链接标识。
- 独立分支：`experiment/zara2-shared-world-objective-20261119`；独立 worktree 和实验目录均由本轮创建，未改动原源码目录。
- 配置 SHA256：`{f['config_sha256']}`；实现文件集合 SHA256：`{f['implementation_sha256']}`。完整配置、原 recipe、冻结判据见 [FROZEN_CONFIG.json](FROZEN_CONFIG.json)。
- 原五折：`PAUSED_RESUMABLE`，ZARA1/head 保存 update **7097**，已完成 epoch 9，当前 epoch 10，cursor 2360。`formal_run status` 的周期进度仍为 7090，实际恢复点以 `last.pt` 为准。
- candidate_cost：原生状态 `SESSION_BUDGET_PAUSED`，研究暂停语义为可恢复暂停；保存 update **1903**，已完成 epoch 11，当前 epoch 12，cursor 880。
- 两个控制锁检查均空闲，本轮未发送信号。受保护 last、STATUS、Top5 哈希及两条运行根目录全部文件的尺寸/mtime 清单前后相同。checkpoint、Top5、bank、历史报告保留。[暂停收据](PAUSE_RECEIPT.json) · [保全核验](PRESERVATION.json)
- 原自有 BCEP 四层 Flow、M256→KMeans20、候选修正头和地图输入冻结。train/dev bank 均为 seed 2035，2681/590 窗；开发集 4262 次有效行人出现，其中 89 单人、501 多人、127 个 ≥9 人窗。仅访问开发集和固定的 2 个训练窗，不读取 heldout 轨迹数据。

## 实际前向与反传路径

| 位置 | 核验结果 |
|---|---|
| {link('ebjd/unified.py',89)}、{link('ebjd/unified.py',111)} | 一个共享 `world_queries`，每个世界 k 在所有人上共用查询；候选列轴与世界行轴不同。 |
| {link('ebjd/unified.py',43)}、{link('ebjd/unified.py',69)} | 每人 Hungarian 在候选列上作双射，`gather_candidates` 保留世界行号；单人和 padding 用恒等排列。 |
| {link('bcep_v2/joint.py',98)}、{link('bcep_v2/joint.py',78)} | 初次 assignment→邻居 compatibility 消息→残差分数→最终 assignment。同一前向仍只有两次 assignment；消息使用初次直通矩阵，梯度通道保留。 |
| {link('bcep_v2/joint.py',123,base)} | 旧 pair 只监督最终 H 的每世界有效人平均距离，再对世界 softmin。初次分数经最终残差与消息获得梯度，初次硬端点没有自己的直接损失。 |
| {link('bcep_v2/joint.py',105)}、{link('bcep_v2/joint.py',133)} | 训练选项复用第一次已算出的 pi/psoft 产生初次坐标直通张量，再以同世界 I/H 算 a/C；无额外编码、Sinkhorn、Hungarian、可学习参数或按真值重排。 |
| {link('bcep_v2/objective.py',17)}、{link('bcep_v2/objective.py',67)} | 新 pair 替换进入实际 `scene_part`、返回总损失及日志；agent_part 不变，也未把两个 pair 另行相加。 |
| {link('bcep_v2/stages.py',160)}、{link('bcep_v2/state.py',29)} | agent 按物理组人数/微批人数，scene 按组窗数/微批窗数，再除原累积数 4。保留尾批原除数语义。新日志另存场景分子与 `valid_scene_count`，恢复后可正确求整轮场景均值。 |
| {link('ebjd/representation.py',207)}、{link('bcep_joint/inference.py',36)} | eta=0 将最后一帧固定在候选终点；20 步 DDIM、keyed noise、候选/噪声共同 gather 不变。桥损失以真值端点为条件，但推理配对只读观测、地图和固定候选。 |

原共享编码、配对头、六层条件桥结构完全保留；原生效损失为 diffusion（coarse 系数 0.25）、geometry 0.05、map 0.10、clean path 0.05、pair 1。默认 `pair_objective=final_coordinate`；只有 `shared_world_bounded` 开启新目标。无新字段的旧权重 strict-load 兼容，旧推理不计算 a、不需要真值。最小运行代码差异见 [MINIMAL_CODE.patch](MINIMAL_CODE.patch)。

## 唯一目标及解释边界

每窗每世界先平均有效人的欧氏终点距离，得到 I/H（米）；不是平方距离、逐人选优或全批人数加权。

```text
a = clip((stopgrad(I) - stopgrad(H)) / (2 * 0.10), 0, 0.5)
C = H + a * (I - H) - 0.10 * a²
L_pair = mean_scene[-0.10 * (logsumexp_world(-C / 0.10) - log(20))]
```

C 是区间内 `(1-u)H + uI - τu²` 的极大值。停止梯度的是解析系数的输入副本，I/H 仍在图中；范数前清理 padding，每窗至少一人。两个阶段的第 k 行始终是同一个共享世界。对标量 I/H 的局部系数为 `q*a` 与 `q*(1-a)`，不是模型参数梯度的比例，也不是硬 JFDE 降低保证。温度 0.10 m 和上限 0.5 是本项目公开的设计选择。

## 工程正确性

15 个独立测试用例均已通过（10 个原测试、5 个新增测试；首次 13 通过/2 失败，问题定位后针对复查通过）。[ACCEPTANCE.json](ACCEPTANCE.json) 与 [测试源码]({url}{code}/tests/test_shared_world_pair.py) 记录范围：

1. 同 seed 3201 的初始 tensor hash 与正式来源一致；无新增参数，旧权重 strict-load。反馈零初始化 I=H 时，新旧实际总损失前向差为 0，所有共享/反馈参数梯度在 atol 1e-6、rtol 1e-5 内一致。完整 20 步 G/H 推理输出完全相等，真值填 NaN 不影响推理。
2. 反馈改善/恶化/相同、不同世界分别获胜、解析最大值网格和有限差分、共同世界标签置换、单人、NaN padding、同分排列均覆盖；a 严格限于 [0,0.5]。有限差分容差 3e-6。
3. 非零反馈样例分拆初次项和最终项：初次直接项可达原 score/shared 分支，不能直达 compatibility 参数；最终项仍可达 score、compatibility 和初次消息矩阵。冻结候选无梯度；Provider 不在该计算图中。
4. 正常 `joint_callbacks.backward` 入口通过。采用同一批实际物理分组前向构造直接标量基准，核对 scene/agent 分母、累积除数和 score 参数梯度；新统计分子/分母一致。
5. 原 `run_stage` 用真实新损失做 24 次合成更新（连续 12；暂停 5+恢复 7），完整状态与连续运行一致。六轮 Top5 为 `[2,3,5,6,4]`，同分按早轮，模型/优化器/调度器/RNG/游标/早停字段齐备、哈希正确。原测试另有 8 次合成更新，合计 **32 次合成、0 次真实更新**。

初次两项测试失败反映了验收基准需区分的问题：随机初始查询很小时，候选重排/物理分组引起的 FP32 归约微差可能改变近同分 Hungarian 结果。候选与 keyed noise 身份测试改用非退化查询；分组测试按同一物理前向验证数学分母与梯度，不把任意重新分组的逐位相等当作承诺。原有正式分组规则未变。实际 epoch 3/5 的全部开发窗硬排列与旧归档逐项相等。

## 冻结权重表现：不属于新训练效果

实际旧 coordinate epoch 3、5 的权重哈希见 [CHECKPOINTS.json](CHECKPOINTS.json)。以 seed 2035 原 bank、同一有效行人顺序和原窗口顺序计算。初/终 pi 对照 `4c9eede` 的归档，原候选身份逐窗核验。世界编号 0–19 不排序；`argmin` 并列取最先世界。所有表格均场景等权；每窗 K=20，因此世界激活比例等价于先逐窗再平均。

| 轮次 | 分组 | 窗数 | I-JFDE | H-JFDE（新旧相同） | I−H 反馈收益 | a>0 世界比例 | 平均 Σ(q·a) | I/H 获胜世界切换 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|''']
    for r in rows:
        lines.append(f"| {r['epoch']} | {r['group']} | {r['windows']} | {float(r['I_JFDE']):.6f} | {float(r['H_JFDE']):.6f} | {float(r['feedback_gain']):+.6f} | {float(r['a_positive_fraction']):.2%} | {float(r['q_a']):.6f} | {float(r['winner_switch']):.2%} |")
    lines.append('''
单人窗的 I/H/JFDE 完全相同，a 精确为 0。逐人 minFDE 集合完全不变；开发集窗口等权 oracle 为 0.339426 m。新旧推理相同是兼容性证据，不能称为精度改善。这里的两个最小误差之差仅用于反馈收益，a 是用逐世界 I[k]−H[k] 计算的；两者不可互换。

| 轮次/分组 | a=0 / 全部世界 | a=0.5 / 全部世界 | Σ(q·a) P25 / P50 / P75 | q 熵：旧→新 | exp(熵) 的窗均值：旧→新 | C 获胜世界相对 H 切换 |
|---|---:|---:|---|---|---|---:|''')
    for r in rows:
        if r['group'] not in ['multi','N_ge_9']:continue
        dd=next(x for x in dist if x['epoch']==r['epoch'] and x['group']==r['group'])
        lines.append(f"| {r['epoch']} / {r['group']} | {dd['a_zero_count']} / {dd['world_count']} | {dd['a_cap_count']} / {dd['world_count']} | "+' / '.join(f"{float(dd['q_a_p'+str(p)]):.4f}" for p in [25,50,75])+f" | {float(r['old_q_entropy']):.4f}→{float(r['new_q_entropy']):.4f} | {float(r['old_effective_worlds']):.3f}→{float(r['new_effective_worlds']):.3f} | {float(r['objective_switch_from_H']):.2%} |")
    lines.append('''
全部 a 的 P50=0、P75=P90=P95=0.5（两轮各分组，单人除外），表明大量世界处于系数边界。完整分位数见 [DISTRIBUTIONS.csv](DISTRIBUTIONS.csv)，23,600 条逐世界 I/H/a/C/q 见 [WORLD_WEIGHTS.csv](WORLD_WEIGHTS.csv)，1,180 条逐窗统计见 [WINDOWS.csv](WINDOWS.csv)。FP32 求和使一个窗口的 Σ(q·a)=0.5000000596，比理论上限多 5.96e-8；a 本身没有超界，保留原值而未裁剪报告。

固定训练窗按索引顺序选首个 2≤N≤4 的窗（index 3，N=2）和首个 N≥9 的窗（index 955，N=9）；两轮各检查一次，不按结果筛选、不增广、不更新参数。使用完整模型参数向量，未用到的桥参数梯度记零。下表只表示损失梯度可观测性，不是对交换或新模型的效果评测。

| epoch/index/N | Σ(q·a) | 初次直接项梯度范数 | 最终项梯度范数 | 初次/最终范数比 | 新旧 pair 梯度余弦 |
|---|---:|---:|---:|---:|---:|''')
    for r in gr:
        ratio=float(r['initial_component_norm'])/float(r['final_component_norm'])
        lines.append(f"| {r['epoch']}/{r['index']}/{r['n_agents']} | {float(r['q_a']):.4f} | {float(r['initial_component_norm']):.3e} | {float(r['final_component_norm']):.3e} | {ratio:.3e} | {float(r['cosine_old_new']):.4f} |")
    lines.append(f'''
新梯度等于两项参数梯度之和，最大残差 {max(float(r['decomposition_max_abs']) for r in gr):.3e}。初次直接项范数约为最终项的 0.012%–0.757%；即使标量 Σ(q·a) 较大，实际直接初配梯度仍可能很弱。不能仅凭非零权重、熵增加或梯度方向变化宣告成功。[逐窗梯度 CSV](GRADIENT_OBSERVABILITY.csv)

## 证据支持什么、尚不能归因什么

候选覆盖：同一冻结候选集合及 oracle 已核验，配对目标不会增加覆盖。初次配对：确认此前缺少直接输出监督，本目标在多人数窗产生非零监督项；尚无训练证据证明它能改善 I。邻居反馈：旧 epoch 3 的密集窗反馈平均改善 0.061791 m，epoch 5 平均恶化 0.010868 m，说明反馈作用随权重变化；同权重新旧完全相同，不能把这些旧收益算作新目标效果。训练梯度：解析系数、路径和有限差分有直接工程支持；真实样例的初次参数梯度较弱，未来 I/H/JADE 是否改善仍无法归因。

限制有三点。I/H 同样差时 a=0，初次没有新增直接信号；C≥H，可能降低那些“初次差、反馈后很好”的世界的 softmin 权重，从而压低有用反馈重排；只读的两个旧权重和四个梯度样例不代表新目标训练过程。多人窗 C 的获胜世界相对 H 已在 33.33%/37.72% 的窗口改变，这是权重重分配的规模，既可能有益也可能有害。

## 尚无新训练证据；未来判据已冻结

| 权重 | dev H-JADE | dev H-JFDE | dev ≥9 人 H-JFDE | 用途 |
|---|---:|---:|---:|---|
| 旧 coordinate epoch 26 | 0.4084707879413993 | 0.8016974424154072 | 1.3188941478729248 | 正式 dev H-JADE rank1 |
| 旧 coordinate epoch 5 | 0.4097670185111337 | 0.7841791655426308 | 1.296843430188697 | 竞争性历史折中 |
| 旧 coordinate epoch 3 | 0.4218764735164784 | 0.7792569560204017 | 1.2382272847994107 | 上下文，不替换选模规则 |
| 新目标训练后 rank1 | 未训练 | 未训练 | 未训练 | 不填入冻结旧权重指标 |

本轮端点独立归约与历史完整轨迹指标有约 1e-9–1e-8 m 浮点差，冻结判据仍用上述归档未截断原值。只有新模型按 dev H-JADE 选出的 rank1 的两项 H-JFDE 均低于旧 epoch 5、且 H-JADE 不高于旧 epoch 5，才支持进一步独立验证；微小差值也不能称稳定提升。若初配改善而最终未改善，或 H-JADE 退步，保留旧坐标基线结案。不得据一次整体对照宣称动态系数优于未实测的固定混合。

未来授权时仍应 seed 3201 从原 Stage 2 初态开始，原增广/数据顺序/FP32/AdamW/5 轮 warmup/100 轮调度、每轮 168 更新、至少 30 轮、patience 15、上限 100 轮；dev H-JADE Top5、20 步 DDIM 及完整恢复保留。此次脚本没有训练启动入口，也没有安排后台任务。

## 实测资源与执行记录

最终成功诊断进程耗时 {q['total_seconds']:.3f} s，其中开发集重建/前向 {q['dev_forward_seconds']:.3f} s、固定窗梯度 {q['gradient_seconds']:.3f} s；{q['gpu']}，torch {q['torch']}，FP32、math SDP、TF32 关闭。峰值 allocated {q['peak_allocated_bytes']/2**30:.3f} GiB、reserved {q['peak_reserved_bytes']/2**30:.3f} GiB。

这不是含失败尝试的总墙钟耗时。最初沙箱无法看到 CUDA，在真实数据模型前向前退出；第一段 GPU 作业已完成 epoch 3 全部 590 窗相等核验，之后一个训练窗因 cuDNN 的 eval-mode RNN backward 限制退出。修正仅涉及诊断模式，未改目标；第二段对 epoch 3 复用已验证的归档排列，只做 epoch 5 的 590 次模型前向，并完成 4 次梯度观测。全轮真实开发窗模型前向合计 **1180**、成功梯度观测 **4**、额外失败训练窗前向 **1**、真实 optimizer.step **0**。最终进程的独立计数与耗时见 [QA.json](QA.json)，跨尝试总计见 [EXECUTION_LEDGER.json](EXECUTION_LEDGER.json)。

## 文献口径与项目方法归属

JMM 的 ZARA2、K=20 Joint AgentFormer：marginal FDE 0.242、JADE/JFDE 0.252/0.509 m，来自 ICCV 2023 主会表 1；联合指标先按同一个世界平均有效人误差，再选世界。第 4 节的联合训练是直接指标/监督参照。本项目历史五种子测试 minFDE/JFDE 为 0.23932/0.65075 m，测试 JADE 未完成；测试与本轮 dev 不同，这些测试数值均不作通过线。[JMM 正文](https://openaccess.thecvf.com/content/ICCV2023/papers/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.pdf)

| 正式主会论文 | 本次来源核验及可借鉴内容 | 本项目假设边界 |
|---|---|---|
| SHARP，CVPR 2026，pp.32103–32112 | CVF 主会页面；补充 B.7 对 AV2 多智能体输出按全场位移挑同一个获胜世界。 | 借鉴共享世界监督；未验证本项目两次 Hungarian 的有界目标。 |
| Den-TP，CVPR 2026，pp.10632–10641 | CVF 主会页面，Argoverse 1/2 按人数考察密度。 | ≥9 人阈值是本项目既有分组，不替代 JMM 全窗指标。 |
| ParkDiffusion++，ICRA 2026 | Crossref 的 IEEE 出版元数据返回 HTTP 200，正式 ICRA 2026 proceedings-article，pp.9345–9352（Vienna，6 月 1–5 日）；作者 camera-ready 正文核实 DLP/inD 上每个自车意图对应一组联合响应。 | 借鉴同假设下的跨人一致性，本项目仍用共享查询和行人对称性。 |
| DSBTR，NeurIPS 2026 | 官方单篇日程已直接核验 HTTP 200：主会 Poster，2026-12-09 17:00–20:00 AEDT，Hall 1–4 #S2309；页面元数据于 2026-09-25 发布、2026-11-17 更新。摘要说明训练扩散终态与无噪声粗预测起点的边界对齐。 | 状态为主会已录用并列入日程（会议尚未举行）；其扩散边界不同于 EBJD 轨迹时间轴固定最后一帧空间终点，不能证明本项目改善多少 JFDE。 |
| NTKMTL，NeurIPS 2025 | 正式 proceedings 标注 Main Conference；分析共享表示及任务收敛速度。 | 仅支持记录训练动力学的动机，不为本项目 a 公式或效果背书。 |

原文入口：[SHARP 主会](https://openaccess.thecvf.com/content/CVPR2026/html/Prutsch_SHARP_Short-Window_Streaming_for_Accurate_and_Robust_Prediction_in_Motion_CVPR_2026_paper.html)、[SHARP 补充 B.7](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Prutsch_SHARP_Short-Window_Streaming_CVPR_2026_supplemental.pdf)、[Den-TP](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Den-TP_A_Density-Balanced_Data_Curation_and_Evaluation_Framework_for_Trajectory_CVPR_2026_paper.html)、[ParkDiffusion++ DOI](https://doi.org/10.1109/ICRA57385.2026.11696838)、[IEEE 登记的 Crossref 元数据](https://api.crossref.org/works/10.1109/ICRA57385.2026.11696838)、[作者机构出版页](https://rl.informatik.uni-freiburg.de/publications)、[ICRA camera-ready 正文](https://arxiv.org/abs/2602.20923)、[DSBTR 附件给出的日程](https://neurips.cc/virtual/2026/poster/154766)、[NeurIPS 官方列表](https://neurips.cc/Downloads/2026)、[作者出版页](https://xbchen82.github.io/publications/)、[NTKMTL 主会论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4522de4178bddb36b49aa26efad537cf-Abstract-Conference.html)。核验日期 2026-11-19。所有数值改进论断以本项目实测为准。

逐世界有界目标及解析系数是本项目针对两次 Hungarian 的待验证方案。文献借鉴的是联合假设与评估原则，不能被写成本方案已有效的外部证据。

内部依据单列：[五种子测试背景]({url}f44652730d0d7660c0eefed2cefa6bf939f1037c/docs/research/bcep_v2/formal_fivefold_top5_20261118/20261119_zara2_test_snapshot/REPORT_ZH.md)、[冻结 train/dev 退步诊断]({url}7ea3a2d1a5a5d654d9e23ba4ed38f94610fc6736/docs/research/bcep_v2/costproxy_stage2_20261119/pause_train_dev_frozen_20261119/REPORT_ZH.md)。后者的 candidate_cost 密集 dev 初配退步 0.14843/0.14473 m 只定位换代理的退步，不解释全部跨论文差距，也不证明本目标有效。

## 复现与交付

从代码提交 `{code}` 运行，先在训练主机配置 Python 环境并设置本地 `WORKSPACE` 与一个新的空 `OUT`。路径只作运行参数，不发布真实绝对路径。

```bash
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest tests/test_shared_world_pair.py tests/test_bcep_integrated_v2.py -q
python scripts/shared_world/audit.py --workspace "$WORKSPACE" --out "$OUT"
```

上述 pytest 会做 32 次合成更新，audit 无优化器。正常全新复现会对两轮各前向一次；本轮中断续查使用 `--reuse-verified-epoch3`，只改变已有证据的复用方式。报告可由交付分支的 `python scripts/shared_world/report.py --data docs/research/bcep_v2/shared_world_objective_20261119` 重建。公开包只含代码、哈希和匿名误差/统计；权重、bank、原始轨迹、进程 cmdline 和内部绝对路径保留在主机。
''')
    (d/'REPORT_ZH.md').write_text('\n'.join(lines))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);main(p.parse_args().data)
