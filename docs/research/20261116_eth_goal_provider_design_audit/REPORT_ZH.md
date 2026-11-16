# EBJD 终点提供器与网络路线复核（2026-11-16）

## 结论先行

**现有证据不支持把整个 EBJD 联合建模方向判为错误；它支持把已完成五折验证的六层统一方案恢复为论文主方案。** ETH 从零训练的八层终点密度版目前明显落后，差距首先发生在进入联合桥之前的 **20 个终点候选覆盖**。当前桥固定终点，不能用更深的轨迹网络补回候选未覆盖的真实终点。将八层从零版直接替代已验证的六层方案，没有指标依据。

旧统一方案并非在推理时运行完整 GDTS：它使用从同折 matched GDTS 权重提取的独立 Goal U-Net、原生预处理和 TTST 产生终点，再由 EBJD 的学习联合配对与条件桥生成轨迹。**运行结构可独立，权重来源仍依赖 matched GDTS 训练。** 已核对的 ZARA1/ZARA2 中 Goal-pretrain 权重不能直接替代 matched 权重；这一逐参数结论尚未单独外推到 ETH。

当前审查阶段只复核冻结资产和代码，不重新训练、调参或选模。当前 ETH 从零训练后台任务未被本次审查修改；下述第 14、16、63、76 轮数字均保留其原身份，不拼接为同一正式五种子结果。

## 一、可比身份与核心数字

旧统一 ETH 与当前从零 ETH 使用同一 validation manifest（SHA256 `03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b`），253 个同步场景、364 次有效行人出现，米制坐标，K=20，20-step DDIM，训练 seed 3101。旧方案选中 epoch 19 的 `best_jade`（checkpoint SHA256 `48dc2144fb20cb3e678eed7b7df793598fe8f2ac0eda7a026b4bcf1696626b97`）；当前从零版较佳的 epoch 14 权重 SHA256 为 `935a9073563dfd002015d2df8d815b3bf8b949a9d9634251778d00363a3ae81b`。旧 matched GDTS 权重 SHA256 为 `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950`。

| ETH 方法与口径 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| matched GDTS，2035–2039 均值 | 0.4158 | 0.5771 | 0.4927 | 0.7492 |
| 旧六层统一 EBJD H，2035–2039 均值 | **0.4032** | 0.5811 | **0.4706** | **0.7341** |
| 当前八层从零版 epoch 14，**仅 seed 2035 完整轨迹** | 0.4539 | 0.6749 | 0.5345 | 0.8587 |
| 当前八层从零版 epoch 76，**仅当轮 seed 2035 验证** | 0.6508 | 1.2602 | 0.7602 | 1.4759 |

上表后两行只有单个推理 seed，不能与前两行的五 seed 均值进行方差或显著性判断；同 seed 2035 的旧 H 为 0.4071/0.5819/0.4698/0.7303 m，仍优于当前从零版 epoch 14 的四项。当前运行的 selector 仍指向 epoch 14，epoch 76 不是最终选择权重，也不是已完成的 100 轮结果。旧 ETH 的四指标合同只因 minFDE 比 matched GDTS 高 **0.0041 m** 而未全过；因此也不能写成“ETH 四项全面超越”。

在同一 validation 成员上做的**五 seed 终点 bank**对照更直接：旧原生 provider 的 minFDE 为 **0.5811 m**、Recall@0.5 m 为 **64.1%**；当前从零版 epoch 14 为 **0.7157 m**、**51.0%**，差 **+0.1345 m** 和 **−13.1 百分点**。N=1 的 minFDE 差为 +0.1727 m，N≥2 为 +0.0959 m；两组均有缺口，并非只由社会交互或最坏少数样本造成。旧 provider bank 与 GDTS 完整轨迹最终终点分别为 0.5811 和 0.5771 m；两者来自不同导出入口，只能说明汇总指标很接近，不能声称同一 world 的轨迹去噪严格只改善 0.0041 m。

已完成的**旧统一五折** H 等权宏平均为 0.2346/0.3569/0.3625/0.6778 m，matched GDTS 为 0.2369/0.3667/0.3940/0.7186 m。UNIV、ZARA1、ZARA2 四项合同通过；ETH 仅 minFDE 退化，HOTEL 仅 minADE 退化。它证明旧统一方法的联合收益，不证明八层从零版已复现五折收益。原始逐 seed、来源和选择身份见 [五折正式报告](../../results/ebjd_eth_ucy_unified_fivefold_20261101/REPORT_ZH.md)、[原始指标](../../results/ebjd_eth_ucy_unified_fivefold_20261101/FIVEFOLD_METRICS.csv) 与 [来源清单](../../results/ebjd_eth_ucy_unified_fivefold_20261101/SOURCE_MANIFEST.json)。

## 二、当前网络与旧方案到底改了什么

| 环节 | 已验证的旧统一方案 | 当前 ETH 从零八层方案 |
|---|---|---|
| 主干 | 192 维、4 头、FFN 768；coarse 3 + refine 3，共 6 个 FJAB；FULL/Raw | 同宽；coarse 4 + refine 4，共 8 个 endpoint-gated block；FULL/Raw |
| 终点来源 | 同折 matched GDTS 更新过的最小 Goal U-Net；语义图与历史热图；原生 TTST 从大量热图样本聚成 K=20 | EBJD 共享 MSCE 编码器上的 64×64 空间密度＋crop 外尾部；从该密度独立抽取 K=20 |
| 联合部分 | 学习联合配对 H，在原候选集合内重配；固定终点条件桥 | 仍学习联合配对与条件桥，另加终点 NLL 和有限 K 覆盖风险 |
| 训练与推理 | 桥训练以真实终点为条件；推理以冻结候选为条件，20-step DDIM | 同样以真实终点训练桥；`proposal_eta=0`，推理终点严格固定为采样候选 |
| 资产依赖 | 需要 matched 训练后的 Goal U-Net 权重；运行时不用 GDTS 轨迹 denoiser | 不加载外部 Goal 权重；目标是全模型从随机初始化 |

实际链路可概括为：历史轨迹与语义图 → 地图/历史编码 → 终点候选 → 学习配对 → 终点条件联合桥/CTF → 20-step 轨迹。当前 `SpatialGoalDensity.sample()` 在 `no_grad` 下执行；配对只重排同一组候选；`proposal_eta=0` 让最终帧严格等于所选候选。因此 **H 和 G 的 minFDE 在同一 bank 上相同**，联合损失不能直接把缺失的终点补回来。旧方案的 H 相对 G 在 ETH 五 seed 中将 JADE 从 0.4879 降至 0.4706 m、JFDE 从 0.7748 降至 0.7341 m，说明配对/桥不是完全无效；其贡献主要在联合 world 的一致性，而非边缘终点覆盖。

桥内部保留恒速基线和 EBD 表示：前 11 个白化桥系数控制中间轨迹，第 12 槽表示终点；coarse/refine 块在时间与同场景行人间交换信息，CTF 从预测轨迹的物理位置读取地图与交互反馈，最后 Raw 解码为 12 帧轨迹。八层门控使条件表达更强，但训练时第 12 槽在 `eta=0` 条件下被固定为提供器终点；“网络更深”并未改变这一终点边界。这是设计约束而非代码错误，也解释了为什么桥可以改善 JADE/JFDE，却不能在保持同一候选集合时提高 minFDE。

相关实现：[独立 goal provider](../../../ebjd/goal_provider.py)、[当前空间密度](../../../ebjd/spatial_density_goal.py)、[当前训练与推理入口](../../../scripts/train_learned_goal_bridge.py)、[旧统一模块](../../../ebjd/unified.py)。旧 provider 的逐参数与最小运行依赖见 [依赖核查](../../audits/goal_provider_dependency_20261029/REPORT_ZH.md)。

## 三、目前能确认和不能确认的原因

1. **已确认：候选阶段就有覆盖缺口。** 上述同 manifest、同 K、同身份的五 seed bank 在桥之前已有 +0.1345 m minFDE 缺口；单人和多人都存在。把问题只归于联合配对、CTF 或第 8 层没有依据。
2. **已确认：当前从零终点分支后期漂移。** 冻结回放的 epoch 14→63 五 seed minFDE 从 0.7157 变为 1.2336 m，Recall@0.5 从 51.0% 降到 44.4%，验证 goal NLL 从 3.4167 升到 4.1525；同期训练 goal NLL 下降。交叉替换后期编码器或终点头都会损害覆盖，但这种未训练的混搭不是严格因果分解。逐 seed、逐行人和 checkpoint 身份见本地资产分析目录 `RSJG_EBJD_assets/analyses/eth_goal_density_drift_20261116_095451/`。
3. **尚未分离：** 旧新同时改变了 goal 权重来源、地图/热图处理、TTST 与独立采样、六层与八层、终点目标和训练长度。现有数据只能定位性能从终点候选进入，不能声称“250 轮 GDTS 训练单独带来全部收益”“八层必然更差”或“仅把采样换成 TTST 就能追平”。
4. **训练/推理条件差异仍在。** 桥训练看真实终点，推理看候选终点；24 个场景的 GT 终点 oracle 显示前 11 帧 JADE 可从 0.5021 降至 0.2940 m，但它同时改变了终点质量与条件距离，不能独立量化该失配。原始证据见本地 `RSJG_EBJD_assets/analyses/eth_goal_bridge_frozen_20261115_171545/REPORT_ZH.md`。

本轮先前的两个有界备选也已给出边界：双编码器候选在同轮第 16 轮优于原模型，但第 19 轮五 seed minFDE 为 **0.7666 m**，比原 epoch 14 的 **0.7157 m** 更差，按预定闸门 **DO_NOT_ADOPT**，未做更长训练。另一个从 epoch 14 完整状态分叉、固定从零学得的终点塔继续训练桥的第 16 轮，seed2035 minFDE 保持 **0.6749 m**，JADE/JFDE 为 **0.5092/0.8355 m**；候选有序终点与 epoch 14 逐值相等。这只通过两轮、一个推理 seed 的早期闸门，**不是五 seed 正式结果**，而且仍落后旧 provider 的终点水平。完整收据留在 `RSJG_EBJD_assets/experiments/eth_dual_context_goal_v1_20261116_095641/` 和 `RSJG_EBJD_assets/experiments/eth_frozen_goal_bridge_fallback_v1_20261116_103952/`；均未写入当前正式模型。

## 四、应如何选择方案

**论文主线现在选旧六层统一方案＋同折 matched 权重的独立 Goal U-Net。** 保留可学习联合配对与条件桥为 EBJD 核心，明确披露可靠终点先验的训练来源、冻结方式和额外计算成本；不要把它包装成“完全从零、完全不依赖 GDTS”。这一选择依据是已完成的五折正式评测与更好的 ETH 终点覆盖，而非因为旧方案名称或训练更久。旧五折合同仍有 ETH minFDE 和 HOTEL minADE 两个单折限制，论文陈述必须保留。

若研究目标仍是摆脱 GDTS 训练权重，可把**同协议 goal 热图/候选分布蒸馏到独立终点提供器**作为下一条单独受控路线；它可能去掉部署时的 matched 权重加载，但训练仍使用 teacher，不能宣称“从零无外部先验”。要真正完全从零，就需把目标头与 K=20 选点策略作成匹配原生覆盖的独立实验，先在固定 ETH 验证身份上同时达到原生 bank 的 Recall@0.5、P90 和 minFDE，再投入五折长训。可学习终点残差是另一种改变模型语义的研究方案，有机会突破 `eta=0` 的定值限制，但可能牺牲多模态候选与联合指标；当前没有验收数据，不应作为论文主结果或直接替换正式入口。

因此本次不切换正式训练的 checkpoint/HEAD、不启动新实验。**优先行动只有一个：把已验证的旧统一方案固定为论文主比较版本，保留从零八层版为未通过的研究变体。** 需要新主模型时沿用旧方案的固定选模与五 seed 结果，而不是让当前第 76 轮的较差权重覆盖其结果。当前运行是否提前停止是独立的训练控制决定，不能把运行中状态写成已完成。

## 证据索引与口径限制

- 旧统一正式指标与身份：[五折原始数据](../../results/ebjd_eth_ucy_unified_fivefold_20261101/PER_SEED_METRICS.csv)、[正式报告](../../results/ebjd_eth_ucy_unified_fivefold_20261101/REPORT_ZH.md)。五 seed 指五个**推理** seed，不是五个训练 seed。
- 当前有限 K 方法与训练合同：[设计说明](../20261115_finite_k_density_v1/DESIGN_ZH.md)、[研究报告](../20261115_finite_k_density_v1/REPORT_ZH.md)。
- 冻结终点和漂移诊断的原始哈希、逐 seed、逐行人表保存在上述持久资产目录，checkpoint 和 bank 未上传 GitHub。
- 本报告未检验 ETH 的 Goal-pretrain 与 matched Goal U-Net 是否逐参数等价；ZARA1/ZARA2 的已验证结论不能自动外推。也未重新运行完整五折、计算统计显著性或用不同验证集拼接成绩。
