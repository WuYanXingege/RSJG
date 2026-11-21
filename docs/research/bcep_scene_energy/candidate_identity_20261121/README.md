# SSEC 候选重编号稳定性定点修复

**本轮限定门槛通过：`READY_ENGINEERING_TRAINING_DISABLED`。原 N57 的 101 个跨人选择变化已降为 0。** 在固定 ZARA2 成员顺序、冻结 bank 和原训练协议内，上一轮候选重编号工程阻塞已解除；不等于获得训练授权，也不等于对任意行人重排、跨设备运行或训练效果提供保证。本轮未启动正式训练。

基点：`54ef175c3cb2601c60ec178402831cee406eec6e`。真实 optimizer updates **0**，合成 optimizer updates **0**。新增模块、损失及参数量均未改变。权重仍为同一 seed3201 合成验收初态；没有加载历史 v3 权重作为 SSEC 初始化。

[新 READINESS](READINESS.json) 保留上轮全部准入门槛名称，增加身份和噪声检查；[CPU 收据](CPU_RECEIPT.json)、[GPU 收据](GPU_RECEIPT.json)、[原生接口收据](INTERFACE_RECEIPT.json)、[资产收据](ASSET_RECEIPT.json) 可独立核验。

## 1. 原失败如何发生

按上轮相同 `fixture(seed=71)`、相同耦合头初始化 seed3201 和重编号 seed2035，先执行固定提交的原入口，精确复现旧结果。

| N | 原 S0 首配改变格数 | 原最终跨人改变格数 | 修复后 CPU/GPU 首配改变 | 修复后 CPU/GPU 最终改变 |
|---|---:|---:|---:|---:|
| 2 | 0 | 0 | 0 / 0 | 0 / 0 |
| 9 | 12 | 42 | 0 / 0 | 0 / 0 |
| 20 | 2 | 18 | 0 / 0 | 0 / 0 |
| 42 | 34 | 119 | 0 / 0 | 0 / 0 |
| 57 | 22 | **101** | 0 / 0 | **0 / 0** |

逐层定位结果：

- 输入端点逆映射、原 endpoint 编码和 candidate_norm 特征均逐值相同，最大差为0。
- 差异首次出现在 S0 world-query attention。候选 token 的数组顺序改变归约顺序，world 输出最大差约1.19e−7～2.98e−7；评分最大差约5.96e−8～1.19e−7。
- 这些变化会改变近并列完整匹配。N9/N42/N57 初配中分别有1/6/5位行人的最优解严格并列。这里用原整数格匹配证书区分严格并列，没有以评分RMS代替匹配间隔。
- 将同一物理初配传递到重编号后的原能量协调器，最终跨人变化均为0，支持“首配敏感性是这组失败的起点”。随后顺序协调可以扩大初配差异，但不是候选特征已经变化的证据。

原始数据见 [BASE_LAYER_LOCALIZATION.csv](BASE_LAYER_LOCALIZATION.csv)、[BASE_INITIAL_MATCHES.csv](BASE_INITIAL_MATCHES.csv)。

## 2. 限定修复与身份来源

只在原 `SceneEnergyCoupling.assign` 前后增加可逆身份变换，神经计算与协调规则保持原设计。新代码见 [identity.py](../../../../bcep_scene_energy/identity.py)、[coupling.py](../../../../bcep_scene_energy/coupling.py)、[inference.py](../../../../bcep_scene_energy/inference.py)。

原 bank 没有独立的 candidate-ID 数组，但经校验的 **bank哈希 + 行成员/文件哈希 + 行人身份 + 原KMeans槽位** 可以标识一个不可变的候选出现实例。`bank_occurrence_ids` 在读取边界显式导出这个身份；重排时ID随候选一起gather，刚体增广不改变ID。重排后的数组位置不能重新冒充原槽位。

内部规范顺序采用精确FP32的 `(x, y)` 字典序；坐标完全相同时，以已传递的原槽位区分实例。对互异坐标，有无ID的默认入口采用完全相同的规范顺序。这样保持现有训练回调可用，也不把ID作为神经输入。

未做坐标取整、容差合并、随机epsilon、评分放大或序号embedding。候选坐标仅gather，输出排列精确映射回调用方数组顺序。ST坐标代理也在相同规范顺序内归约，避免硬排列一致但反向归约仍随候选数组顺序改变。

完整 bank 只读扫描覆盖训练2681窗、开发590窗：**完全重合候选对为0**，最小候选L∞间距约0.01104546 m。[BANK_IDENTITY_SCAN.csv](BANK_IDENTITY_SCAN.csv) 保留逐窗哈希和最小间距；未重新生成任何终点。

重复身份仍有信息论边界：只给完全相同的两个端点，无法恢复它们原本对应的两个不同keyed-noise实例。携带原槽位时可精确区分；没有ID则明确抛出 `CandidateIdentityAmbiguity`，不按当前下标猜测。极近但不相等的FP32坐标仍分别排序，测试包含相邻1 ULP、正负零及增广舍入后重合的情况。

[IDENTITY_PROOF_ZH.md](IDENTITY_PROOF_ZH.md) 给出逆映射公式、单人与padding约定和可调用接口；[CANDIDATE_MAPPING_PROOF.csv](CANDIDATE_MAPPING_PROOF.csv) 保留逐候选映射证据。

## 3. 端到端与梯度验收

**32项CPU测试通过**：原19项测试文件未删除、未修改，新增13项覆盖默认入口、严格S0并列、重复/近重合身份、同步增广、padding、梯度及可选CSEE接口。

CPU及GPU上，五个多人规模的规范输入、endpoint特征、候选特征、world attention、S0评分均逐值一致；初次Hungarian和SSEC最终排列映射回原身份后完全一致。检查没有传入旧incumbent，均重新计算默认S0。φ/ρ和最终评分梯度有限、非零，重编号前后φ/ρ梯度逐值相同；候选梯度为None。见 [LAYER_AND_GRADIENT_COMPARISON.csv](LAYER_AND_GRADIENT_COMPARISON.csv)。

原完整 `bridge_and_pair_loss` 也在N=1、2、9、20、42、57分别进行了同噪声合成反向：总损失差为0，全部模型参数梯度逐值相同，没有optimizer更新。[FULL_OBJECTIVE_GRADIENTS.csv](FULL_OBJECTIVE_GRADIENTS.csv) 记录完整训练图检查。

同样六个人数的原20步、K20 G/H DDIM完成了两次身份对照：G全路径、H全路径、H末帧最大差均为0。另有N2重复端点、N9原生同步旋转/反射增广案例，结果也逐值一致。[END_TO_END_DDIM.csv](END_TO_END_DDIM.csv) 记录结果。

**G的定义保持原bank槽位组合。** 独立打乱每个人后直接按新数组下标组装G，会改变跨人组合，不是纯重编号。薄适配器先按原实例ID恢复bank槽位，再调用原G/H推理；原噪声key仍是 `(scene, agent, original_slot, seed)`。旧噪声函数、DDIM及桥网络均未修改。原顺序输入下，适配器与未经修改的原生推理G/H逐值一致。

噪声负对照进一步证明末帧检查不够：若错误地按重排后下标重新生成噪声，N2/N9的H末帧仍相同，但完整H最大差分别约0.38443/0.48819 m。见 [NOISE_IDENTITY_NEGATIVE_CONTROL.csv](NOISE_IDENTITY_NEGATIVE_CONTROL.csv)。这也是为什么不能仅修复输出排列而忽略噪声身份。

官方数据额外检查仅限4个selection_dev只读前向探针，N=1、2、9、42；初配、最终配对和边缘minFDE差均为0。未计算新模型效果表、未选模、未读取测试集。见 [OFFICIAL_BANK_RELABEL.csv](OFFICIAL_BANK_RELABEL.csv)。

## 4. 行人顺序依赖没有消失

本轮没有新增行人ID排序，默认访问顺序仍是有效行人数组顺序。须区分三件事：

1. **只给world统一换编号**：携带同一incumbent并映射world后，全部CPU/GPU样本0变化。
2. **真正反转物理访问顺序**：会改变坐标式局部更新轨迹。默认入口、原行人数组不变，仅反转访问顺序的跨人变化如下。
3. **重排行人数组但保持同一物理访问顺序**：给定同一物理incumbent时0变化；若重新计算S0，attention的行人token顺序仍会改变浮点结果，可能得到不同首配。这是残留的行人数组敏感性，不应全部归因于协调器访问顺序。

| N | 仅反转物理访问顺序：CPU / GPU | 行人数组重排且保持物理访问顺序、重算S0：CPU / GPU |
|---|---:|---:|
| 2 | 0 / 0 | 0 / 0 |
| 9 | 98 / 97 | 0 / 61 |
| 20 | 272 / 269 | 0 / 0 |
| 42 | 462 / 427 | 57 / 129 |
| 57 | **652 / 588** | **119 / 133** |

以上“跨人变化”将两份完整排列分别按**同一物理参考行人的候选身份**对齐world后计算；它衡量相对该行人的关系，不是最小全局world置换距离，可能大于原数组逐格差。CSV同时给出原始逐格差，避免将两种分母混淆。

[VISIT_ORDER_AUDIT.csv](VISIT_ORDER_AUDIT.csv) 保留固定/反序/行人数组重排/统一world重编号全部结果。当前准入范围是原已冻结成员顺序；没有降低上一轮门槛，也没有将“任意行人重排严格等变”新写成已通过。若下一轮改变行人排列协议，应先单独验收，不能引用本轮候选门槛为其背书。

## 5. 成本变化

同一GPU、同一模块参数、同一输入，切换固定基点与修复入口；每项1次warmup、3次测量，FP32，无optimizer.step。分配次数和参数量未变，仍为7,627,950个参数；多人Hungarian调用仍为2N，N57为114次。

| N | 原完整目标前反向中位秒 | 修复后中位秒 | 后/前 |
|---|---:|---:|---:|
| 1 | 0.16761 | 0.16758 | 1.000 |
| 2 | 0.20451 | 0.20571 | 1.006 |
| 9 | 0.34665 | 0.34556 | 0.997 |
| 20 | 0.58705 | 0.58726 | 1.000 |
| 42 | 1.14759 | 1.15013 | 1.002 |
| 57 | 1.73792 | 1.68686 | 0.971 |

这组短测没有明显额外完整目标成本；N57较快不能解释成可靠算法加速，规范顺序可能改变近并列首配及后续分支，且只有3次测量。分配单独前向的额外排序开销在小人数约0.25～1.1 ms量级。N57本进程allocated峰值约2892MiB，含该脚本同时保留的比较输入/上下文等；不能与上轮不同生命周期的峰值直接做显存优化结论。未测AdamW首次step状态分配。

[GPU_COST_RAW.csv](GPU_COST_RAW.csv) 和 [GPU_COST_SUMMARY.csv](GPU_COST_SUMMARY.csv) 提供原始计时及分配/完整目标分项。唯一最优检查源码完全未改，其既有300例计时结论冻结引用上轮收据；精确算法及原CPU数学测试本轮重新执行，没有声称新修复进一步加速唯一性检查。

## 6. 研究边界、准入与资产

公共φ/ρ、共享能量、替换评分、整数唯一性检查、FP64能量核验、proposal-ST、六层桥与final_coordinate均保持原设计；`CoordinatedAssignment`类体逐字节不变。原bcep/ebjd/v2/v3/正式控制器及配置目录均无源码修改。原SSEC测试全部保留。

上轮完整GPU脚本也已原样重跑：v2、完整v3、Energy-only、可选CSEE × 全部人数，另含最密57人图，共168条正式测量记录（每项另有warmup），均通过。最密57人主结构allocated峰值约2252MiB，可选CSEE约2263MiB；这套回归的对象生命周期与主成本脚本不同，峰值分别报告。[完整GPU回归收据](gpu_full_regression/GPU_RECEIPT.json) 和 [原始记录](gpu_full_regression/GPU_BENCHMARK.csv) 可核对。

原生分组backward及update=0状态保存/严格模型优化器重载已重新执行；没有伪造完整epoch，也没有通过被禁止的step重测Top5/早停。原正式五折仍PAUSED_RESUMABLE，candidate_cost仍SESSION_BUDGET_PAUSED，sharedworld/v3仍COMPLETE。历史权重、bank、旧审计资产及上一轮SSEC验收资产共45,650个文件、12,875,492,295字节逐文件核验未变；新零更新测试快照仅写本轮独立诊断目录。

五项明确回答：

- **原101个变化能否消除？能，本轮原案例0变化，CPU/GPU均通过。**
- **S0是否逐值一致？是，同一规范输入、同一设备和固定参数下逐值一致；不承诺跨CPU/GPU逐值一致。**
- **重复身份是否存在不可避免的歧义？是，只有坐标时无法恢复不同噪声实例；原bank槽位元数据能解除该歧义，无元数据则拒绝。**
- **协调顺序依赖还剩多少？N57反转物理顺序仍有652/588个相对参考行人的改变；行人数组重排本身仍可能影响S0。**
- **是否达到正式训练工程准入？在当前冻结ZARA2候选、成员顺序及原协议范围内，原有门槛全部通过，可以提交一次受控训练的审批；当前发布仍禁用训练入口，尚未获本轮真实更新授权。**

本结果未解决Sinkhorn近均匀、硬软梯度偏差或学习有效性，也没有新的JADE/JFDE效果结论。不自动扩折、换seed、恢复旧队列或启动测试集。复现见 [REPRODUCE.md](REPRODUCE.md)，配置与源码哈希见 [FROZEN_CONFIG.json](FROZEN_CONFIG.json) 和 [CONFIG_SHA256.txt](CONFIG_SHA256.txt)。
