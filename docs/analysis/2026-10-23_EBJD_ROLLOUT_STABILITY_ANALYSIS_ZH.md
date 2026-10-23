# EBJD rollout 稳定增益复核

日期：2026-10-23  
结论状态：`NO_STABLE_GAIN_CONFIRMED`

## 结论

现有证据不能支持“EBJD rollout 带来稳定净增益”。更准确的结论是：
rollout 可能在部分多人场景中改善联合预测，但其总体效果存在明显的
边缘—联合指标权衡，且跨 fold 方向不一致。

本文所称 rollout 机制包含可微 20-step rollout loss 与 paired marginal
actual-step projection。已有干预没有把这两部分进一步拆开，因此不能把结果
单独归因于其中任一部分。

## 核心证据

### 1. ETH 严格配对 ON/OFF：总体负向

同一 epoch20 完整状态分叉，ON/OFF 各训练 epoch21–25、各 1,285 次更新；
模型、优化器、学习率、样本、增强及基础 RNG 完成配对。ON 执行 322 次
rollout/projection，OFF 为 0 次。五个推理 seed 为 2035–2039。

| 对象 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| OFF epoch25 | 0.474538 | 0.714222 | 0.548284 | 0.890216 |
| ON epoch25 | 0.477116 | 0.725344 | 0.551257 | 0.898173 |
| ON−OFF | +0.002579 | +0.011122 | +0.002973 | +0.007957 |

四项指标均为越低越好；ON 在四项上均退化，且四项都是 `0/5` 个推理 seed
改善。冻结判断为：`CONFLICT: ON退化而OFF保持或改善`。

存在一个条件性信号：ETH 的 `N>=2` 子群中，ON 相对 OFF 的四项均值均略有
改善；但 `N=1` 子群明显退化，最终使总体指标反向。因此该结果更符合
“人数条件下的偏置”，而非稳定总体收益。

### 2. HOTEL rollout 权重配对：边缘与联合指标发生权衡

从同一 epoch45 状态继续到 epoch50，仅将 rollout 末端权重从控制臂 `0.20`
改为候选臂 `0.05`，rollout 频率及 forward/backward 保持不变。

| 对象 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| 控制 `0.20` | 0.146598 | 0.219778 | 0.176315 | 0.295207 |
| 候选 `0.05` | 0.144859 | 0.214739 | 0.177966 | 0.296955 |
| 候选−控制 | -0.001739 | -0.005039 | +0.001651 | +0.001749 |

降低 rollout 权重后，minADE/minFDE 在 `5/5` 个推理 seed 改善，但
JADE/JFDE 在 `0/5` 个 seed 改善。短边缘目标通过，联合保持与综合目标均未
通过。候选和控制也都没有超过保留的 epoch45 checkpoint；进一步权重搜索被
判定为 `NOT_SUPPORTED`。

这说明更强 rollout 可能轻微偏向联合指标，但不能产生四项一致的净改善。

### 3. UNIV rollout-off 续训：不能证明因果，但削弱“必要性”主张

UNIV epoch81 Control 是从正式 epoch73 checkpoint 继续 8 轮、2,040 次更新的
`rollout_mechanism_off` 运行。epoch81 相对 epoch73 的 minADE、minFDE、JADE、
JFDE 均改善，但它没有相同预算的 ON 对照，因此不能据此证明 OFF 优于 ON。
它只能说明：这些改进能够在 rollout 关闭时出现，不能作为 rollout 独立贡献证据。

### 4. FULL 与 GDTS 的方法级差异不能归因于 rollout

HOTEL epoch45 FULL 相对 matched GDTS 改善 JADE/JFDE，却退化 minADE/minFDE，
原性能合同为 `FAIL`。FULL 同时包含表示、网络、反馈、损失与优化器约束等多项
差异，不能把联合指标改善单独归给 rollout。

## 证据边界

- 每个 checkpoint 只有一个训练 seed（3101）。
- 2035–2039 是同一 checkpoint 的五次随机推理，不是五个独立训练重复。
- HOTEL 与 UNIV 的 validation/test 来自官方包中的镜像成员，不是独立测试集。
- ETH ON/OFF 只覆盖 rollout ramp 的早期 epoch21–25。
- ETH 干预同时开关 rollout loss 和 marginal actual-step projection。

因此，当前最严谨的论文表述应为：rollout 是正式训练合同的一部分，但尚未被
验证为稳定、独立、跨 fold 的净增益来源。它不宜作为已被消融充分支持的核心
贡献；所有负结果与权衡应保留。

## 对当前训练的处理

当前 ZARA1 流水线应继续使用冻结的正式训练合同，不在训练中途移除 rollout，
以保持与 HOTEL/UNIV 的协议可比性。本报告不修改现有模型选择、配置或运行任务，
也不自动启动新的消融实验。

## 固定来源

- 论文工作区实验登记：`docs/EXPERIMENT_REGISTRY.md`
- HOTEL rollout-weight 0.05 pilot：本地正式归档
  `EBJD_ROLLOUT_WEIGHT005_PILOT_2026-10-12.{md,json}`
- ETH rollout boundary pilot：实现提交
  `b669f621e2e189d59a740253943476dc0cfa1c1c`
- UNIV epoch81 Control 正式评测来源：
  `b338611b35d75d2fcebea9c25aca89cb6a0035be`
- 本报告基线提交：`de2a5b01344a435c02b9474995cbe5f7211d2403`
