# HOTEL A0/A1 当前状态、可比性与设计结论边界

快照时间：2026-10-07 20:19:23 +08:00

审查仓库 HEAD：`bd7c5c035056669b6e6e9f10e09fb92abd337ad7`

正式训练执行源码：`664ad0410f5cf3b44d3a6c1892c98e37aa936b9e`

状态：`PARTIAL_QUEUE_RUNNING_FINAL_ARM`

> 2026-10-07 勘误：本报告初版第 5 节曾把待执行机制干预误写为
> `objective-off、edge-permutation`。冻结队列 manifest、生成脚本与 sampler
> 源码的实际合同一致，三项是 pair-cost `off`、`physical` 和
> `pure_interaction_off`。这是文案错误，不是队列接线或执行错误；核验记录见
> [QUEUE_MANIFEST_CHECK.json](QUEUE_MANIFEST_CHECK.json)。

## 1. 结论

目前不能得出“设计无意义”或“网络有问题”的结论。已经能够确认的是：

- **A0 与 A1 是当前最强的配对因果比较。** 每个训练种子共享认证父权重、零步状态、数据、预算、优化器、选模、精度与推理种子；配置审计显示实质变量是目标函数。
- **原生 GDTS 与 A0/A1 只能通过尚待执行完毕的 `E_joint` 作方法级比较。** 它们对齐外部评估窗口、单位、mask、P20、五个推理种子和指标实现，但内部生成算法、参数量和计算量不同；这些差异是被比较的方法本身。
- **历史 `E_official=0.13394/0.19140 m` 和论文表格不能直接与 A0/A1 指标作严格差值。** 它们不是同一随机采样与联合指标认证入口。
- HOTEL validation/test 当前为同字节数据。这个边界不破坏同暴露条件下的内部 A0/A1 配对，但不构成独立 held-out 泛化证据。
- 是否证明设计有效，必须等待全部六臂结束，并完成 `E_joint`、五推理种子正式评估和机制干预；训练期间的 validation best 值不能替代正式结果。

## 2. 运行状态

队列已完成 5/6 个训练臂：

| arm | 状态 | 完成轮数 | 日志中的选模 best 摘要（仅阶段性） |
|---|---:|---:|---|
| A0_3101 | COMPLETE | 40 | ADE_world 0.135, FDE_world 0.190, JADE 0.199, JFDE 0.336 |
| A1_3101 | COMPLETE | 40 | ADE_world 0.134, FDE_world 0.189, JADE 0.198, JFDE 0.332 |
| A1_3102 | COMPLETE | 40 | ADE_world 0.134, FDE_world 0.189, JADE 0.199, JFDE 0.339 |
| A0_3102 | COMPLETE | 40 | ADE_world 0.134, FDE_world 0.188, JADE 0.199, JFDE 0.333 |
| A0_3103 | COMPLETE | 40 | ADE_world 0.134, FDE_world 0.190, JADE 0.199, JFDE 0.336 |
| A1_3103 | RUNNING | epoch 6 已保存，epoch 7 运行中 | 正式结果 PENDING |

注意：表中每个 `best_*` 可能来自不同 epoch，只用于证明训练/选模链在运行，不能把同一行拼成一个实际 checkpoint 的联合成绩，也不能据此执行预注册门槛。

在快照时，A1_3103 日志文件在 2 秒观察窗内由 3,557,407 增至 3,557,710 bytes，说明输出仍在推进。队列登记的 manager/worker PID 在当前受限审查命名空间内不可从 `/proc` 回读，因此本报告只声明日志推进，不补造进程存活认证。

## 3. 哪些比较公平

### 3.1 A0 vs A1：是，严格配对

预注册比较单位是每个训练种子内部先对五个推理种子求均值，再计算 `A1-A0`，最后跨训练种子汇总。A0/A1 配置差异审计为 PASS；三对实验都从对应的相同认证零步状态开始。

这个比较回答：**expected-conditional 目标相对于 A0 原目标是否有增量贡献。**

### 3.2 E_joint baseline vs A0/A1：设计上公平，结果尚未验收

`E_joint` baseline 评估器冻结官方 epoch 110 checkpoint，并关闭 JDV2、dynamic relation、joint energy 与 corrector；它使用与 A0/A1 相同的 HOTEL loader、同步窗口、P20、2035--2039 推理种子以及 ADE/FDE、minADE/minFDE、JADE/JFDE、Collision Rate 计算。

内部采样不完全相同：baseline 必须保留原生 GDTS 生成语义，A0/A1 使用 K21 joint assignment/refinement。若强行让 baseline 使用 JDV2 的内部候选或 assignment，它反而不再是原生 baseline。因此这是有效的**方法级比较**，不是逐候选、逐噪声完全配对的单开关实验。

这个比较回答：**在统一评估条件下，给冻结 GDTS 父模型增加 Stage-A joint selection 是否产生净收益。** 需要同时披露额外参数、耗时和显存。

### 3.3 E_official/论文 vs A0/A1：不能严格直接比较

官方本地 HOTEL 复现的五次 TTST 为 ADE/FDE `0.13394/0.19140 m`，可认证官方父模型本身，但该入口没有和 A0/A1 共用完整的 `E_joint` 随机与联合指标合同。因此它只能作背景参考，不能作为 A0/A1 的正式差值基线。

## 4. 什么结果才说明设计有效或有问题

| 最终证据 | 可支持的解释 |
|---|---|
| A0 优于 E_joint，A1 又稳定优于 A0 | Stage-A 机制与新目标都有增量价值 |
| A0 优于 E_joint，A1≈A0 | joint 机制有价值，新目标贡献未证实 |
| JADE/JFDE或碰撞率改善，ADE/FDE基本保持 | 符合“改善联合世界、保持 marginal”的设计目的 |
| joint 指标改善但 marginal 明显恶化 | 存在 joint--marginal 权衡，需要修改目标权重或约束 |
| A0/A1 都与 E_joint 相当 | 当前实验未证明 Stage-A 增益；需区分候选上限、能量辨识和训练目标错位 |
| A0/A1 都稳定劣于 E_joint | 才应重点怀疑 assignment/refinement 或训练语义 |
| A1 稳定劣于 A0 | 首先否定 A1 目标，不等于整个网络结构错误 |

Stage-A 只在既有 K21 候选上进行联合选择，不能创造候选集合中不存在的轨迹。因此负结果还必须结合 oracle/candidate-bank 上限与机制干预，才能定位为候选不足、能量辨识失败或目标错位。

## 5. 尚未完成的正式证据

快照时以下项目均为 PENDING：

1. A1_3103 完成及第六份训练收据；
2. 冻结父模型的 `E_joint` baseline；
3. A0/A1 三对训练种子 × 五推理种子的正式测试；
4. 预注册的 paired effects 与不确定性汇总；
5. 预注册的三项机制干预：pair-cost `off`（pair-off）、`physical`，以及
   `pure_interaction_off`（保留可加行/列项的 interaction-off）；
6. 完整窗口数、样本数、seed 列表和 checkpoint SHA 的最终收据验收。

在这些证据生成前，阶段性 validation 数字不得写成“优于 baseline”，也不得把当前结果解释为网络失败。

## 6. 证据入口

- [实验预注册](../2026-10-06_62cfcae_hotel_a0_a1_bounded/EXPERIMENT_PREREGISTRATION.md)
- [A0/A1 配置差异](../2026-10-06_62cfcae_hotel_a0_a1_bounded/PAIRED_CONFIG_DIFF.json)
- [官方协议 manifest](../2026-10-06_62cfcae_hotel_a0_a1_bounded/OFFICIAL_PROTOCOL_MANIFEST.json)
- [阶段性 CPU 审查](../2026-10-07_3eead0c_hotel_a0_a1_final_cpu_diagnostics/REVIEW.md)
- [机器可读状态](STATUS.json)
- [队列机制任务核验与勘误](QUEUE_MANIFEST_CHECK.json)

本轮仅归档和上传状态，没有修改模型、配置或训练过程，也没有启动新的 GPU 任务。
