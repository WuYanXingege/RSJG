# ZARA1 真实终点条件桥三轮监督对照

## 结论

状态：**COMPLETE / SUPERVISION_SUPPORTED_BUT_DO_NOT_ADOPT_AS_MAIN_OR_PARETO**。

训练时用增强后的真实终点作为条件、并用完整真实轨迹监督，确实修复了旧条件桥的主要路径学习缺口；但固定 H 推理仍在 JADE/JFDE 上明显弱于原 Candidate，未达到预注册的路径竞争力容差。因此：保留原 Candidate 为完整预测研究主线，保留 MPJC 为终点耦合机制证据，停止固定终点条件桥作为主模型或覆盖优先 Pareto 变体的继续调参。本轮关闭 ZARA1 局部优化循环。

## 设置与身份

- 唯一初始化：Candidate SHA256 `4b08aa68fef82a0ee31663d4c860311aa5ff260afb4946072f05de75b668dfed`。
- 新模型：`GT_CONDITIONED_BRIDGE_E3`；固定第 3 轮 checkpoint SHA256 `9ecf25b4ff607e4c3a6cdce985b9ee96415b175b87ef358c59d6267a663288cb`。
- 训练 seed 3101，严格 3 轮、810 次 optimizer update；主干/U-Net 学习率 `2e-5/2e-6`，rollout 与 paired projection 全程关闭。
- 结构保持 192 hidden、4 heads、FFN768、6 FJAB，FULL、legacy CTF；训练条件为增强后的 GT endpoint，推理只读取冻结 G/H bank。
- validation manifest SHA256 `9c12e30ac17e91bbf8efa9e23a3d61a16c9e9c8dfc6414871fc7431b76379478`；20 worlds、20 DDIM steps；五个 seed 是一个训练 checkpoint 的推理重复。

## 五种子正式结果

均值 ± 样本标准差，越低越好。

| 模型/条件 | minADE | minFDE | JADE | JFDE | prefix minADE/JADE（前11帧） |
|---|---:|---:|---:|---:|---:|
| 新桥 + G | 0.184354 ± 0.000906 | **0.274816 ± 0.000666** | 0.302674 ± 0.000727 | 0.572203 ± 0.003532 | 0.166456 / 0.272282 |
| 新桥 + H | **0.183750 ± 0.000796** | **0.274816 ± 0.000666** | 0.287978 ± 0.002386 | 0.535355 ± 0.005775 | **0.165920 / 0.259491** |
| 原 Candidate | 0.192292 ± 0.000934 | 0.309826 ± 0.003587 | **0.275205 ± 0.000737** | **0.510408 ± 0.003630** | 未归档 |

H−G 均值为：minADE `-0.000604`、minFDE `0`、JADE `-0.014696`、JFDE `-0.036848`。H 在 minADE、JADE、JFDE 上均为 5/5 seed 改善；minFDE 由相同逐人终点集合决定，5/5 完全相同。

## A：监督改动有实际价值

结论：**SUPPORTED**。仅比较同一 frozen bank 的 seed2035：

| 对照（新−旧） | minADE | JADE | prefix minADE | prefix JADE |
|---|---:|---:|---:|---:|
| 新 G − 旧 BRIDGE_NATIVE | -0.041468 | -0.015422 | -0.021190 | -0.003469 |
| 新 H − 旧 BRIDGE_COUPLED | -0.037945 | -0.028034 | -0.018734 | -0.017217 |

四类路径误差方向一致下降，说明将条件终点与完整监督轨迹对齐具有实际价值，收益并非只来自冻结终点。它支持“旧监督构造是重要影响因素”，但三轮 warm-start 不能证明这是全部差距的唯一根因。

## B：不替代原 Candidate，也不保留为预注册 Pareto 变体

新 H−Candidate 五种子均值：minADE `-0.008542`、minFDE `-0.035010`、JADE `+0.012773`、JFDE `+0.024947`。逐 seed 方向为：minADE 5/5 改善、minFDE 5/5 改善、JADE 0/5 改善、JFDE 0/5 改善。

新 H 的 minADE 满足“相对 Candidate 不差超过 0.003 m”，但 JADE 比 Candidate 差 `0.012773 m`，超过预注册的 `0.005 m` 实用容差；固定 H 的 JFDE 也按预期高于 Candidate。因此它是清楚的边缘覆盖/联合精度权衡，但路径联合精度尚不足以作为覆盖优先 Pareto 变体进入主线。

## C：GDTS 合同

绑定 seed2035 上，新 H 为 `0.184084/0.274462/0.287672/0.542792`，matched GDTS 为 `0.188109/0.285034/0.319607/0.586878`。两项 marginal 均不退化、两项 joint 均改善，故原 seed2035 合同为 **PASS**。matched GDTS 没有五个绑定推理 seed，因此五种子合同仍为 **NOT_EVALUABLE**，没有补造或重跑 baseline。

## 成本、边界与最终取舍

- 训练 `2825.15 s`，新 G/H 五种子评测 `2022.35 s`，旧桥两次 prefix 回放 `412.15 s`；合计 `5259.65 s`（1.461 h），低于 7200 s 硬上限。
- 训练峰值显存 `3593318400 bytes`（约 3.35 GiB）。
- 本轮完成后没有训练进程；除规定的 810 次短训更新外，没有恢复原 250 轮训练或追加优化器更新。
- 论文中可诚实陈述：GT 一致的条件桥监督显著改善路径学习；MPJC 在保持逐人终点集合时稳定改善联合终点；但固定终点条件桥仍未达到原 Candidate 的联合路径精度，不能作为统一正式模型。
- 最终路线：**原 Candidate 保留为完整预测主线；MPJC 保留为机制证据；固定终点条件桥停止。** 后续若开展工作，只能另立统一跨折复验/必要消融任务，本轮不自动启动。
