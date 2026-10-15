---
experiment_id: EBJD-GPU-261015-001
date: 2026-10-15
snapshot_time: "22:12:24+08:00"
system: EBJD
experiment_type: training_status_and_diagnostic_audit
dataset: ETH
training_seed: 3101
run_id: eth_main_seed3101_d266681_20261015
execution_commit: d266681b7f08940f6caf5c20a01b41bb867b70b7
status: running_preliminary_negative
raw_log: outputs/formal_eth_main_20261015_seed3101/formal_training/eth/3101/none/eth_main_seed3101_d266681_20261015/train.jsonl
tags:
  - EBJD
  - ETH
  - JMM-253
  - CUDA
  - protocol-audit
  - distribution-shift
---

# EBJD ETH 当前状态与问题诊断

## 结论

截至本快照，ETH 正式运行已完成 **26/100 epochs**，epoch 27 正在运行，最后观察到 update 6872/25700。训练数值稳定，没有 JSON 损坏、NaN 或 Inf，也没有单位换算、场景覆盖或指标定义错误的直接证据；但尚无 checkpoint 同时满足冻结的 GDTS marginal 门槛。

当前性能问题不能归结为单一代码故障。已确认的主要因素是：当前 JMM-253 评测人口与 GDTS 论文/旧 raw-ETH 窗口不同；JMM ETH 的 endpoint/bridge latent 尺度明显超出四折训练分布；EBJD 的 Gaussian-init DDIM 没有复用 GDTS 的 TTST/tree endpoint proposal；以及 72.33% 的 ETH 窗口只有一名行人。前两项直接影响 marginal endpoint coverage，后两项使本折既难发挥 joint interaction，又不是只增加 joint 模块的单变量比较。

当前结果仍是**中途快照**：rollout 从 epoch 21 才开始，至 epoch 40 才完成权重 ramp。因此本报告可以确认当前差距和协议问题，不能提前宣告 epoch 100 的最终选择结果。

## 运行状态

| 项目 | 快照值 |
| --- | ---: |
| 状态判据 | 状态文件为 `RUNNING`，训练 JSONL 持续追加 |
| 已完成 epoch | 26 / 100 |
| 当前 epoch / update | 27 / 6872 |
| 计划更新数 | 25,700 |
| rollout 区间 | epoch 21 开始，epoch 40 ramp 完成 |
| 当前 rollout weight | 0.07；频率补偿后系数 0.28 |
| 最近完整 epoch 墙钟 | 2,346.50 s，约 39.11 min |
| 状态文件粗略剩余时间 | 172,365 s，约 47.88 h（epoch 26 边界估计） |
| 当前参数范数 | 118.781，稳定 |
| 非有限数值 | 0 个 NaN/Inf token |

PID 不作为本快照的独立存活证据，因为审查命令运行在隔离的进程命名空间中。运行状态以带时间戳的状态文件和持续增长的训练日志为准。

## 匹配协议指标

所有下表结果均为当前固定的 JMM ETH manifest：253 scenes、364 agent occurrences、20 worlds、20 DDIM steps、selection seed 2035。数值单位为米。

| 模型/epoch | minADE | minFDE | JADE | JFDE | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| GDTS matched baseline | 0.417053 | 0.580032 | 0.489907 | 0.743412 | 原生 TTST + tree/DDIM |
| EBJD epoch 5 | 0.576088 | 1.004969 | 0.634319 | 1.093059 | 当前最小总 marginal violation 的完整 checkpoint |
| EBJD epoch 6 | 0.576480 | 1.032828 | **0.631659** | 1.111155 | 单项最佳 JADE；不能与 epoch 5 其他指标拼接 |
| EBJD epoch 26 | 0.617004 | 1.152468 | 0.656580 | 1.208065 | 最近完整 checkpoint |

相对 matched GDTS，epoch 5 的 minADE/minFDE 分别高 **38.13%/73.26%**，JADE/JFDE 分别高 **29.48%/47.03%**。epoch 26 的对应差距扩大为 **47.94%/98.69%** 和 **34.02%/62.50%**。截至 epoch 26，符合 `minADE <= 0.417053` 且 `minFDE <= 0.580032` 的 eligible epoch 数为零。

## 已确认的问题

### 1. 当前 JMM-253 不是论文/旧 raw-ETH 的同一窗口人口

- 当前来源为固定的 `biwi_eth_agentformer.txt`，得到 253 scenes / 364 agent occurrences，源帧网格对应 stride 10。
- 旧 GDTS raw ETH 导出得到 904 scenes / 2,614 agent occurrences，主要使用 stride 6。
- 两套窗口的 start/end identity 交集为零，不能把 GDTS 论文中用户核对的 `0.31/0.48` 直接当作当前 manifest 的门槛。
- 这不是两份无关轨迹：把 raw track 插值到 JMM frame 后，覆盖 5,134/5,492 行，位置 L2 误差均值约 0.00384 m、P95 约 0.00599 m、最大约 0.00711 m。它们主要是同源轨迹的重采样和不同窗口选择。

当前 matched GDTS 在 JMM-253 上本身为 `0.417053/0.580032`，已经不同于论文 `0.31/0.48`。因此当前 EBJD 与 matched GDTS 的方法级差距是真实的，但不能再叠加一次与论文表值的直接差距解释。

### 2. 四折训练到 JMM ETH 存在明显 latent 尺度偏移

| 表征分量 | 四折训练 RMS | JMM ETH RMS | 比例 |
| --- | ---: | ---: | ---: |
| endpoint/goal | 1.059668 m | 2.071457 m | 1.95x |
| bridge residual | 0.207774 m | 0.500010 m | 2.41x |

EBJD 用四折训练数据拟合固定 scale，再用该 scale 标准化 held-out ETH。JMM ETH 的标准化目标因此明显大于训练常见范围。四折训练的 endpoint displacement P90 约 5.75 m，而 JMM ETH 约 9.71 m。这是已确认的 distribution shift；它导致 endpoint coverage 不足仍属于高可信机制解释，需要预测 payload 的 endpoint spread 审计才能最终闭环。

### 3. 方法级比较同时替换了 marginal sampler

GDTS matched baseline 使用原生 goal heatmap、TTST clustering 和 tree/DDIM sampling。当前 EBJD 直接从 Gaussian latent 生成 20 个 worlds，并执行 20-step deterministic DDIM，没有 TTST/tree endpoint proposal。因此 baseline 到 EBJD 的差异不只包含 joint module，也包含 endpoint candidate generator、完整模型初始化和训练目标变化。

EBJD 仅导入 epoch 16 的 goal U-Net（46 tensors），history/diffusion/joint 模块为 fresh initialization；GDTS comparator 是 epoch 90 的完整 checkpoint（112 tensors）。这是一项完整方法比较，而不是同一 marginal generator 上的严格 joint-only 消融。

### 4. ETH 对 joint interaction 的统计杠杆较弱

当前 253 个窗口的行人数分布为：`N=1: 183`、`N=2: 38`、`N=3: 25`、`N=4: 5`、`N=5: 2`。单人窗口占 **72.33%**，大多数场景不存在多人配对收益空间。epoch 26 的 coordination gap 只有 0.0270 ADE / 0.0724 FDE。

这与 UNIV 结果并不矛盾：UNIV 的多人密度更高，已观察到 EBJD 在相同评测下优于 matched GDTS。当前证据更支持“ETH 的 marginal coverage 问题压过 joint 收益”，而不是“joint 机制在所有数据集都无效”。

## 已排除或尚未证实的解释

- **未发现单位/坐标错误：** world-pixel-world 最大往返误差为 `7.105e-15 m`。
- **未发现场景漏评：** EBJD 与 matched GDTS 都覆盖 253 scenes / 364 agent occurrences。
- **未发现数值崩溃：** 训练日志可完整解析，未出现 NaN/Inf；参数范数和梯度保持有限。
- **公式代码审查未发现差异：** 两边的 marginal 与 scene-common joint 公式在数学上等价；但“同一 prediction tensor 输入两套 evaluator”的数值 parity 尚未执行，所以仍保留为 OPEN。
- **BF16/FP32 不足以解释主差距：** matched GDTS receipt 使用 BF16 CUDA autocast，EBJD 正式评估路径使用 FP32。它是次要可复核差异，不可能合理解释约 38%--73% 的 marginal 差距。

## 下一步建议

当前 run 保持冻结配置继续完成，作为正式结果和负结果证据，不在 epoch 27 中途修改 scale、sampler 或目标。GPU 空闲后按以下顺序诊断：

1. 对同一 prediction payload 同时运行 EBJD 与 GDTS evaluator，关闭公式 parity 的最后一个 OPEN 项。
2. 按 `N=1` 与 `N>=2` 分层比较 EBJD/GDTS，判断 joint 模块是否只在多人窗口改善。
3. 审计 endpoint best-of-20 距离、预测终点半径/方差、GT endpoint correction 分位数，验证 coverage/under-dispersion 假设。
4. 用同一冻结 GDTS checkpoint 在旧 raw-ETH 与 JMM-253 上以统一精度评估，定量拆分 protocol shift。
5. 下一轮建立同架构同 sampler 的 `interaction-off` EBJD marginal control；再单变量加入 joint 模块。若修复 endpoint coverage，优先测试 observation-conditioned/robust scale 或把 TTST 类 endpoint proposal 接入 EBJD，但不得用 held-out future 拟合 scale。

## 可复核证据

- 结构化快照：`ebjd_clean/docs/validation/EBJD_ETH_STATUS_AND_DIAGNOSIS_2026-10-15.json`
- 训练状态：`ebjd_clean/outputs/formal_eth_main_20261015_seed3101/PIPELINE_STATE.json`
- 正式配置：`ebjd_clean/outputs/formal_eth_main_20261015_seed3101/archive/endpoint_bridge_joint_eth_seed3101.yaml`
- 数据收据：`ebjd_clean/outputs/formal_eth_main_20261015_seed3101/archive/ETH_DATA_RECEIPT.json`
- matched GDTS：`ebjd_clean/outputs/formal_eth_main_20261015_seed3101/baseline/GDTS_ETH_JMM_SELECTION_SEED2035.json`
- 验证曲线：`ebjd_clean/outputs/formal_eth_main_20261015_seed3101/ebjd_training.log`
- update 级日志：`ebjd_clean/outputs/formal_eth_main_20261015_seed3101/formal_training/eth/3101/none/eth_main_seed3101_d266681_20261015/train.jsonl`

大日志、模型 checkpoint 和数据 shard 未复制进 Git；GitHub 归档只包含本报告和结构化小型快照。
