---
experiment_id: EBJD-GPU-261012-001
date: 2026-10-12
snapshot_time: "2026-10-12T09:44:40+08:00"
system: EBJD
experiment_type: five_seed_inference_evaluation
dataset: HOTEL
training_seed: 3101
inference_seeds: [2035, 2036, 2037, 2038, 2039]
evaluation_source_commit: 9ecde66aa86f7a577fa29bd94b0168ae735a1f16
checkpoint_epoch: 45
status: completed_internal_mirror_evaluation
device: NVIDIA_GeForce_RTX_5070_Ti_16GB
tags:
  - EBJD
  - HOTEL
  - CUDA
  - P20
  - five-seed
  - internal-evidence
---

# EBJD HOTEL epoch 45 五推理种子评估

## 结论

epoch 45 权重已完成 seeds 2035–2039、20 worlds、20 DDIM steps 的完整推理。
五种子均值为 minADE/minFDE `0.138007 / 0.202037`、JADE/JFDE
`0.168566 / 0.276314`。全部 95 个数值叶节点有限，每个 seed 均覆盖 445 个场景
窗口和 1,197 个行人实例。

相对绑定的 GDTS 五种子均值，EBJD 的 minADE/minFDE 分别差 2.70% / 5.45%，
JADE/JFDE 分别改善 19.14% / 22.61%。五个 seed 的差值方向一致：marginal
均退化、joint 均改善。因此结果支持“联合一致性收益伴随 marginal 代价”，不支持
“联合指标改善且 marginal 无损”的预注册合同。

## 冻结身份

- 评估源码：`9ecde66aa86f7a577fa29bd94b0168ae735a1f16`，工作树 clean。
- checkpoint：epoch 45，SHA256
  `425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1`。
- frozen config SHA256：
  `a3800984c4876bec69caa62a587ab2db609d2dce1c7670995ddee0ecb6440278`。
- test manifest SHA256：
  `84c95cb431caf7214d1b85e8e7dbfeaf48c9fe301b1cda7f3201c3e8cf199c78`。
- 原始评估 JSON SHA256：
  `5e0f13ab4cb272aebe9ea19e6f0307a9eb3213301256c2f6c826cf449bab800f`。
- GDTS 对照 JSON SHA256：
  `df9aa35cd26296558c864a2bcda792a8411bbc83a530376c45d3429a768eae6c`。

## 五种子结果

| seed | minADE | minFDE | JADE | JFDE |
| ---: | ---: | ---: | ---: | ---: |
| 2035 | 0.138630 | 0.204783 | 0.168829 | 0.278901 |
| 2036 | 0.139402 | 0.200550 | 0.169781 | 0.277454 |
| 2037 | 0.137635 | 0.201341 | 0.169484 | 0.275743 |
| 2038 | 0.138304 | 0.204555 | 0.168958 | 0.279622 |
| 2039 | 0.136066 | 0.198955 | 0.165779 | 0.269852 |
| **均值** | **0.138007** | **0.202037** | **0.168566** | **0.276314** |
| 样本标准差 | 0.001258 | 0.002553 | 0.001605 | 0.003906 |

## 与绑定 GDTS 五种子对照

| 方法 | minADE | minFDE | JADE | JFDE |
| --- | ---: | ---: | ---: | ---: |
| GDTS | **0.134384** | **0.191589** | 0.208469 | 0.357055 |
| EBJD epoch 45 | 0.138007 | 0.202037 | **0.168566** | **0.276314** |
| EBJD 相对变化 | +2.70% | +5.45% | -19.14% | -22.61% |

指标越低越好。这里的 GDTS 和 EBJD 使用相同五个推理 seed 和同步 HOTEL 窗口；
方法采样器不同，因此是预算与数据条件一致的方法级比较，不是逐噪声候选配对。

## 协议边界

本结果明确是 `independent_test=false`。HOTEL validation/test manifests 各含 445
个窗口，窗口集合与顺序完全相同；epoch 45 又是通过这些镜像窗口选择的。因此
这只能作为官方 HOTEL held-out scene 上的内部多推理种子稳定性证据，不能称为
独立测试泛化结果。

正式五种子入口使用 `batch_size=1` 逐场景推理；训练内 checkpoint selection 使用
`batch_scenes=4`。由于随机张量的形状与消费次序随 batching 改变，本次 seed 2035
不会逐位复现训练日志里的 epoch 45 单 seed 数值。五种子表内部均使用同一个
正式入口，因此汇总自洽；旧训练验证数值没有混入本表。

当前只有一个训练 seed 3101。五个 inference seeds 衡量采样随机性，不能替代多个
训练 seeds，也不能用于声明训练层面的统计显著性。
