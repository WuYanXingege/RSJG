# EBJD ETH seed 3101 正式训练与评测结果

- 状态：`COMPLETED`；训练 100 epoch，训练种子 3101。
- 正式 checkpoint：epoch 5，选择依据 `least_violation`，SHA256 `08a6f86b1dff53ce75fa52ab30befac0d7abfe1c52ecd235fe7df3d6a5d6ea55`。
- 推理：2035–2039，20 worlds × 20 DDIM steps，batch=1、FP32、无 autocast。
- validation/test 为官方镜像，因此 `independent_test=false`。

| 指标 | GDTS 均值±样本标准差 | EBJD 均值±样本标准差 | EBJD 相对变化 | 改善种子数 |
|---|---:|---:|---:|---:|
| minADE | 0.415750 ± 0.005153 | 0.587924 ± 0.010845 | +41.41% | 0/5 |
| minFDE | 0.577091 ± 0.010063 | 1.023315 ± 0.023947 | +77.32% | 0/5 |
| JADE | 0.492722 ± 0.004842 | 0.648937 ± 0.009628 | +31.70% | 0/5 |
| JFDE | 0.749165 ± 0.008598 | 1.123946 ± 0.021509 | +50.03% | 0/5 |

## 预注册问题

- 联合收益是否在 ETH 复现：**否**。
- 边际退化是否在 ETH 复现：**是**。
- 原 marginal-preserving 选模目标是否通过：**否**。

结论只覆盖当前已完成的折；ETH 仍只有一个独立训练种子，五个推理种子的离散度不能作为多训练种子显著性证据。
