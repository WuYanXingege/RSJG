# EBJD ETH/UCY 五折最佳推理 seed 补充表

## 结论先行

在每个 scene fold 已冻结的 H checkpoint 上，从推理 seed 2035–2039 中按
**H-JADE 最低**选择一个 seed，并整行报告该 seed 的 minADE、minFDE、JADE、
JFDE。五折所选行的等权平均为：

- minADE：`0.232861`
- minFDE：`0.355610`
- JADE：`0.360261`
- JFDE：`0.676047`

这是一张补充性的 best-inference-seed 表。它经过 seed 选择，存在乐观偏差；
与其他论文或 baseline 的严格主比较应继续使用五个推理 seed 的均值，而不是
此表。

## 选择规则

1. 模型、checkpoint、数据成员和 20-world/20-step 协议保持原正式五折结果不变。
2. 每折只按 H-JADE 最低选择一个推理 seed。
3. 选定 seed 后同时报告该行全部四项指标；没有为不同指标分别挑 seed。
4. GT 没有参与模型生成或候选重排；这里只是在已有完整推理结果上做事后汇总。
5. 五折 AVG 是五个已选整行的等权平均，不是重新运行推理。

## 最佳单 seed 结果

| Fold | 选中推理 seed | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|---:|
| ETH | 2036 | 0.396153 | 0.575785 | 0.464633 | 0.730578 |
| HOTEL | 2035 | 0.150064 | 0.181109 | 0.202743 | 0.307055 |
| UNIV | 2038 | 0.276638 | 0.497043 | 0.586222 | 1.273965 |
| ZARA1 | 2038 | 0.184263 | 0.275277 | 0.278114 | 0.526221 |
| ZARA2 | 2039 | 0.157188 | 0.248836 | 0.269594 | 0.542417 |
| **五折等权 AVG** | — | **0.232861** | **0.355610** | **0.360261** | **0.676047** |

## 与正式五推理 seed 主结果的关系

| 汇总口径 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| best-inference-seed 五折 AVG | 0.232861 | 0.355610 | 0.360261 | 0.676047 |
| **正式 H 五种子五折 AVG** | **0.234632** | **0.356945** | **0.362519** | **0.677804** |
| 正式 matched GDTS 五折 AVG | 0.236927 | 0.366733 | 0.393959 | 0.718617 |

best-inference-seed 相对正式 H 均值分别低约
`0.001771/0.001335/0.002258/0.001757 m`。这正是事后选择产生的乐观量级，
因此论文主表应使用正式 H 五种子均值。若其他 baseline 只提供单次结果，可将
本表作为敏感性补充，但必须同时展示正式均值并说明口径不同。

## 来源与证据边界

- 原始逐 seed 数据：
  `docs/results/ebjd_eth_ucy_unified_fivefold_20261101/PER_SEED_METRICS.csv`
- 原始文件 SHA256：
  `549a26f2d99032f5ffe89c71b49a31ddce242d7fe3b7a8a314b55cf18d5bc8a1`
- 正式五折汇总：
  `docs/results/ebjd_eth_ucy_unified_fivefold_20261101/FIVEFOLD_METRICS.csv`
- 正式汇总 SHA256：
  `b981688ff63c95c71604df0cd39b6f3e93632d35f5c4bd20d06574fb8dbc89a2`

每折仍只有一个原正式 training seed 3101；2035–2039 是同一 checkpoint 的推理
随机重复，不是五个训练 seed。正在运行的 HOTEL/ZARA2 训练 seed 复验未混入本表。
