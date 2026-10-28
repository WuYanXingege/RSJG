# ZARA2 正式跨折复验进度快照（2026-10-28 19:51 CST）

> 本文是运行中快照，不是最终结果报告。不得据此选择模型、宣称合同通过，
> 或替代流水线完成后的五种子 G/H/GDTS 正式汇总。

## 1. 固定协议与运行状态

- 分支：`train/ebjd-zara2-crossfold-v1`
- 训练源码固定提交：`d83c3be524a3eecbbf297ec15bb528611d06094d`
- 方法：可靠原生终点先验 + MPJC + GT 一致条件联合桥
- 训练 seed：`3101`；正式推理 seed：`2035–2039`
- ZARA2 留出窗口：998 scenes / 5,910 agent occurrences，平均 `5.921844`
- 未裁剪有效行人，未拆分或重采样同步窗口
- 当前服务：`active/running`，systemd user linger 已启用，重启次数为 0
- GPU：单一正式进程；快照时显存约 12,640 MiB
- 流水线状态：`RUNNING / reference_base_epoch96`，无失败信息

## 2. 已完成资产阶段

### Goal pretrain

- 完成 150/150 epochs；按 validation goal BCE 选中 epoch 30
- 绑定权重 SHA256：
  `50175952e2a5dca2984320b3d7751dcf851e09264b324fcf50a511f423a014db`

### Matched GDTS

- 完成 250/250 epochs；原生 validation ADE selector 选中 epoch 140
- 绑定权重 SHA256：
  `bba3e538775e57fac45fd693ae6308c84fd92d6bf8d5451ab337005b292b2d80`

同步场景 cache 与 validation manifest parity 已通过：998/998 窗口匹配，
5,910 名有效行人全部覆盖，最大世界坐标误差约 `4.77e-7 m`。

Matched GDTS 五个随机推理 seed 的正式结果如下（一个训练 seed）：

| 指标 | 均值 | 样本标准差（ddof=1） |
|---|---:|---:|
| minADE | 0.158421 | 0.000305 |
| minFDE | 0.257847 | 0.001030 |
| JADE | 0.295602 | 0.001261 |
| JFDE | 0.582360 | 0.002315 |

## 3. 当前 EBJD reference base

固定设置：192 hidden、4 heads、FFN 768、6 个 FJAB、FULL/Raw、legacy CTF；
rollout 与 paired actual-step projection 全程关闭。EBJD 主体从随机初始化开始，
只加载本折 goal U-Net。学习率调度保持 250-epoch 时间轴，本阶段固定在 epoch 96 停止。

快照时最近完整 checkpoint 为 epoch 6 / update 1506，epoch 7 正在运行。

| Epoch | train loss | minADE | minFDE | JADE | JFDE | 完整 epoch 分钟 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.405525 | 0.242882 | 0.427930 | 0.444442 | 0.898010 | 12.33 |
| 2 | 2.167499 | 0.214892 | 0.377382 | 0.346084 | 0.692687 | 12.24 |
| 3 | 2.007329 | 0.211568 | **0.363231** | 0.351651 | 0.698923 | 12.28 |
| 4 | 1.907068 | 0.213251 | 0.382856 | 0.352852 | 0.720784 | 12.29 |
| 5 | 1.884979 | 0.238436 | 0.388366 | 0.447642 | 0.906546 | 12.33 |
| 6 | 1.800610 | **0.207151** | 0.369487 | **0.334233** | **0.680278** | 12.36 |

早期曲线总体在学习，但仍明显弱于 matched GDTS。只有六个完整 epoch，
不能据此判断 epoch 96、后续 Candidate 或条件桥的正式表现。
按约 12.3 分钟/epoch 估计，reference base 从该快照到 epoch 96 尚需约 18–19 小时。
控制台显示的 `/250` 和约 50 小时 ETA 来自调度器总时间轴，不是本阶段实际停止点。

## 4. 后续冻结流程

流水线将自动串行执行：reference base epoch 96 → native proposal bank →
Stage A（3 epochs）→ Candidate（3 epochs）→ MPJC banks →
GT 一致条件桥（3 epochs）→ G/H 五种子阶段评测 →
最多 30 epochs 的有界续训 → 最终 G/H 五种子评测与报告。

有界续训才启用预注册早停：连续 8 轮没有满足合同且 JADE 至少改善
`0.0005 m` 时停止。当前 reference base 不使用早停。

## 5. 证据边界

- 当前数据仅来自一个训练 seed 与若干随机推理 seed，不是多训练 seed 统计。
- reference base 指标来自训练中的 validation，不是最终冻结五种子结果。
- 不应把当前曲线与 matched GDTS 的正式五种子表直接拼成最终方法结论。
- 流水线尚未完成 Candidate、MPJC、GT bridge、有界续训和最终 G/H 评测。
- 未启动任何其他 fold。
