# EBJD ZARA1 正式训练快照（2026-10-23 19:08 CST）

## 当前状态

- 状态：`RUNNING`，仅一个 GPU 训练主进程（PID 46199），supervisor PID 46086。
- 正式 run：`zara1_full_wide192_rollout_off_seed3101_e250_v1`。
- 固定代码：`6f86d31f7987f2e8c2fff802e12d8eb04e8d82da`。
- 模型：FULL / Raw，hidden 192，4 heads，FFN 768，7,248,732 参数。
- 训练：seed 3101，共 250 epochs；当前已完整完成 epoch 6，正在 epoch 7。
- 快照进度：全局 update 1631；epoch 7 内 11/270 updates。
- GPU 显存：约 7,148 MiB。
- `rollout_update` 在全部 1,631 条记录中始终为 0；`projection_active` 始终为 0。
- 所有已记录 loss 均为有限值，未发现 OOM、NaN 或重复 GPU 训练进程。

旧 `zara1_full_rollout_off_seed3101_e250_v1` 是停止的 128/4/512 历史运行，
停在最后一个完整 epoch 3；本报告只汇总新的 wide192 正式运行。

## 完整 epoch 指标

| Epoch | loss | minADE | minFDE | JADE | JFDE | 完整 epoch 耗时 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.381528 | 0.249093 | 0.440292 | 0.375114 | 0.740327 | 13.83 min |
| 2 | 2.053559 | 0.256733 | 0.430023 | 0.380531 | 0.732143 | 13.33 min |
| 3 | 1.869795 | 0.247539 | 0.431639 | 0.356294 | 0.693990 | 13.29 min |
| 4 | 1.825112 | **0.235631** | **0.423452** | **0.332346** | **0.660581** | 13.35 min |
| 5 | 1.775968 | 0.259110 | 0.454887 | 0.368957 | 0.727897 | 13.25 min |
| 6 | 1.763418 | 0.245991 | 0.438647 | 0.362619 | 0.725959 | 13.52 min |

目前四项验证指标均在 epoch 4 最好，因此 `least_violation.pt` 指向 epoch 4。
epoch 5–6 相比 epoch 4 回落，属于训练早期的验证波动；尚不能据此判断最终退化。

matched-GDTS validation 的边缘门槛为 minADE 0.188109、minFDE 0.285034。
当前没有 epoch 同时满足这两项，因此 `best_eligible=null`，尚未生成 `best.pt`。
这不等同于训练失败，正式 selector 会继续检查后续 epoch。

## 耗时与执行优化

- 前 6 个完整 epoch：平均 13.428 min，样本标准差 0.219 min。
- 最新 epoch 6：13.518 min；日志 ETA 为 3,276.48 min，即约 54.61 h。
- 停止的旧 128/4/512 运行前三轮平均 19.973 min/epoch。
- 新 wide192 运行的实际 epoch 墙钟减少约 32.77%，对应约 1.487x 吞吐。

上述完整 epoch 时间包含训练、705-scene validation（20 worlds、20 DDIM
steps）和 checkpoint 保存。旧/新实际 epoch 对比同时包含架构宽度变化，不能
当作严格的等架构性能消融。严格的同一 wide192 初态短基准显示，
`prefetch_runtime_v1` 将端到端更新从 5.119 s/update 降至 2.975 s/update，
为 1.720x；四次更新的模型、优化器和 RNG 验收通过。

执行配置为 4 个有序 DataLoader workers、prefetch factor 2、
`training_runtime_optimizations=true`。训练与验证 loader 各有 4 个持久 worker，
它们是同一训练主进程的子进程，不是重复训练任务。

## Checkpoint 与证据边界

- 最新完整 checkpoint：epoch 6 / update 1620。
- `last.pt` SHA256：
  `b5121ff8d49804e3527ce7135f2ad75828f96984d553f4ed01cc1aebcf5fc005`。
- 目前结果来自一个训练 seed（3101）的逐 epoch validation，不能表述为多训练种子结果。
- 最终正式结果仍须训练至 epoch 250，经既定 selector 冻结 checkpoint，再运行
  2035–2039 五个推理 seed。
- `STATUS.json` 中残留的旧 epoch/update 和旧 `failed_reason` 是历史字段；真实进度
  以 wide192 的 `epoch_metrics.csv`、`train.jsonl` 和 checkpoint 为准，训练不受影响。

本快照只归档当时已经写盘的完整结果；训练仍在后台继续运行。
