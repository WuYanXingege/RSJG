# EBJD rollout backward 训练提速验收（2026-10-22）

## 结论

状态：**ACCEPTED_SPEEDUP**。

正式默认执行配置改为 `small_group_uncheckpoint_v1`：一个物理组的 padded
`B×N≤32` 时，20 步可微 DDIM rollout 保留完整计算图，让 marginal ADE、marginal
FDE 与 total 三次 VJP 复用同一图；`B×N>32` 时继续使用 DDIM-step activation
checkpoint。地图 U-Net checkpoint、20 个时间步、4 worlds、三组梯度、actual-step
AdamW、FP64 投影和所有训练协议保持不变。

本轮没有启动正式训练，也没有修改任何正式 checkpoint。

## 重复性基线

同一 HOTEL epoch45 完整 checkpoint、优化器、RNG、loader 状态和真实 batch 序列下，
baseline A/B 各连续运行 16 个 optimizer updates（4 次 rollout ON）。固定
`atol=rtol=2e-5`，结果如下：

- loss 与约束日志最大差均为 0；
- 模型参数最大绝对差 `4.47035e-8`；
- 优化器状态最大绝对差 `1.52597e-6`；
- CPU、CUDA、loader RNG 完全一致。

因此 gate 能区分真实实现偏差与正常运行波动，上一轮未通过结果不能归因于 gate 自身。

## ON update 热点

使用 CUDA Event 只记录已有命名区间；表内为 9 个物理图的累计 inclusive 时间，计时
调用本身不用于速度结论。

| 区间 | 调用数 | CUDA累计时间 |
|---|---:|---:|
| total backward | 9 | 11.610 s |
| marginal ADE backward | 9 | 11.155 s |
| marginal FDE backward | 9 | 11.145 s |
| rollout forward | 9 | 4.883 s |
| FP64 projection | 1 | 0.570 s |
| base diffusion/loss | 9 | 0.289 s |
| context encoder | 9 | 0.145 s |
| AdamW proposal | 1 | 0.060 s |
| optimizer apply | 1 | 0.003 s |

原路径有 `9×20=180` 次 DDIM-step checkpoint。rollout 明确传入
`checkpoint_blocks=false`，所以不存在 DDIM-step 与 denoiser-block 的嵌套 checkpoint；
记录到的 54 次 block checkpoint 属于同一 update 的常规 diffusion 分支。三次 backward
各自约 11 秒并分别重算 rollout，是主要瓶颈；投影和 optimizer 不是优先优化对象。

一次 Kineto 调用虽关闭 memory/shape/stack，仍在事件解析阶段超时，已主动中止并通过
`finally` 记录 342.187 秒成本。它没有产生或上传 trace，也没有进入速度数字。随后改用
仅 11 个命名边界的 CUDA Event，正常 ON update 增量成本为 65.975 秒（含加载与两个
边界对齐 OFF updates）。

## HOTEL 正式 16-update 结果

计时窗口：相同真实数据、checkpoint、完整状态和 batch 哈希；1 个 warm-up update 排除，
随后 16 个 updates，其中 4 个 rollout ON。CUDA 仅在窗口边界同步。

| 路径 | baseline (s/update) | optimized (s/update) | 提速 |
|---|---:|---:|---:|
| rollout OFF | 1.811243 | 1.793210 | 1.0101× |
| rollout ON | 42.226339 | 27.146795 | 1.5555× |
| 1/4 rollout 混合 | 11.915017 | 8.131606 | 1.4653× |
| 含数据准备端到端 | 12.696035 | 8.915829 | 1.4240× |

optimized 的 4 个 ON updates 中，保留 step checkpoint 的物理组比例分别为
`0.1250、0.2708、0、0.4167`；较小组复用完整图，较大组保持 bounded-memory fallback。

数值验收逐值通过：16 次更新的所有 loss 完全一致，模型参数和优化器状态最大差均为 0，
RNG 完全一致；约束日志最大差仅 `7.93e-12`。state_dict、参数名称/形状和 checkpoint
格式没有变化。

## 显存与 UNIV fallback

HOTEL 峰值 allocated 从 5,085,664,768 增至 10,163,870,208 bytes；峰值 reserved 从
9,493,807,104 增至 15,357,444,096 bytes。速度收益以显存换取，因此阈值固定为 32，
未扩大到 dense groups。

额外使用 UNIV epoch73 checkpoint 和 16/15/14/13 人的真实场景完成一个临时 rollout
update。最大物理组 59 slots，所有组均走 checkpoint fallback；结果 finite，峰值
allocated/reserved 为 4,087,243,264 / 5,951,717,376 bytes，完整 update 19.189 秒。
这项检查只证明执行与显存安全，不是正式训练或性能评测。

## 实际修改

1. `configs/training.yaml` 默认启用经验证的
   `small_group_uncheckpoint_v1` 和固定阈值 32。
2. 保留原 `rollout_step_checkpoint_policy` 的 dense fallback，不关闭整体 checkpoint。
3. 增加 baseline 重复性、候选配对、命名区间计时、UNIV dense update 的小型可复现入口；
   所有入口只更新临时模型，并在 `finally` 写 GPU 成本。
4. 增加默认配置定向测试；没有保留失败复杂候选，也没有尝试第二候选，因为第一候选已
   通过数值门槛并超过 1.30× 目标。

## 复现命令

```bash
python -u scripts/check_baseline_repeatability.py \
  --config configs/resolved/hotel.json \
  --checkpoint assets/checkpoints/hotel/epoch_045_full.pt \
  --updates 16 --warmup-updates 1 --epoch 46 --device cuda \
  --output results/rollout_backward_v2_20261022/BASELINE_REPEATABILITY.json \
  --task-state results/rollout_backward_v2_20261022/TASK_STATE.json

python -u scripts/benchmark_rollout_candidate.py \
  --config configs/resolved/hotel.json \
  --checkpoint assets/checkpoints/hotel/epoch_045_full.pt \
  --candidate small-group --updates 16 --warmup-updates 1 \
  --epoch 46 --device cuda \
  --output results/rollout_backward_v2_20261022/SMALL_GROUP_FORMAL16.json \
  --task-state results/rollout_backward_v2_20261022/TASK_STATE.json
```

## GPU 成本与边界

全部开发调用累计 `1384.419 / 1800` GPU 墙钟秒，剩余 `415.581` 秒。累计值包含成功
gate、正式窗口、UNIV 检查以及被中止的 profiler，不把失败调用记为零。完成后未启动
其他 GPU 任务或正式训练。

原始小型收据位于 `results/rollout_backward_v2_20261022/`。五推理种子研究评测、验证、
选模和正式 checkpoint 均不在本轮范围内。
