# EBJD 正式 FULL 训练速度审查（2026-10-22）

## 结论

本轮状态为 **NO_DEFAULT_CHANGE_ACCEPTED**。正式默认配置继续使用
`compact_grouped_v1`，网络、损失、随机过程、rollout、actual-step 投影优化器、
验证与保存协议均未改变。实测候选没有同时通过固定数值门槛和混合路径性能门槛，
因此没有为追求速度而把它写入正式训练默认值，也没有启动正式训练。

代码保留了一个默认关闭的 `paper_training_v1` 执行候选，用于复现“OFF 更新不分配
两组无用的模型大小约束梯度缓冲，并将多个日志标量同步合并为一次”的安全方向。
它不改变 state_dict，也不改变网络或目标。关闭地图 U-Net activation checkpoint 和
valid-agent pack 两项未通过候选已经从最终实现及配置中移除。

## 固定环境与基准

- 源 checkpoint：HOTEL epoch 45 / update 11970。
- SHA256：`425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1`。
- Python 3.10.21，PyTorch 2.8.0+cu128，CUDA 12.8。
- GPU：NVIDIA GeForce RTX 5070 Ti，16,303 MiB。
- BF16 保持原策略；batch 4、梯度累积 4；epoch 46 调度。
- 正式计时：真实 HOTEL 数据，相同完整 checkpoint/优化器/RNG/loader generator，
  1 个 warm-up update 不计时，随后连续 16 个 optimizer update，其中 4 个 rollout ON。
- 每个计时区间在边界同步 CUDA；batch 内容以完整 SHA256 成对核对；审计哈希耗时排除。

## 正式 16-update 结果（候选已拒绝）

该候选同时启用了运行时同步优化和“地图 U-Net 不做 activation checkpoint”。后者在
短 gate 中未显现问题，但在 16 次连续更新中产生累积数值漂移，故整项拒绝。

| 路径 | baseline (s/update) | candidate (s/update) | 倍数 |
|---|---:|---:|---:|
| rollout OFF | 1.758489 | 1.517295 | 1.15896× |
| rollout ON | 40.340641 | 40.623500 | 0.99304× |
| 1/4 rollout 混合 | 11.404027 | 11.293846 | 1.00976× |
| 含数据准备端到端 | 12.150752 | 12.058726 | 1.00763× |

数据准备均值为 baseline 0.746725 s/update、candidate 0.764880 s/update。峰值 allocated
显存从 5,085,664,768 bytes 增至 8,254,147,584 bytes；峰值 reserved 从
9,493,807,104 bytes 增至 10,582,228,992 bytes。它既未达到 1.30× 目标，也没有降低
显存。

16 次更新使用相同 batch、相同 rollout 调度，CPU/CUDA/loader RNG 最终状态逐字节一致；
但模型参数最大绝对差为 `6.876185e-4`，优化器状态最大绝对差为 `1.996286e-3`，超过
预先固定的 `atol=rtol=2e-5`。因此不能归入正式实现。完整逐 update 收据见
`results/training_performance_20261022/REJECTED_UNCHECKPOINTED_MAP_BENCHMARK.json`。

## 保留候选及实际瓶颈

仅保留的默认关闭候选在 4-update（含一个完整 rollout 周期）定向 gate 中通过模型、
优化器、RNG 和 loss 检查；其 OFF/ON/混合倍数分别为约 1.147×、0.995×、1.011×。
这说明减少 OFF 路径同步确有局部收益，但不能改善占主导的 rollout ON 路径，因而不具备
明确的端到端收益，不默认启用，也不据此声称训练已加速。

从正式窗口可直接量化：baseline 的 rollout ON 比 OFF 多 38.582 s/update；按原 1/4
频率，它支配混合耗时。数据准备约 0.75 s/update，不是首要瓶颈。高开销 profiler
尝试在写收据前异常退出，其结果没有用于任何提速主张；本轮未以局部算子数字替代真实
optimizer update。

后续若要达到 1.30×，需要针对 differentiable 20-step rollout 内的反向重算和三组
约束梯度计算做保持数值等价的实现，而不是继续优化 OFF 日志或取消必要 checkpoint。
本轮没有自动开展该研究。

## 正确性与兼容性

- `python -m pytest -q`：30 passed。
- 默认 `configs/training.yaml` 未启用候选，保持归档 FULL 行为。
- retained candidate 不新增参数；checkpoint 仍 strict-load，state_dict 键与形状不变。
- 已拒绝实现没有留在模型、配置或默认训练入口中。
- 所有基准只操作临时模型状态，没有覆盖 checkpoint、训练日志或选模结果。

## GPU 成本与边界

正式可审计 16-update 窗口耗时 438.732 GPU 墙钟秒。此前定向 gate 也计入本轮 1800 秒
上限；一个带 profiler 的调用在写收据前异常结束，因此无法给出伪精确的总和。确认总过程
未继续启动 GPU 工作，并在预算边界停止。报告只把有原子收据的 438.732 秒作为正式性能
数字，不把失败调用遗漏为“零成本”。

## 复现

以下命令重新运行 paired benchmark；它每完成一臂即原子写入 progress 收据，最终结果
不能用于续训或选模：

```bash
python -u scripts/benchmark_training_speed.py \
  --config configs/resolved/hotel.json \
  --checkpoint assets/checkpoints/hotel/epoch_045_full.pt \
  --updates 16 --warmup-updates 1 --epoch 46 --device cuda \
  --output results/training_performance_20261022/BENCHMARK.json
```

若显式试验保留候选，必须把 resolved config 的 execution-only 字段设置为
`execution_strategy=paper_training_v1`、`training_runtime_optimizations=true`、
`rollout_uncheckpointed_max_agent_slots=0`；这不是正式默认配置。
