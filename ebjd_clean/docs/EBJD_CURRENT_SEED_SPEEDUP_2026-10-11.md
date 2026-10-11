---
experiment_id: EBJD-GPU-261011-001
date: 2026-10-11
snapshot_time: "2026-10-11T11:59:27+08:00"
system: EBJD
experiment_type: current_seed_execution_speedup
dataset: HOTEL
training_seed: 3101
origin_execution_commit: 9ecde66aa86f7a577fa29bd94b0168ae735a1f16
validation_commit: 55c8cb9788e5d1300e891118d8b59503d224f13b
execution_commit: d3b35df23f5dc181e73b0ef74937574ac53476c3
status: adopted_epoch70_confirmed
device: NVIDIA_GeForce_RTX_5070_Ti_16GB
tags:
  - EBJD
  - HOTEL
  - CUDA
  - execution-only
  - rollout
  - checkpointing
---

# EBJD HOTEL 当前种子执行提速认证

## 结论

当前唯一正式种子 HOTEL 3101 已在完整 epoch 69 边界暂停、认证并从 epoch 70
恢复。最终采用单一执行策略 `small_group_uncheckpoint_v1`：对 padded physical
agent slots 不超过 32 的组保留全部 20 个 DDIM step 激活；超过 32 的组继续使用
原有外层 step checkpoint。N=57 和原 OOM 布局均保留安全回退。

配对完整更新基准的加权 p50 从 11.7962 s 降到 7.6406 s，即 **1.5439x**、
耗时下降 **35.23%**。BF16 与 FP32 数值认证、32-update 连续压力测试和两次严格
checkpoint 回读均通过。模型、loss、S4、20 steps、rollout 频率、三组完整 VJP、
FP64 边际投影、优化器提交和推理协议均未改变。

## 边界暂停与源 checkpoint

- 正式旧进程在 epoch 69 / update 18,354 的原子边界后停止；未开始 epoch 70。
- 源 checkpoint SHA256：
  `5ebe981d0ce3f2baf53e2240e2a254b9b0289c4c139e4fe485231d54ca133fd7`。
- 原件与只读保全副本 SHA256 相同；旧 trainer 和旧 supervisor 均已退出。
- seed 3102/3103 保持取消；UNIV waiter 未占用 GPU。

## 候选筛选

| 候选 | 直接证据 | 完整更新结果 | 决策 |
| --- | --- | --- | --- |
| A：图内复用 FP32 map features | 小场景 rollout 的转换次数 162→2，避免 160 次；峰值 allocated 不增 | 加权 p50 11.4310→11.4372 s，-0.054% | 拒绝；真实总体无收益 |
| B：≤32 slots 取消外层 step checkpoint | 20-step 梯度链、三组 VJP 和投影均保留；>32 自动回退 | 加权 p50 11.7962→7.6406 s，1.5439x | **采用** |
| C：三组 batched VJP | 保留三行独立 VJP 的实现进入真实模型探针 | cuDNN GRU 反向缺少 `aten::_cudnn_rnn_backward` batching rule，且无 fallback | 拒绝；不升级环境、不重写 GRU |

一次 mixed-rollout profiler 的事件表达到约 18 GB 主机 RSS，因此按预注册安全
条件终止。没有使用该次 profiler 作为性能证据；完整吞吐结论全部来自无 profiler
运行。A 的转换计数改用实例级计数，不保留事件表。

## B32 配对完整更新性能

每个策略完成 8 次普通更新和 4 次 rollout 更新；每次从同一 epoch 69
checkpoint 重新加载，CUDA seed、数据和噪声配对。计时包含 H2D、context、全部
loss、三组 VJP、clip、AdamW proposal、FP64 projection 和参数提交。

| 路径 | normal p50 / p90 (s) | rollout p50 / p90 (s) | 3:1 加权 p50 (s) |
| --- | ---: | ---: | ---: |
| `compact_grouped_v1` | 1.8491 / 2.0264 | 41.6374 / 41.8063 | 11.7962 |
| `small_group_uncheckpoint_v1` | 1.8505 / 1.8618 | 25.0106 / 25.1761 | 7.6406 |

候选 rollout 最大 allocated/reserved 为 10,085,382,656 / 10,255,073,280 bytes；
保守驱动显存余量 5,228,199,936 bytes。参考 rollout 最大 allocated/reserved 为
3,680,033,280 / 4,179,623,936 bytes。加速来自以显存换取小组激活保留；普通更新
没有被宣称为加速。

配对更新计时不含预先构造测试 batch 和 JSONL 写入。32-update 连续测试对真实
loader 取数与增强单独计时：均值 / p50 / p90 为
`1.0191 / 0.9866 / 1.4630 s/update`；这一部分没有因候选而改变。JSONL 写入没有
为追求微小差异再做侵入式单独计时，正式 epoch 的 41.3457 min 端到端墙钟已包含
loader、增强、每步日志、完整验证与 checkpoint。训练从边界暂停到正式恢复共
33 min 22 s，显著低于 4 h GPU 研究预算。

## 数值与梯度认证

配置 BF16 路径中，rollout on/off 的 loss、有效轨迹、total/MA/MF 梯度、AdamW
候选、投影减量、moments、更新后参数与 CUDA RNG 端点均与参考路径精确一致。

FP32 路径使用预先固定容差并通过：rollout-on total/MA/MF 梯度相对差分别为
`8.29e-6 / 6.88e-6 / 5.99e-6`；AdamW 候选和投影减量相对差为 `2.47e-6`；
更新后参数相对差为 `1.36e-9`；有效 rollout 轨迹差为零。base 与 rollout 到
trainable U-Net、denoiser、future geometry 和 map injection 的梯度连接均存在，
投影可行性通过。

## 32-update 连续压力测试

连续测试包含 32 次真实混合更新和 8 次 rollout，并显式插入原 OOM 人数布局、
真实 N=57 B4×acc4、以及 `[4,4,1]` 尾批。

| 指标 | 数值 |
| --- | ---: |
| normal p50 / p90 | 2.0973 / 2.5554 s |
| rollout p50 / p90 | 28.4298 / 74.5919 s |
| 实际 3:1 序列均值 | 9.8395 s/update |
| 最大 allocated / reserved | 10,746,273,280 / 10,896,801,792 bytes |
| 最小保守驱动余量 | 4,604,887,040 bytes |

原 OOM 布局 rollout 用时 24.3495 s，checkpoint 组比例 0.125；N=57 压力更新
用时 74.5919 s，checkpoint 组比例 1.0，证明密集回退生效；尾批用时 1.3214 s。
全部记录有限，投影后最小 ADE/FDE 点积为 `0.0012345 / 0.0016107`。

## 迁移与正式恢复

生产执行提交为 `d3b35df23f5dc181e73b0ef74937574ac53476c3`，worktree 在
迁移与启动时为 clean。迁移只允许 `source_commit`、`output_root`、
`execution_strategy` 和 `rollout_uncheckpointed_max_agent_slots` 这些明确执行字段
变化。模型、optimizer、RNG 和 update index 在初次恢复及迁移后严格二次回读中
均 exact；selector 保留 69 个候选，least-violation 仍为 epoch 45。

迁移后 checkpoint SHA256：
`3067c25e87c0a408627af366ebd0f58e574c59801fe2dc08a8d5432988f2392a`。

正式后台 trainer 已从 epoch 70 启动，首个真实 rollout 为 update 18,357：仅
1/16 physical groups 使用 step checkpoint，全部 loss 有限，投影后 ADE/FDE 点积
为 `0.0011599 / 0.0028183`。

完整 epoch 70 已于 `2026-10-11 11:57:47 +08:00` 原子保存，达到 update
18,620。由正式进程启动到 checkpoint 落盘的完整墙钟为 **41.3457 min**，其中
包含模型恢复、266 次更新、完整验证和原子保存。相对旧记录的完整 epoch 平均
57.91 min，耗时下降 **28.60%**，实际 epoch 级提速为 **1.4006x**。epoch 70
checkpoint SHA256 为
`cb689897121af8f63d50e265ab0cf3ff95501361d41799dfe9b6f7e335597291`。

epoch 70 完整验证为 minADE/minFDE `0.137318 / 0.205548`、JADE/JFDE
`0.163499 / 0.274512`；scene-weighted minADE/minFDE 为
`0.136367 / 0.211747`。这些值均为同一 epoch 70 checkpoint 的成套结果，不能与
其他 epoch 的单项最优拼接。selector 候选数由 69 增至 70，仍无 eligible
checkpoint；least-violation 仍是 epoch 45，总违反量 `0.0109564`，因此没有发生
错误选模或覆盖。

快照时正式进程 PID 1356033 已进入 epoch 71，记录到 update 18,629；非沙箱核验
显示它仍是唯一 GPU trainer，显存使用约 11.48 GB，剩余约 4.37 GB。按 epoch 70
实测墙钟外推，剩余 30 个完整 epoch 约需 **20.67 h**，预计在
`2026-10-12 08:38 +08:00` 左右完成；这是单 epoch 实测外推，不是完成承诺。

## 证据边界

- 配对基准达到超过 20% 的筛选目标；32-update 压力序列因刻意包含 N=57，均值
  相对旧 12.0093 s/update 约下降 18.1%，不能用其 p90 代表普通数据分布。
- A 确实减少转换，但没有完整更新收益，因此最终生产代码已移除 A/C 原型，只
  保留 B32。
- 本次是当前 seed 的纯执行迁移，不是新训练、不增加种子，也不改变模型或论文
  方法定义。
- 正式采用结论以完整 epoch 70 的 28.60% 实测降时为准；微基准的 35.23% 只说明
  rollout 热路径上限，不替代端到端 epoch 证据。
