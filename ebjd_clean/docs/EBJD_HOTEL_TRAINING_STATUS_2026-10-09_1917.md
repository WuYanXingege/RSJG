# EBJD HOTEL 训练状态快照

快照时间：2026-10-09 19:17:36（Asia/Shanghai）

执行代码提交：`9ecde66aa86f7a577fa29bd94b0168ae735a1f16`

状态：`RUNNING_STABLE_JOINT_IMPROVED_MARGINAL_GATE_NOT_MET`

## 结论

正式 seed 3101 训练稳定运行，已经完成 epoch 28，正在进行 epoch 29。训练日志连续，所有已记录数值有限，没有 OOM、NaN 或 Inf。epoch 28 是当前最小 marginal violation checkpoint。

当前结果呈现清晰但尚未满足正式门槛的权衡：相对于绑定的 GDTS seed-2035 验证基线，epoch 28 的 JADE/JFDE 分别改善 13.77%/15.67%，但 minADE/minFDE 分别退化 5.92%/8.85%。因此 `best_eligible` 仍为空，不能宣称已经实现“joint 改善且 marginal 不退化”。

## 运行身份与进度

| 项目 | 快照值 |
| --- | --- |
| 队列 | `hotel_main_speed_9ecde66_20261009_115417` |
| 当前 run | `hotel_main_seed3101_9ecde66_speed_20261009` |
| 执行策略 | `compact_grouped_v1` |
| seed 3101 | RUNNING |
| seed 3102 / 3103 | QUEUED / QUEUED，串行启动 |
| 已完成 epoch | 28 |
| 当前 epoch | 29/100 |
| epoch 29 进度 | 188/266 updates，70.68% |
| 最新 update | 7636 |
| 当前正式日志记录 | 2050 条，update 5587–7636 连续 |
| 监督器 / 子进程 PID | 1263005 / 1263016 |
| GPU | RTX 5070 Ti；进程显存约 6626 MiB |

seed 3101 从认证的 epoch-21/update-5586 边界恢复。源 checkpoint SHA256 为 `9a7a312b44aca95976f727c854b6a9fbd83891dc5cb212733355acffdee133d3`。当前最新完整 checkpoint 是 epoch 28/update 7448：

`dbc03db7990d7553021a235ae6eb9681dbd35bddde35dce3860f4383c86ca65b`

## 验证指标

比较使用与当前 selection 合同绑定的 seed 2035、445 个同步窗口、P20 × 20 推理。数值越低越好。

| 指标 | GDTS baseline | EBJD epoch 28 | 绝对变化 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| minADE | 0.135220 | 0.143224 | +0.008004 | **+5.92%（退化）** |
| minFDE | 0.193382 | 0.210496 | +0.017114 | **+8.85%（退化）** |
| JADE | 0.211186 | 0.182108 | -0.029078 | **-13.77%（改善）** |
| JFDE | 0.359854 | 0.303459 | -0.056395 | **-15.67%（改善）** |

epoch 28 的 joint score 为 `0.3338375152`，total marginal violation 为 `0.0251181615`。它是当前 `least_violation` checkpoint；`best_eligible=null`，`simultaneous_improvement=false`。

从 epoch 21 到 epoch 28，四项 EBJD 验证指标总体继续改善：minADE `0.149505 → 0.143224`、minFDE `0.224833 → 0.210496`、JADE `0.192113 → 0.182108`、JFDE `0.329299 → 0.303459`。这说明训练方向仍在进步，但尚未把 marginal 指标拉回基线以内。

## 训练数值与吞吐

| epoch | updates | mean loss | mean diffusion | mean rollout loss |
| --- | ---: | ---: | ---: | ---: |
| 27 | 266 | 1.5448 | 1.4120 | 1.8594 |
| 28 | 266 | 1.5914 | 1.4408 | 1.8769 |
| 29（partial） | 188 | 1.5733 | 1.4064 | 1.8361 |

loss 在相邻 epoch 间有正常随机波动，没有发散趋势。当前普通 update 的 p50/p90 是 1.85/2.20 秒，rollout update 的 p50/p90 是 42.40/51.24 秒。

截至快照共有 512 次 rollout update，其中 79 次触发投影。投影后 protected marginal dot 的最小值分别为 `-1.16e-11` 和 `-2.28e-11`，仅为浮点误差量级，满足 `-1e-6` 可行性门槛。最大 CUDA allocated/reserved 为 6,059,829,760 / 6,557,794,304 bytes。

训练时的 soft marginal ADE/FDE 均值 `0.4432/0.8363` 来自 S4 可微 rollout，不等同于 P20 × 20 验证 minADE/minFDE，不能直接拼接比较。

## 完整性说明

2026-10-09 12:02，默认沙箱看不到宿主 PID，曾导致对原进程状态的短暂误判并启动第二个实例。核验宿主 PID 后，第二个监督器和子进程立即终止；它没有到达 epoch 边界，也没有写入 checkpoint。其 6 条混合日志和首更新收据已隔离，SHA256 分别为：

- `a2d3c79f65d6a2bd154193462d7f6aac158308c2cb20558d6c1dad7f4a10c410`；
- `52580b1220f3c3bc4686ddacb5d5a16c9f09e8376466f3aaedf1e032c427550a`。

正式 canonical `train.jsonl` 仅保留原 PID 1263016 的轨迹，并已验证 update 5587–7636 严格连续。重复实例约 44 秒的 GPU 竞争可能影响极少量 wall-time 观测，但不共享模型/RNG 状态，也未改变正式参数轨迹。原正式进程始终未中断。

## 绑定哈希

- `SELECTION_STATUS.json`: `3a0bef898542bc319e05cc7d66e8cb3f5f6a344529d1c0c1c36ffda9fda125e8`
- `RUN_IDENTITY.json`: `5c6306f3b876b750391a0727cd8c791827c0e5abb147b7db9ce50232c9977594`
- `RESOLVED_CONFIG.json`: `a3800984c4876bec69caa62a587ab2db609d2dce1c7670995ddee0ecb6440278`
- queue manifest: `30a1d292d0d233983c12d409543d42823ca6a72a15faa91f41b2ed3aedf68463`
- first formal update receipt: `84f81a20ea396a8fa30b54102ca8fe0a90ea0c10aa4cd97bbc248dd5bd4c22e2`

## 证据边界

这些指标属于内部 development validation，且验证/测试窗口身份仍标记为 `mirrored_official_package_internal_only`。它们可用于当前同协议下的方向判断和 checkpoint selection，不能替代最终独立测试、三训练种子汇总或标准五折外部 benchmark 结论。
