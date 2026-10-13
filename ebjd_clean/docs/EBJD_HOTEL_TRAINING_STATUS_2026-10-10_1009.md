# EBJD HOTEL 训练状态快照

快照时间：2026-10-10 10:09:10（Asia/Shanghai）

执行代码提交：`9ecde66aa86f7a577fa29bd94b0168ae735a1f16`

状态：`RUNNING_STABLE_VALIDATION_VARIANCE_BEST_EPOCH35_GATE_NOT_MET`

## 结论

正式 seed 3101 已完成 epoch 44，正在进行 epoch 45。训练日志从 update 5587 到 11776 严格连续，所有已记录数值有限，没有 OOM、NaN 或 Inf。GPU 训练进程和串行监督器均正常存活。

epoch 44 的验证结果较 epoch 43 回退，因此“最新 checkpoint”不是“当前最佳 checkpoint”。在严格 marginal 不退化门槛下仍没有 eligible checkpoint；当前应继续保留 epoch 35 作为 least-violation checkpoint。epoch 35 相对绑定的 GDTS seed-2035 基线将 JADE/JFDE 改善 21.08%/22.65%，但 minADE/minFDE 仍退化 1.77%/5.64%。

## 运行身份与进度

| 项目 | 快照值 |
| --- | --- |
| 队列 | `hotel_main_speed_9ecde66_20261009_115417` |
| 当前 run | `hotel_main_seed3101_9ecde66_speed_20261009` |
| 执行策略 | `compact_grouped_v1` |
| seed 3101 | RUNNING |
| seed 3102 / 3103 | QUEUED / QUEUED，串行启动 |
| 已完成 epoch | 44 |
| 当前 epoch | 45/100 |
| epoch 45 进度 | 72/266 updates，27.07% |
| 最新 update | 11776 |
| canonical 训练记录 | 6190 条，update 5587–11776 连续 |
| 监督器 / 子进程 PID | 1263005 / 1263016 |
| GPU | RTX 5070 Ti；训练进程显存约 6626 MiB |

seed 3101 从认证的 epoch-21/update-5586 checkpoint 恢复。当前最新完整 checkpoint 为 epoch 44/update 11704，SHA256：

`ec6f90d8f81c9bee341990b1853dad27197db3a85fd533ca484a006cfe090e84`

## 最新验证与当前最佳

比较使用 selection 合同绑定的 seed 2035、445 个同步窗口、P20 × 20 推理；所有指标越低越好。

| checkpoint | minADE | minFDE | JADE | JFDE | total violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDTS baseline | 0.135220 | 0.193382 | 0.211186 | 0.359854 | — |
| EBJD epoch 44（最新） | 0.146926 | 0.230293 | 0.179736 | 0.312879 | 0.048618 |
| EBJD epoch 35（least violation） | 0.137610 | 0.204282 | 0.166669 | 0.278341 | **0.013291** |

相对 GDTS baseline：

| checkpoint | minADE | minFDE | JADE | JFDE |
| --- | ---: | ---: | ---: | ---: |
| epoch 44 | +8.66% | +19.09% | **-14.89%** | **-13.05%** |
| epoch 35 | +1.77% | +5.64% | **-21.08%** | **-22.65%** |

epoch 44 的 joint score 为 `0.3361753709`；epoch 35 为 `0.3058399567`。当前 `best_eligible=null`、`simultaneous_improvement=false`，所以还不能宣称实现“joint 改善且 marginal 不退化”。

epoch 44 的单轮回退不等于训练发散。验证采样具有方差，selection 合同会保留更优的旧 checkpoint；判断正式收益应依据冻结的 checkpoint selection 和最终多种子汇总，而不是强制使用最后一轮。

## 训练稳定性与吞吐

| epoch | updates | mean loss | mean diffusion | mean rollout loss |
| --- | ---: | ---: | ---: | ---: |
| 43 | 266 | 1.7166 | 1.3673 | 1.7258 |
| 44 | 266 | 1.7152 | 1.3714 | 1.7243 |
| 45（partial） | 72 | 1.7356 | 1.3977 | 1.6823 |

epoch 43–45 的训练均值稳定，没有随 epoch 44 验证回退而出现 loss 爆炸。普通 update 的 p50/p90 为 1.84/2.18 秒，rollout update 为 42.26/51.42 秒。

截至快照共完成 1547 次 rollout update，其中 97 次触发投影。投影后 protected marginal dot 最小值为 `-1.16e-11/-2.28e-11`，仅为浮点误差量级，满足 `-1e-6` 可行性门槛。最大 CUDA allocated/reserved 为 6,059,829,760 / 6,557,794,304 bytes。

训练 soft marginal ADE/FDE 均值 `0.4192/0.8035` 来自 S4 可微 rollout，不等同于 P20 × 20 验证 minADE/minFDE，不能直接横向拼接。

## 完整性与队列说明

`QUEUE_STATE.json` 只在队列状态迁移时更新，不是心跳文件；其中 `last_updated_at` 停留在启动时刻不表示训练停滞。宿主 PID、GPU 上下文、连续增长的 `train.jsonl` 和 epoch-44 checkpoint 共同证明队列仍在运行。

2026-10-09 12:02 的短暂重复实例已在上一份快照中完整记录。该实例未到达 epoch 边界、未写 checkpoint；其记录已隔离，canonical 参数路径仍由原 PID 1263016 唯一产生。

## 绑定哈希

- epoch-44 `last.pt`: `ec6f90d8f81c9bee341990b1853dad27197db3a85fd533ca484a006cfe090e84`
- `SELECTION_STATUS.json`: `429645688f36ab34230902df0f55ebabf6ec9185a5d583ead6100658cfbd7fa7`
- `RUN_IDENTITY.json`: `5c6306f3b876b750391a0727cd8c791827c0e5abb147b7db9ce50232c9977594`
- `RESOLVED_CONFIG.json`: `a3800984c4876bec69caa62a587ab2db609d2dce1c7670995ddee0ecb6440278`
- queue manifest: `30a1d292d0d233983c12d409543d42823ca6a72a15faa91f41b2ed3aedf68463`
- first formal update receipt: `84f81a20ea396a8fa30b54102ca8fe0a90ea0c10aa4cd97bbc248dd5bd4c22e2`

## 证据边界

这些结果属于内部 development validation，窗口身份仍标记为 `mirrored_official_package_internal_only`。它们支持当前同协议下的训练诊断与 checkpoint selection，但不能替代最终独立测试、三训练种子汇总或标准五折外部 benchmark。
