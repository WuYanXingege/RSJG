---
experiment_id: EBJD-GPU-261010-001
date: 2026-10-10
snapshot_time: "15:36:19+08:00"
system: EBJD
experiment_type: training_speed_audit
dataset: HOTEL
training_seed: 3101
run_id: hotel_main_seed3101_9ecde66_speed_20261009
execution_commit: 9ecde66aa86f7a577fa29bd94b0168ae735a1f16
status: running_expected_slow
device: NVIDIA_GeForce_RTX_5070_Ti_16GB
raw_log: outputs/formal_hotel_speed_9ecde66_20261009/hotel/3101/none/hotel_main_seed3101_9ecde66_speed_20261009/train.jsonl
tags:
  - EBJD
  - HOTEL
  - CUDA
  - speed-audit
  - rollout
---

# EBJD HOTEL 训练速度审计

## 结论

当前训练的绝对速度较慢，但没有卡死、退化失控或偏离已认证性能。最近完整 checkpoint 为 epoch 49，训练正在 epoch 50；实测完整 epoch 墙钟均值为 **57.91 分钟**，与速度优化报告给出的 update-only 约 `0.96 h/epoch` 预测一致。

真正瓶颈不是验证、日志写入或显存交换，而是每四个 optimizer update 执行一次的 `S=4 × 20-step` 可微 rollout。rollout 更新占完整更新数的约 25%，却占记录训练计算时间的 **88.46%**。在不改变目标、rollout 频率、world 数或 step 数的前提下，没有可安全即时启用的数量级加速项。

## 实时状态

| 项目 | 数值 |
| --- | ---: |
| 最近完整 checkpoint | epoch 49 / update 13034 |
| 当前训练位置 | epoch 50 / update 13212（178/266） |
| 已完成正式 epoch | 28（22–49） |
| 日志有效更新 | 7,630 |
| 非有限数或损坏 JSON | 0 |
| 队列 | HOTEL seed 3101 RUNNING；UNIV seed 3101 WAITING |
| HOTEL GPU 进程 | PID 1263016 |
| 预计剩余时间 | 48.57 小时 |
| 预计 HOTEL 完成 | 2026-10-12 16:11 +08:00 |

ETA 按截至 epoch 49 的实际墙钟均值和 epoch 50 已完成比例线性外推；系统重启、OOM、验证波动或人工暂停均会改变 ETA。

## 墙钟时间分解

从 2026-10-09 11:56:43 启动到 epoch 49 边界：

| 分量 | 累计 | 每个完整 epoch |
| --- | ---: | ---: |
| 总墙钟 | 27.02 h | 57.91 min |
| 已记录 update 计算 | 24.85 h | 53.24 min |
| 验证、数据边界、checkpoint 等其余开销 | 2.18 h | 4.66 min |

因此即使完全消除验证与 checkpoint 开销，理论上也只能节省约 8%；主要优化对象必须是 rollout 路径。

## 更新级耗时

| 更新类型 | 数量 | 均值 | 中位数 | 训练计算时间占比 |
| --- | ---: | ---: | ---: | ---: |
| 普通更新 | 5,586 | 1.848 s | 1.849 s | 11.54% |
| rollout 更新 | 1,862 | 42.492 s | 42.435 s | 88.46% |
| 全部更新 | 7,448 | 12.009 s | — | 100% |

最近五个完整 epoch（45–49）的训练计算均值为 55.77 分钟/epoch，比 epoch 22–40 高 5.68%。这是一项轻微但有限的波动：显存峰值没有增加、所有数值有限、GPU 温度和频率正常，尚无持续恶化或资源泄漏证据。

此前认证的 48-update 连续测试均值为 12.992 s/update；当前全程均值 12.009 s/update，反而快约 7.6%。相对优化前固定配比基准的 25.064 s/update，当前实际均值约快 2.09 倍。不同数据批次不能作为严格逐样本配对，但足以排除“优化失效”。

## GPU 与 CPU 观察

30 秒、1 Hz 宿主机采样：

- GPU SM 利用率：11–42%，均值约 24.1%；
- 显存控制器利用率：1–17%；
- framebuffer：约 7.39 GB；训练进程约 6.63 GB；
- 温度：49–53 °C；功耗：61–81 W；核心时钟约 2.8–2.92 GHz；
- 进程 CPU：约 173%，33 个线程；
- 训练记录峰值 allocated/reserved：5.644/6.107 GiB。

没有温度降频、显存逼近上限或第二个 GPU 计算进程。较低 SM 利用率与当前工作负载一致：模型仅约 3.6M 参数，场景分组较小，20 个 DDIM step 串行执行，并包含三次 VJP、activation checkpointing、Python 控制与小矩阵 kernel。它说明仍有工程优化空间，但不说明进程空转。

## 指标与稳定性边界

epoch 49 验证为：minADE `0.143628`、minFDE `0.220318`、JADE `0.171455`、JFDE `0.289634`。epoch 45 仍是 least-violation checkpoint。速度审计期间 7,630 条训练记录全部是合法 JSON，所有浮点字段有限，未发现 NaN、Inf 或 OOM。

## 是否应立即改速

不建议中途改变当前 HOTEL 正式运行。下列直接提速手段都会改变训练或选模协议：降低 rollout 频率、worlds、DDIM steps，减少验证频率，改变 scene grouping，或启用未经认证的编译/融合路径。它们需要在独立分支完成数值等价、恢复一致性、显存和 checkpoint 身份验收后，才能用于新的 UNIV 或后续折；不能无记录地切换当前 run。

当前建议是让 HOTEL seed 3101 按既定协议完成。UNIV 保持等待，避免两个训练器在单张 16 GB GPU 上竞争并在 P20/20-step 验证阶段触发 OOM。
