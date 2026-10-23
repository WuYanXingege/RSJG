# ZARA1 192/4/768 rollout-OFF 执行优化

## 结论

本轮接受 `prefetch_runtime_v1`。新正式运行使用 FULL/Raw、hidden 192、
4 heads、FFN 768，共 7,248,732 个参数；rollout 与 paired actual-step
projection 始终关闭。旧 128/4/512 运行完整保留在 epoch 3，因参数形状
不同，不作为新运行的恢复点。

优化只改变执行：训练 DataLoader 使用 4 workers、prefetch factor 2，
并启用已验收的 rollout-OFF 运行时路径，避免分配未使用的两组约束梯度
缓冲及逐组标量同步。样本顺序、增强、扩散噪声、逻辑 batch、损失、
优化器和验证协议不变。

## 真实短基准

相同 fresh 192/4/768 完整初态，真实 ZARA1 数据，1 次预热后连续 4 个
optimizer updates；计时不含 profiler。

| 路径 | 数据准备 s/update | GPU update s/update | 端到端 s/update | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|
| workers=0 / runtime OFF | 2.9321 | 2.1869 | 5.1190 | 3.348 GB | 7.896 GB |
| workers=4 / runtime ON | 0.9841 | 1.9912 | 2.9753 | 3.232 GB | 7.774 GB |

端到端提速 **1.720x**（耗时降低 41.9%）；数据准备提速 2.979x，GPU
update 提速 1.098x。主要剩余计算热点是 GCJD 去噪及 checkpoint 重算，
本轮不继续改变 activation-checkpoint 策略。

数值门槛固定为 atol=rtol=2e-5。四个更新的 batch SHA256、rollout
日程和 CPU/CUDA/loader RNG 一致；307 个模型状态张量与 594 个优化器
状态张量最大差均为 0。逐更新最大 loss 差为 1.01e-7。

完整 705-scene validation（20 worlds、20 DDIM steps、seed 2035）耗时
从 62.175 s 降至 58.202 s（1.068x），两路所有原始聚合量和四项指标
逐值相同。此 validation 使用未训练的临时模型，只验证执行等价性，
不是研究结果。

## 证据与边界

- 机器结果：`results/zara1_wide_prefetch_v1_20261023/BENCHMARK.json`
- 验证结果：`results/zara1_wide_prefetch_v1_20261023/VALIDATION_BENCHMARK.json`
- 复现入口：`scripts/benchmark_zara1_off_training.py` 与
  `scripts/benchmark_zara1_validation.py`
- 成功基准 GPU 墙钟约 234 s；含两次基准夹具失败调用约 273 s。

该结果认证 ZARA1 rollout-OFF 的当前硬件与配置，不外推为 rollout-ON
或 UNIV 的同倍提速。没有修改正式 checkpoint，也没有把短测更新用于
续训或选模。
