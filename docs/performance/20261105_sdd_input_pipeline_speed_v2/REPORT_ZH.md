# SDD matched-GDTS 输入管线优化复测

## 结论

本轮没有稳定达到采用门槛的执行优化，正式训练恢复为原 FP32、2-worker、外部 zstd 解压和原增强路径，状态记为 `NO_ACCEPTED_SPEEDUP`。两个候选均通过短数值闸门，但速度收益不稳定，因此没有进入正式默认配置。

当前慢点仍是输入供给：原始路径两次 100-update 复测平均为 1.0090 s/update，其中 DataLoader 阻塞平均为 0.7213 s/update。GPU 计算和其余训练消费约 0.288 s/update。

## 2544 次 update 的含义

旧 GDTS independent cache 含 2544 个 `train_batch_*.pkl.zst` 文件；每个文件已是一个预组装物理 batch，典型有效槽位为 64，外层 `DataLoader(batch_size=1)` 每次取一个文件。因此每轮是 2544 次 optimizer update，名义槽位总数为 162816。

JMM 同步 manifest 的统计口径不同：30 个训练序列、5720 个同步窗口和 8940 次有效行人出现。对 manifest shard 的轻量身份核对得到 30 个序列、5720 个 `(sequence,start_frame)` 以及 8940 个 manifest 行人身份。二者分别是旧 GDTS 合法单轨迹滑窗展开与同步场景 manifest，不能用 162816 对 8940 的差异推断重复导出，也没有据此删减样本。

## 候选与速度

每条路径均使用同一 epoch 8 完整 checkpoint，正式增强开启，先预热 10 次，再测量 100 次 update；配对重复两次。

| 路径 | rep1 s/update | rep2 s/update | 均值 | 配对提速 | 结论 |
|---|---:|---:|---:|---:|---|
| legacy | 1.00715 | 1.01086 | 1.00901 | 1.000× | 正式保留 |
| 预分配 replay 输出 | 1.00342 | 0.89596 | 0.94969 | 1.004× / 1.128× | 波动过大，拒绝 |
| 进程内流式 zstd＋预分配 replay | 0.98624 | 1.01461 | 1.00043 | 1.021× / 0.996× | 不稳定且均值仅 1.009×，拒绝 |

候选主要减少高通道热图变换的中间 `concatenate`，并测试了 worker 内进程内流式 zstd；但第二项没有稳定降低数据等待。未继续扫描 worker 数或新增缓存。

代表性峰值 CUDA allocated 约 6.51 GiB，reserved 在约 12.81–13.93 GiB 间随缓存池波动，候选未造成新的 OOM。主进程峰值 RSS 约 2.16–2.17 GiB；worker 子进程峰值记录约 2.22–2.55 GiB。

## 正确性

增强实现使用同一 replay 覆盖翻转、转置、90°旋转、透视、仿射、平移和任意角旋转；预分配输出与旧 `concatenate` 路径逐值一致。真实 zstd 缓存的进程内解压结果与旧 CLI 路径逐值一致。

相同 epoch 8 状态、增强关闭的 8-update 更新闸门中：

- 预分配候选总 loss 相对差 `4.43e-6`，梯度范数最大相对差 `2.15e-4`；
- 组合候选总 loss 相对差 `4.34e-7`，梯度范数最大相对差 `7.33e-5`；
- 均低于预设的 loss 0.1% 和梯度范数 1% 门槛，参数有限且发生更新。

正确性通过不等于速度采用；本轮拒绝原因仅为端到端收益不稳定。

## 暂停、恢复和资源

用户授权立即停止时，最后完整 checkpoint 为 epoch 8，SHA256 为 `6bbf73973b31822685fb2e7f4ba4c6927c95438394710162e8a3ede1622f5973`。epoch 9 停在 1137/2544，其未保存更新明确作废；恢复从 epoch 9 开始，模型、optimizer、scheduler、epoch 8 RNG 与 patience=3 合同均从完整断点恢复。

本轮 GPU 开发调用约 13.5 分钟（含初始化、一次失败接口调用、两轮配对复测和数值闸门），未重训 Goal-pretrain。正式训练仍由单一后台 supervisor 串行衔接 matched-GDTS、baseline 评测、proposal banks、EBJD 训练与最终评测。

手动恢复入口为 `scripts/launch_sdd_pipeline_background.sh <SOURCE_COMMIT>`；脚本会拒绝错误 HEAD、重复 supervisor/trainer、不可见 GPU 或缺失完整断点。
