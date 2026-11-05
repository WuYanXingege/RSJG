# SDD matched GDTS 执行效率与早停修正

## 结论

- 正式执行路径保持原 FP32、`num_workers=2`。两个候选均未稳定通过，因此没有用不稳定的短测收益修改正式训练语义或资源配置。
- GDTS 早停已改为每 10 epoch 验证一次、`patience_validations=3`；`max_epochs=250`、最短/最早停止 epoch 50、`valid_ADE`（pixel）、0.1% significant improvement、Adam 和 ExponentialLR 0.995 均保持不变。
- checkpoint 中的早停合同若与当前配置不同，会从已有 `log_curve.txt` 的完整验证历史重建，而不会重置模型、optimizer、scheduler、epoch 或 RNG。

## 2544 个 batch 的实际含义

训练缓存包含 2544 个 `train_batch_*.pkl.zst` 文件。每个文件已经是由旧 GDTS 预处理器组成的物理 batch，典型张量为 `[T=20, B=64, 2]`；外层 PyTorch DataLoader 使用 `batch_size=1` 读取一个预组装文件。因此一个 epoch 是 2544 次 optimizer update，对应 162816 个轨迹样本槽位。它不是梯度累积，也不是把一个 optimizer update 再拆成多个物理组。

## 轻量测量

现场表现为在线 zstd 读取、Albumentations 增强和高通道轨迹热图变换造成显著 DataLoader 等待；GPU 计算与输入准备交替。带分解的 20-update 窗口中，2-worker 数据等待总计 11.58 s（中位数 0.475 s），其余训练消费 5.56 s。主要瓶颈是 CPU 数据准备与供给，而非显存容量。

| 路径 | 第一次 s/update | 第二次 s/update | 稳定性复测 s/update | 结论 |
|---|---:|---:|---:|---|
| 原始 2 workers | 1.0437 | 0.9142 | 0.8587 | 保留 |
| 4 workers | 0.9228 | 0.7369 | 0.9576 | 长尾不稳定，拒绝 |

前两次 4-worker 配对窗口分别为 1.131× 和 1.241×，但等待分解复测为 0.897×，不满足“稳定提速”条件。另一候选“合并 loss/梯度有限性同步”首轮为 1.041×，第二轮反而变慢，已完全撤回，没有进入正式代码。

峰值 CUDA allocated 为 6,960,819,200 bytes（约 6.48 GiB），最大 reserved 为 13,262,389,248 bytes（约 12.35 GiB）；候选没有改变代表性 GPU 峰值。

## 正确性短测

为隔离 worker 数量对随机增强分配的影响，使用同一初始化、同一批次顺序并仅在 8-update 接口检查中关闭增强；正式训练仍开启增强。2/4-worker 的总 loss 分别为 0.89878947/0.89878449，相对差 0.000555%；8 步全局梯度范数最大相对差约 0.01394%，均低于预设 0.1%/1% 阈值。两侧均有 94 个参数张量获得梯度，参数有限且发生更新，无 OOM。

## 暂停与恢复边界

用户要求立即停止时，matched GDTS 正处于 epoch 1 的 1733/2544，尚未生成 `last_model.pt`。这 1733 个未形成完整 epoch checkpoint 的更新已明确作废；Goal-pretrain best epoch 35 及其 SHA256 `6f000e5c6f1642dd959107d1908e8b041706f09db1a135bc4f6a43d668145a88` 保持不变。正式流水线从 matched GDTS epoch 1 重新开始，不重新训练 Goal-pretrain。

本轮短测 GPU 墙钟约 255 秒。原 epoch 等待与正式恢复后的训练不计入该优化短测成本。
