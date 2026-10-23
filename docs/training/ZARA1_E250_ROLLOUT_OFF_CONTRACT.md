# ZARA1 FULL 250 轮 rollout-OFF 正式训练合同

本合同只约束尚未启动的 ZARA1 EBJD 正式训练；GDTS goal pretrain 与
matched baseline 沿用各自已经冻结的配置，不受本变更影响。

- 训练范围：epoch 1--250，共 250 轮；不启用自动早停。
- 模型与输出：原版 FULL、Raw；CTF 和正式 20-world/20-step DDIM 推理保留。
- 训练 rollout：全程关闭（`rollout_training_enabled: false`）。
- 配对边缘 actual-step projection：全程关闭
  （`paired_marginal_actual_step_projection_enabled: false`）。
- epoch 21 及其后的 rollout 权重调度不会覆盖上述显式总开关。
- 优化器仍使用 `ActualStepAdamW` 的 AdamW proposal/apply 路径；仅关闭依赖
  rollout 边缘梯度的二半空间 FP64 投影。
- 学习率调度仍为 5 轮 warmup 后的 epoch 归一化余弦衰减。总长度改为
  250 后，余弦区间为 epoch 6--250，分母为 245；初始学习率、最低比例
  0.01 与 warmup 规则不变。实现中没有独立 `total_steps` 或 `T_max`。
- 每轮保存可恢复的 `last.pt`；第 250 轮另存 `final.pt`。正式评测仍由
  原 selector 在 `best.pt` 与 `least_violation.pt` 语义下冻结模型，不能把
  `final.pt` 自动解释为最佳模型。
- 故障恢复只允许从同一 run 的 `last.pt` 恢复模型、优化器、epoch、update、
  selector、loader 及全套 RNG 状态，并继续到总 epoch 250。

正式入口必须传入流水线生成的 `--startup-contract`。训练进程会在第一个
optimizer update 之前独立核对：代码提交、resolved config 路径及 SHA256、
semantic config hash、启动参数、250 轮、两个关闭开关以及 20/20 推理设置，
并写出 `STARTUP_GATE_RECEIPT.json`。任一项不一致即拒绝训练。
