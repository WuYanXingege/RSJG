# SDD JMM Goal-pretrain 训练状态（2026-11-05）

## 结论摘要

截至北京时间 2026-11-05 09:56，SDD JMM 流水线处于
`GOAL_PRETRAIN / RUNNING`。训练已完整完成 epoch 1--80，正在执行
epoch 81/150；后台 supervisor 与唯一训练子进程均存活。本报告是运行中快照，
不是完成报告，也不代表 matched GDTS 或 EBJD 主模型已经启动。

当前训练未启用早停（`early_stopping_patience=0`），计划运行满 150 epochs，
但会持续保存验证集 `goal_BCE` 最佳权重。当前最佳权重来自 epoch 35。

## 固定身份

- 实验：`sdd_jmm_unified_coupling_bridge_v1_20261103_155713`
- 数据协议：JMM TrajNet-SDD，train 5720 windows，validation/test 各 1999 windows
- 训练 seed：3101
- 执行源码提交：`8f9e101c917e769097be3ccf5e459530aa0276bf`
- Goal-pretrain 计划：150 epochs，Adam，初始学习率 0.001，ExponentialLR (`gamma=0.99`)
- 训练与验证：每轮均验证；每 10 轮保存编号 checkpoint；每轮保存可恢复 last
- 早停：关闭

缓存身份与坐标审计已经通过：train 5720 scenes、validation 1999 scenes，
两者的最大世界坐标绝对误差均为 `7.62939453125e-06 m`。

## 当前曲线

| epoch | learning rate | train goal BCE | valid goal BCE | 说明 |
|---:|---:|---:|---:|---|
| 1 | 0.0009900000 | 0.0065592033 | 0.0037091153 | 首轮 |
| 7 | 0.0009320653 | 0.0028025220 | 0.0030180669 | 早期快速改善 |
| 22 | 0.0008016306 | 0.0025228194 | 0.0028051823 | 接近当前最佳 |
| 35 | 0.0007034477 | 0.0024400066 | **0.0028039473** | 当前验证最佳 |
| 50 | 0.0006050061 | 0.0023765354 | 0.0028458155 | 验证未刷新 |
| 80 | 0.0004475232 | 0.0022698339 | 0.0029385426 | 最新完整轮 |

训练 loss 在 epoch 35--80 继续下降约 7.0%，但 epoch 80 验证 BCE 比当前最佳值高约
4.8%。因此现阶段呈现“训练目标继续改善、验证目标未刷新”的趋势；这支持持续保留
best checkpoint，但仅凭单次训练曲线不能作统计显著的过拟合结论，也不在运行途中
改变既定训练合同。

## 耗时与预计进度

- 已完成：80/150 epochs（53.3%）
- 最近 20 个完整 epoch：平均 16.84 分钟/轮，中位数 16.81 分钟/轮
- 按最近 20 轮速度估算，剩余 70 轮约需 19.6 小时
- 该估算只覆盖 Goal-pretrain；后续 matched GDTS、proposal bank、EBJD 训练与评测不在其中

## Checkpoint 收据

| 身份 | epoch | SHA256 |
|---|---:|---|
| `goal_pretrain_best_model.pt` | 35 | `6f000e5c6f1642dd959107d1908e8b041706f09db1a135bc4f6a43d668145a88` |
| `goal_pretrain_last_model.pt` | 80 | `033c3347690a7394d4c6330f118b10e40a07acdb8127c8914aaa210c5a2a1e48` |

这些权重保存在持久资产目录，不上传 Git。后续流水线将使用完成训练后冻结的验证最佳
Goal-pretrain 权重，而不是默认使用 epoch 150 的最后权重。

## 后续自动流程

Goal-pretrain 完成后，现有单实例 supervisor 将依次执行：matched GDTS 训练、
matched GDTS 评测、train/validation proposal bank 导出、EBJD 统一模型训练与最终评测。
当前未启动额外训练 seed，也未修改正在运行的配置和进程。
