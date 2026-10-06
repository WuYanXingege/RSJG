# 历史 GDTS ETH checkpoint 的官方原生窗口重评

## 结论

状态：**OFFICIAL_EVALUATION_COMPLETE_PAPER_METRIC_NOT_REPRODUCED**。

在锁定的 GDTS 上游 commit `297d508558c10831983ea4b19c2b3e657459a449`
工作树中，历史 ETH checkpoint 已严格加载成功，识别为 epoch 90，并在历史原生
train 579 / valid 41 cache 上完成官方测试入口的 5 次 K=20 推理。官方入口实际
调用 `mode=valid`；其 valid/test 均为同一 ETH 来源，这是公开代码的既有边界。

五轮平均为：

- `ADE20 = 0.34644 m`；
- `FDE20 = 0.49529 m`。

按论文两位小数为 `0.35/0.50`，没有复现正式论文 Table I 的 ETH
`0.31/0.48`。绝对差为 `+0.03644/+0.01529 m`，相对论文值约
`+11.75%/+3.19%`。因此旧 checkpoint 可以继续作为历史内部实验父权重，但不能
替代论文主表中的已认证本地 GDTS ETH 复现。

## 输入绑定

| 对象 | 绑定 |
|---|---|
| GDTS 上游源码 | `297d508558c10831983ea4b19c2b3e657459a449` |
| 兼容补丁范围 | 仅 `src/data_loader.py` 的 Albumentations/OpenCV 兼容；不改模型、loss、sampling 或 metrics |
| patched loader SHA256 | `c6c66e761455c126dda173d107033dd30e60d4777acafeb285bd671632376952` |
| checkpoint | `../GDTS/output/eth/saved_models/best_model.pt` |
| checkpoint SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| checkpoint epoch | 90 |
| 原生 cache | train 579 / valid 41 / test 41 |
| 运行配置 SHA256 | `644e93aa424b44730bc1c00aeaa3ad218013e9e55f0ac96eeedce868a7731b43` |
| 完整日志 SHA256 | `e365cd2b03c71c5440f3ef208306a2ef1e528070ae289ac44a4f56782e7aa0da` |

运行参数为 `eth5/eth`、phase test、checkpoint best、batch 64、K=20、TTST、
DDPM 100、DDIM 20、trunk 30、5 test runs、seed 2025、FP32 CUDA。一次进程中
设置 seed 后连续执行五轮，与官方 `trainer.test()` 控制流一致。

## 逐轮结果

| run | ADE20 (m) | FDE20 (m) |
|---:|---:|---:|
| 0 | 0.34435 | 0.49592 |
| 1 | 0.34807 | 0.49693 |
| 2 | 0.34433 | 0.49458 |
| 3 | 0.34806 | 0.49223 |
| 4 | 0.34736 | 0.49679 |
| 官方打印均值 | **0.34644** | **0.49529** |

由日志中五位小数逐轮值重算的 population std 约为 ADE `0.001729 m`、FDE
`0.001744 m`；这是推理随机性，不是五次独立训练的不确定性。

## 与既有结果的关系

此前 `0.286332/0.394404 m` 使用同一 checkpoint，但通过 JDV2 审计 loader 在
139 个同步窗口上计算。当前官方原生 41-batch 重评直接证实两者不是同一评测
population。`0.286/0.394` 仍可作为 JDV2 139-window 的内部配对参照，但不能与
论文 Table I 直接比较。

历史训练曲线完整到 epoch 250，配置也包含 batch 64、lr `1e-3`、250 epochs、
K=20、TTST 和 seed 2025；但历史训练没有 source-tree SHA/dirty-state 收据，且
`pretrain_path=null`。论文第 5 页 Implementation Details 描述 goal 先训练 150
epochs、再与 trajectory module 训练 250 epochs。公开固定代码路径未实际加载该
goal pretrain。这个代码/论文边界可能影响结果，但本轮数据不足以把数值差异归因于
某一个因素。

## 执行与资源

- 运行时间：2026-10-06 15:00--15:07（Asia/Shanghai），7分19秒；
- GPU：RTX 5070 Ti；启动前约 12.8 GB 空闲；ETH 推理新增约 1.1 GB；
- 同时运行的 HOTEL 训练未停止、未修改、未加载 ETH 权重；ETH 完成后显存释放；
- 本轮 optimizer updates 0，checkpoint 写入 0，正式训练 0；
- HOTEL 在重评结束后的观测点已完成 epoch 64，仍继续运行。

## 后续决策

1. 论文表格必须把 `paper reported 0.31/0.48`、本次
   `retrospective official-code reevaluation 0.34644/0.49529` 和未来 fresh
   local retraining 分栏，不混写。
2. 若目标是可审计的五折本地 GDTS baseline，ETH 仍应像当前 HOTEL 一样在锁定
   工作树 fresh 训练；旧权重不再充当最终主表复现。
3. 若目标还包括论文意图复现，另设 goal-150 → joint-250 两阶段实验臂，不能把
   公开代码的 fresh joint-only 结果冒充论文两阶段结果。
4. 当前不抢占或停止 HOTEL；等其完成后再安排 ETH fresh 或其他标准折。
