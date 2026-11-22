# ZARA2 场景级 I/H0 选择实验

固定源为旧 coordinate 原生提交 `468c5eddde48bdeec47c4bb3c39418352be16bb9`，epoch 5 权重 SHA256 为 `9281673fda5c439bdd463b0bfc9af1a65a3a0fea30941ad3c956cbb617943317`。本分支仅给原配对前向增加返回第一次 Hungarian 排列的可选参数；默认调用保持原数学运算和返回接口。

选择头有 18,562 个参数。节点输入为相对末观测点的八步历史、末四步速度、冻结的 192 维 social_h 和候选位移。每个联合场景的边输入为历史位置差、速度差和配对终点差。采用原历史最近八邻居规则并包含边界同距邻居，先编码边，再对邻居、行人和全部 20 个联合场景做有效掩码 mean/max 池化。I 与 H0 使用同一个打分器，双输出之差拟合米制 dA、dF，不缩放标签。

特征均值和标准差仅由 train 的有效节点与边拟合；标准差下限固定为 1e-3。逐窗双输出 MSE，每次更新 16 窗，末尾实际窗口数作分母。除学习率、衰减、warmup、调度和早停使用原配方外，不新增损失、阈值或调参轮次。

正式入口：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../.conda/rsjg/bin/python -m scene_pair_selector.run --wait-gpu
```

同一命令在技术中断后读取本实验 last。缓存逐窗原子写入，训练复用原 `bcep_v2/state.py` 完整状态 writer。保留全部完整 epoch、best、last 及按选择后 dev JADE 排序的五个不同完整 epoch；严格同分取较早轮次，不删除已有资产。

CPU 接入检查覆盖掩码/世界重标号不变性、跨人组合敏感性、固定规则边界、keyed noise 的 bank 槽位身份、原生联合指标计算顺序，以及真实 writer 的非零更新恢复、Top5 排序与状态完整性。GPU 释放后另以三个真实窗口验证强制 H0 与原入口逐值一致，然后直接生成标签和进行一次训练。最终结果仅来自 rank1 的实际单路解码复核。
