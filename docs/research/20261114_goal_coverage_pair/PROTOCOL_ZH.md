# ETH逐人覆盖与联合配对分工：预先固定协议

本轮是目标函数对照，不是新的网络结构或执行加速。沿用旧可学习20分量高斯终点头与8层桥；不使用已拒绝的空间查询头，不恢复旧100轮，不加载外部GDTS/Goal-pretrain权重。

## 唯一干预

Control维持NLL＋联合配对目标。Candidate保留NLL，增加逐人20候选soft-min终点覆盖项（按有效行人平均、权重1、温度0.1、同原goal_scale归一化）；配对时传入`proposals.detach()`，只隔离到终点头的直接梯度。配对模块及共享上下文仍参与训练，不冻结共享编码器。

覆盖项先沿候选轴soft-min再平均行人；原配对项先平均同world的行人误差再soft-min。两个目标分别面向边缘覆盖和共同world质量。单人时两种误差数值一致；候选的单人配对项无直接坐标梯度，由覆盖项承担终点更新。候选总loss多了一项，不能用total loss变大或变小评价方案。

这项研究同时改变覆盖监督与梯度分工，属于一个联合假设，不能据此分别声称两个子改动的独立贡献。它不是已证实的收益或新的独占算法贡献。

## 配对与判据

- 共同父状态：原从零训练的epoch44/update11308，SHA256 `ae048b91555046f61398f28a18dbef182d1193173d20c9cef0f5c5d9d4e2515e`。
- 两臂各8次真实更新检查，随后分别独立恢复父状态，各训练epoch45–47，固定比较epoch47。
- 模型、Adam状态、100轮LR时间轴、输入顺序、增强和随机状态保持对应；不改batch、损失其他权重、推理20/20或rollout/projection OFF。
- 保留全部训练窗口，每epoch257次update。每轮完整验证seed2035；末轮复用对应结果并补齐2036–2039。父模型也按相同五推理seed评测。三个模型均不重新选模。
- 筛选：Candidate五seed均值minFDE比Control至少低0.01m，其他三指标均值不退化，Recall@0.5m不降低，至少4/5seed minFDE改善；相对Parent的minFDE改善，其他三项均值不退化。
- 均值±SD仅描述一个训练seed下的推理波动，不能称为独立训练稳定性或统计显著性。验证/test镜像事实不变。

候选未通过即DO_NOT_ADOPT；即使通过也只标记PROMISING_SHORT_SCREEN，不自动启动100轮或扩展其他实验。

## 运行与恢复

资产目录：`RSJG_EBJD_assets/experiments/eth_goal_coverage_pair_v1_20261114_134751/`。

入口：`scripts/run_goal_coverage_pair.py --root <资产绝对路径> --source-commit <冻结提交>`。外部启动使用nohup＋独立session，内部使用已有GPU排他锁。重启从各臂最后完整epoch恢复模型、Adam、RNG和学习率时间轴；未保存的更新记录为待重放，不伪装成完整epoch。不重复已落盘的最终seed评测。完成后重复启动只返回完成状态。

主要状态：STATUS.json、GPU_COST.json；控制台每epoch一行，原始逐update日志在logs。完整last.pt只留在资产目录。所有状态JSON原子写入。异常保存原因后停止，不绕过失败继续训练。原模型的parent.pt仅引用，不重复复制大缓存或checkpoint。

最终自动生成reports/REPORT_ZH.md、SUMMARY.json、PER_SEED_METRICS.csv，并复制小型结果到本目录；不会自行提交或推送，也不会改变冻结源码HEAD。
