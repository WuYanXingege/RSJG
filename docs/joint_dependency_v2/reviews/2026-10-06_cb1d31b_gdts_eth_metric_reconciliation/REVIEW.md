# GDTS ETH 论文指标与历史本地结果口径核对

## 结论

状态：**HISTORICAL_ETH_CHECKPOINT_REUSABLE_EVALUATION_NOT_PAPER_COMPARABLE**。

用户提供的 IROS 2025 正式 PDF 在第 5 页 Table I 报告 GDTS 的 ETH
`ADE20=0.31 m`、`FDE20=0.48 m`。此前本仓库引用的
`0.286332±0.005157 m / 0.394404±0.008789 m` 不是这两个论文数的复现：
它们虽然加载了历史 GDTS ETH checkpoint，却在 JDV2 的 139 个同步场景窗口上、
用后续审计器和固定推理种子 2035--2039 计算，而不是在历史 GDTS 的 41-batch
valid/test loader 上计算。

因此必须修正此前表述：历史 checkpoint 的训练配置与标准 ETH 折高度一致，权重可
保留并回读；但 `0.286/0.394` 只能作为 JDV2 139-window 内部同协议参考，不能写入
论文主表作为 GDTS 公开 ETH 指标复现，也不能与 Table I 的 `0.31/0.48` 直接作
复现误差比较。

## 论文原始依据

- 来源：用户提供的正式会议 PDF，SHA256
  `5067c46a25df4c571665af5a88fc715a78ac9245d88d6635b9004fb1c88e1b84`；
  DOI：<https://doi.org/10.1109/IROS60139.2025.11246846>。
- PDF 第 4 页说明 ETH/UCY 使用 leave-one-scene-out、观测 8 帧、预测 12 帧。
- PDF 第 5 页 Table I 的 `GDTS (Ours)` / `ETH` 行为
  `ADE20 0.31`、`FDE20 0.48`。
- 同页 Evaluation Metrics 段说明报告 best-of-N ADE/FDE；Implementation Details
  给出 batch 64、Adam 初始学习率 `1e-3`、goal 150 epochs 后 joint 250 epochs、
  DDPM 100、DDIM 20、最终预测数 20。

本归档不复制受版权保护的整页或全文，只记录核验所需事实、页码与表格位置。

## 历史训练 artifact

历史 ETH baseline 配置为 `eth5/eth`、batch 64、250 epochs、Adam `1e-3`、
ExponentialLR、TTST、K=20、seed 2025、每 10 轮验证。训练曲线完整到 epoch 250；
best checkpoint 元数据为 epoch 90。

| artifact | SHA256 |
|---|---|
| `../GDTS/output/eth/config.yaml` | `1906d36c39512de376359a890473de7f5678ec165d07344b47e41305b3a3481c` |
| `../GDTS/output/eth/log_curve.txt` | `4c14f523f45692a1a999481c2fb52ed69f3efd837e65b737a7b03ce107c35081` |
| `../GDTS/output/eth/saved_models/best_model.pt` | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| `../GDTS/output/eth/data_batches/cache_manifest.json` | `e7744060eb45d3274c0f80f34e3cfdebebb281ee47bb48fc82d90cfe7864ac61` |

历史原生 cache 的实际文件/完成标记计数为：train 579、valid 41、test 41。
这说明历史权重训练和选模使用的是 GDTS 原生 batching，而不是后来 JDV2 的
139-window loader。

## `0.286/0.394` 的真实来源

机器结果位于
`outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_engineering_fix/stage_a_audit_results.json`。
该文件明确记录：

- checkpoint 是上述 SHA256 为 `126acf...` 的 GDTS baseline；
- split 是 `valid`；
- 20 samples；
- 推理种子为 2035--2039；
- 通过后续 `GDTS.compute_model_metrics` 和
  `scene.make_world_coord_torch` 计算；
- mean±population std 为 minADE@K
  `0.286332±0.005157 m`、minFDE@K `0.394404±0.008789 m`。

但审计源码 `tools/audit_jdv2_stage_a.py::_baseline_evaluator` 先用 JDV2 配置创建
loader，再把网络替换为 baseline。对应运行日志明确是 train 4110、valid 139、
test 139。也就是说，它实现的是“相同 JDV2 窗口上的 baseline 参照”，并非对
GDTS 原生 579/41/41 loader 的正式重评。

## 为什么不能把差异解释成单纯随机波动

论文值保留两位小数；本地均值按相同精度会是 `0.29/0.39`，尤其 FDE 与论文
`0.48` 相差约 `-17.8%`。但在确认相同样本集合、聚合顺序、推理 RNG 和代码路径
之前，这一百分比没有严格可比意义。当前已经确认样本/loader 口径不同，因此不能
把差异归因为“更好的复现”、checkpoint 改进或普通 seed 波动。

## 处置

1. 不删除、不覆盖历史 checkpoint、曲线或 139-window 结果。
2. 立即停止把 `0.286/0.394` 称为论文 ETH baseline 指标复现；所有旧报告中的
   数值继续作为当时的 JDV2 内部同协议证据，并附本更正。
3. 当前 HOTEL 正式训练继续运行，不插入新的并发 GPU 任务。
4. HOTEL 结束或 GPU 空闲后，在锁定的上游 `297d508` 工作树中加载旧 ETH
   checkpoint，用原生 41-batch ETH test loader 跑官方 5 次 K=20 评估。
5. 若 state dict 严格兼容且重评链通过，可复用旧权重，无需重训；最终报告同时列
   `paper reported 0.31/0.48` 与 `local official-code reevaluation`。若无法严格加载或
   输入/输出不一致，才升级为重训 ETH。

## 边界

- 历史 checkpoint 和曲线的文件时间早于仓库首次 joint 扩展 commit，但没有当时的
  source-tree SHA/dirty-state 收据；这支持“候选可复用”，不足以单独认证逐字官方代码。
- 官方数据包中 valid/test 同内容的问题影响当前公开代码复现的解释，但不是本次数值
  错配的主因；本次已直接证实的主因是 41-batch 与 139-window 评测 loader 不同。
- 本轮仅做 PDF、配置、缓存、源码和既有 JSON 的只读核对；GPU 评测 0、训练更新 0。
