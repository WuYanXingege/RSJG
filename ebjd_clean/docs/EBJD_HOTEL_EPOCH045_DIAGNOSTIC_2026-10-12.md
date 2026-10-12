---
experiment_id: EBJD-DIAG-261012-001
date: 2026-10-12
system: EBJD
type: frozen_prediction_bank_diagnostic
experiment_type: frozen_prediction_bank_diagnostic
dataset: HOTEL
training_seed: 3101
inference_seeds: [2035, 2036, 2037, 2038, 2039]
checkpoint: epoch_045.pt
checkpoint_sha256: 425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1
source: 3b13adb8df5ea0ce3a094ab6fef8054ae0e3e662
execution_commit: 6870c188e038ddb50f2134d814dfb57a8621df4c
status: COMPLETED_ZERO_TRAINING_DIAGNOSTIC
device: NVIDIA GeForce RTX 5070 Ti
tags: [HOTEL, epoch45, frozen-bank, marginal, joint, K-curve, world-permutation, zero-training]
---

# HOTEL epoch 45：边际退化与联合收益诊断

## 结论

本轮诊断通过 parity，可以作为已归档 epoch 45 五推理种子结果的后续分析。结论不是“EBJD 整体失败”，而是一个明确的样本预算权衡：EBJD 在低至中等 K 下兼顾边际与联合误差，但候选集合较窄，GDTS 在 K=20 时依靠更宽的尾部覆盖反超边际指标；EBJD 的联合优势则在多人场景仍然很强，并且部分依赖同一个 world 内不同个人轨迹的配合。

下一轮只建议一个主实验：从 epoch 45 的完整 model/optimizer/RNG/selection 状态，严格配对续训 5 轮，对照 `goal_v_weight=1` 与候选 `goal_v_weight=2`。桥权重保持 1、粗头系数保持 0.25，S4、20 步、frequency compensation 和投影均不变。本轮没有启动该训练。

## 1. Parity 与正式指标

CPU 从冻结预测库逐 seed 回放，和两份原始 JSON 的最大绝对差为 `6.325105800231157e-09 m`，低于预设 `1e-6 m` 门槛。两模型历史和未来身份对齐最大差分别为 `2.146e-6 m` 与 `1.907e-6 m`，低于预设 `1e-5 m` 门槛。场景、完整 frame IDs 和 agent IDs 均参与匹配，没有坐标近邻猜配。

| 模型 | minADE | minFDE | JADE | JFDE |
| --- | ---: | ---: | ---: | ---: |
| EBJD epoch 45 | 0.138007 | 0.202037 | 0.168566 | 0.276314 |
| GDTS | 0.134384 | 0.191589 | 0.208469 | 0.357055 |
| EBJD − GDTS | +0.003623 | +0.010448 | −0.039903 | −0.080740 |

因此原有判断被完整复现：K=20 时 EBJD 的个人边际指标略差，联合指标明显更好。五个 seed 是推理随机种子，不是五个训练重复；validation/test 的 445 个窗口镜像且 `independent_test=false`，不能据此作跨训练种子显著性或独立测试泛化声明。

机器可读主结果见 [诊断 JSON](validation/EBJD_HOTEL_EPOCH045_DIAGNOSTIC_2026-10-12.json)，逐 seed parity 见 [parity.csv](validation/epoch045_diagnostic_20261012/parity.csv)。

## 2. 边际退化分布：少数重尾是主因

逐 agent occurrence 的 EBJD−GDTS 差值中，minADE/minFDE 中位数分别是 `−0.0152/−0.0268 m`，但均值为 `+0.0036/+0.0104 m`，95 分位达到 `+0.1768/+0.3697 m`。也就是说，超过一半的记录实际更好，较少但严重的退化案例把总体均值推高。

| 预定义分层 | agent 数 | ΔminADE | ΔminFDE | 判断 |
| --- | ---: | ---: | ---: | --- |
| N=1 | 144 | +0.02435 | +0.07276 | 明确较差，是最干净的非交互边际证据 |
| N=2–3 | 342 | −0.00328 | −0.00299 | 略好 |
| N≥4 | 711 | +0.00274 | +0.00429 | 均值轻微较差，但联合明显更好 |
| 多人且有历史交互边 | 963 | −0.00091 | −0.00134 | 边际未见损失 |
| 多人但无历史交互边 | 90 | +0.01894 | +0.03689 | 较差 |
| 明显减速 | 26 | +0.22519 | +0.56214 | 最大退化，小样本且标签可重叠 |
| 停止/缓行 | 563 | −0.01378 | −0.02334 | 更好 |

N=1 场景中 joint 与 marginal 相同，EBJD 为 `0.14088/0.23117`，GDTS 为 `0.11652/0.15841`；这排除了“所有边际退化都由多人联合选择造成”的解释。相反，在 N=2–3 和 N≥4 场景，EBJD 的 JADE/JFDE 分别为 `0.16282/0.26095`、`0.19947/0.33227`，均显著低于 GDTS 的 `0.21883/0.38817`、`0.28372/0.51150`。联合收益主要存在于多人窗口。

运动标签是仅由历史和真值生成的预定义诊断标签，不是领域通用类别，并允许重叠。转向类别只有 8 个 agent，虽然结果较好，也不能推广。完整人数、运动与交互分层见 [marginal_strata.csv](validation/epoch045_diagnostic_20261012/marginal_strata.csv)、[paired_deltas.csv](validation/epoch045_diagnostic_20261012/paired_deltas.csv) 和 [motion_continuous.csv](validation/epoch045_diagnostic_20261012/motion_continuous.csv)。固定规则选出的最差 5 个与最好 5 个窗口见 [案例图](figures/epoch045_diagnostic_20261012/epoch045_fixed_rule_cases.pdf)；选择依据是五 seed 平均配对 minFDE 差，不是人工挑图。

## 3. 终点覆盖与 K 曲线

EBJD 的候选中心更接近真值（中心偏移 `0.5210` 对 `0.6172 m`），候选中位/90 分位 FDE 也更小（`0.6089/1.1342` 对 `0.9085/1.4891 m`），但终点 RMS 离散度更窄（`0.6092` 对 `0.7701 m`）。0.1 m 命中率 EBJD 更高（0.463 对 0.419），但从 0.2 m 起 GDTS 更高：0.2/0.5/1.0 m 分别为 `0.680/0.940/0.985`，EBJD 为 `0.669/0.905/0.976`。这更符合“EBJD 候选集中、GDTS 宽尾提供额外覆盖”，而不是简单的中心偏置。

共享随机子集的均值如下；每个场景内所有 agent 使用相同 world 索引，两模型同一重复也使用相同索引。

| K | EBJD minADE/minFDE | GDTS minADE/minFDE | EBJD JADE/JFDE | GDTS JADE/JFDE |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.2098 / 0.3702 | 0.2667 / 0.4801 | 0.2318 / 0.4234 | 0.3131 / 0.5786 |
| 8 | 0.1733 / 0.2860 | 0.1979 / 0.3335 | 0.2001 / 0.3503 | 0.2600 / 0.4679 |
| 12 | 0.1562 / 0.2456 | 0.1667 / 0.2649 | 0.1850 / 0.3149 | 0.2352 / 0.4149 |
| 16 | 0.1455 / 0.2201 | 0.1476 / 0.2220 | 0.1754 / 0.2923 | 0.2197 / 0.3815 |
| 20 | 0.1380 / 0.2020 | 0.1344 / 0.1916 | 0.1686 / 0.2763 | 0.2085 / 0.3571 |

GDTS 直到完整 K=20 才在 marginal 上反超；EBJD 的 joint 优势贯穿 K=4–20。原顺序嵌套 prefix 的四个 hard 指标均严格不随 K 增加而变差，单调性最大正偏差为 0。K=4、`eta=0.05 m` 的 normalized soft proxy 也全部偏向 EBJD，但它只是冻结评价窗口风险，不是训练 rollout loss。

这构成了有限样本覆盖和训练 S4／评价 P20 关系的强诊断证据，但不能单独证明训练目标因果失配。数据与 5–95% 分位见 [K 曲线 CSV](validation/epoch045_diagnostic_20261012/k_curve_summary.csv)，可视化见 [K 曲线图](figures/epoch045_diagnostic_20261012/epoch045_k_curve.pdf)。

## 4. 独立 world 排列负控

100 次独立 agent world 排列保持每个人的 20 条完整轨迹集合不变，只破坏同一 world 内不同 agent 的配对。所有保护不变量均通过：候选集合差 0、marginal 差 0、全员共同排列最大差 `1.11e-16 m`、N=1 joint 差 0。

| 模型 | 原 JADE → 打乱均值 | ΔJADE（5–95%） | 原 JFDE → 打乱均值 | ΔJFDE（5–95%） |
| --- | ---: | ---: | ---: | ---: |
| EBJD | 0.16857 → 0.17623 | +0.00767 (+0.00550, +0.00968) | 0.27631 → 0.29729 | +0.02097 (+0.01655, +0.02519) |
| GDTS | 0.20847 → 0.20782 | −0.00065 (−0.00420, +0.00327) | 0.35705 → 0.35754 | +0.00048 (−0.00690, +0.00918) |

EBJD 相对 GDTS 的联合优势从 JADE/JFDE 的 `−0.03990/−0.08074 m` 缩小到 `−0.03159/−0.06025 m`，但没有消失。证据支持 EBJD 的 world 内配对确有价值，同时说明联合优势并不全由该配对解释。它不能证明社会注意力、桥结构或任何单个模块的因果贡献，也不能证明轨迹物理无碰撞或更安全。完整分布见 [排列 CSV](validation/epoch045_diagnostic_20261012/world_permutation_summary.csv) 与 [排列图](figures/epoch045_diagnostic_20261012/epoch045_world_permutation.pdf)。

## 5. 下一轮唯一优先短程实验

主候选为终点 diffusion 监督权重 `goal_v_weight: 1→2`，原因是：

1. N=1 已有明显边际和终点退化，不需借助交互机制即可复现；
2. 明显减速小组有很强的终点重尾；
3. 多人有边场景的 marginal 并未变差，且排列负控证明保留 world 配对有价值，因此现在优先削弱联合项可能误伤已获得的收益；
4. K 曲线提示候选覆盖是重要问题，所以该 5 轮实验是区分性测试，不保证一定修复。

实验从 epoch 45 边界的完整状态恢复：checkpoint 内含 model、optimizer、Python/NumPy/Torch CPU/Torch CUDA RNG、loader generator、update=11970 和 selection state，因此属于精确续训／目标分叉，不是仅权重微调。两臂都运行 epoch 46–50，保留原 100 epoch 学习率时间轴、数据顺序和 RNG；候选臂唯一语义变化是 `goal_v_weight=2`。每臂约 `5×40.82 min = 3.4 h`，串行约 6.8 GPU 小时。

配置草案见 [NEXT_SHORT_EXPERIMENT_CONFIG_DRAFT.yaml](validation/epoch045_diagnostic_20261012/NEXT_SHORT_EXPERIMENT_CONFIG_DRAFT.yaml)，精确但未执行的命令见 [NEXT_SHORT_EXPERIMENT_COMMANDS.md](validation/epoch045_diagnostic_20261012/NEXT_SHORT_EXPERIMENT_COMMANDS.md)。训练入口增加了严格白名单 objective fork 与 5 轮边界，并保存 `best_joint_short`、`best_marginal_short`、`least_violation_short` 和 `last`；不传新参数时原训练行为不变。

备选研究问题是 S4 风险与 P20 覆盖的关系。它应在主候选无效或边际仅在 K=20 恶化的现象被再次确认后再设计，不应通过扩大论文主表 K 来规避。

## 6. 仍不能判断的事项

- 不能把边际重尾归因于桥、恒速中心化、社会注意力或联合损失中的某一项；EBJD 与 GDTS 的结构和初始化整体不同。
- 不能由单一训练 seed 判断跨训练随机性的稳定性或统计显著性。
- validation/test 镜像意味着这里不是未参与选模的独立测试。
- “联合误差更低”不等同于碰撞率更低、安全性更高或物理可行性更强。
- 明显减速和转向分层样本量小；它们用于定位后续测试，不能独立支撑总体结论。

## 7. 执行与身份收据

| 项目 | 记录 |
| --- | --- |
| 协议 | HOTEL 官方 leave-one-scene-out 镜像窗口；445 scenes/seed；1,197 agent occurrences/seed；8→12；dt=0.4 s；米制 |
| 采样 | 20 worlds；20 DDIM steps；seeds 2035–2039；P20 |
| EBJD | FP32 model/sampler/geometry，autocast 关闭，native B=1 DDIM20 |
| GDTS | native loader/batch/RNG/TTST/tree-DDIM；model forward 使用 CUDA BF16 autocast；坐标转换 FP32 且只转换一次 |
| GPU forward | 4,450（2 models × 5 seeds × 445 scenes） |
| optimizer updates | 0 |
| GPU 时间 | 1,715.37 s（28.59 min）；EBJD 671.85 s，GDTS 1,043.52 s，串行 |
| 峰值显存 | EBJD 689,397,248 B；GDTS 143,685,120 B；总报告取 689,397,248 B |
| checkpoint 完整性 | 导出后 EBJD/GDTS SHA256 仍分别为 `425988…e1b1` / `5c101c…b97b` |
| bank | 主机持久目录 `RSJG_EBJD_clean/ebjd_clean/outputs/evaluations/hotel_epoch045_diagnostic_20261012/banks/`，不上传 GitHub |

预测库只含必要 FP32 米制轨迹、真值、历史和身份元数据；NPZ 使用 `allow_pickle=False`，每 seed manifest 记录文件哈希。聚合 bank 收据见 [BANK_RECEIPTS.json](validation/epoch045_diagnostic_20261012/BANK_RECEIPTS.json)。本轮未做 loss、backward、optimizer step 或训练；源 checkpoint 文件哈希未改变。

## 8. 实现、测试与图形 QA

实现新增安全预测库、EBJD/GDTS 原生只读 exporter、CPU 诊断 CLI、严格短程 objective fork 和 materializer。原 GDTS、原 checkpoint、原数据和正式输出未修改。运行：

```bash
cd /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_epoch45_diagnostic_worktree/ebjd_clean
/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python -m pytest -q
```

结果为 `44 passed`。测试覆盖独立数学参考、marginal/joint 差异、ADE/FDE 不同最佳 world、各种排列不变量、mask/padding、ID 重排、空分层、K 单调性、goal/末步、单位只转换一次、shape/P=20、NaN/Inf、身份重复/缺失及哈希损坏拒绝。

图使用 Python/matplotlib，病例选择规则预先固定，K 曲线显示 5–95% 随机子集分位；PDF/SVG 保持可编辑文本，另有 300 dpi PNG。三图均通过 panel alignment、渲染碰撞、PDF 文字和最终尺寸检查。TIFF 未生成，因为本轮是网页诊断归档而非投稿生产包。详情见 [FIGURE_QA.json](validation/epoch045_diagnostic_20261012/FIGURE_QA.json)。
