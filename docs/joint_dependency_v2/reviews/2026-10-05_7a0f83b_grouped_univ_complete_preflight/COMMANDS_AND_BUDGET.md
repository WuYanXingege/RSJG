# 实际配置、命令与预算边界

本轮没有启动正式长训练。`FRESH_GOAL_BLOCKED.yaml` 不是 READY 配置：train+inner 的 released semantic raster 尚缺资产级 producer/training-scope 资格。缺少 qualified goal/base 不是 fresh goal 的额外门槛。

仓库目录：`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean`。
在此目录执行以下**检查命令**（预期因明确语义来源缺口非零退出）：

```bash
../.conda/rsjg/bin/python -B tools/grouped_preflight.py check --archive docs/joint_dependency_v2/reviews/2026-10-05_7a0f83b_grouped_univ_complete_preflight
../.conda/rsjg/bin/python -B main.py --p2_config docs/joint_dependency_v2/reviews/2026-10-05_7a0f83b_grouped_univ_complete_preflight/FRESH_GOAL_BLOCKED.yaml --p2_launch_check
```

未来正式入口形状已接通，但下列命令**本轮不执行，当前也必须被 gate 拒绝**：

```bash
../.conda/rsjg/bin/python -B main.py --p2_config docs/joint_dependency_v2/reviews/2026-10-05_7a0f83b_grouped_univ_complete_preflight/FRESH_GOAL_BLOCKED.yaml
```

拿到真实 semantic 证明后需生成新的 hash-bound qualification/manifest/config，重新做 production full-loader 与真实 inner selection 验收，联合 check exit0 后才可明确授权正式运行。不能手改 `UNKNOWN` 字符串、删 gate 或将本文件改名为 READY。

对应接口已提供：`tools/grouped_pack_audit.py --production --archive` 使用生产gate遍历全部pack；
`tools/grouped_inner_selection.py --archive` 在来源gate通过后使用hash核验的隔离smoke checkpoint做完整391-pack接口检查，
不更新参数。后者必须置于supervisor的GPU累计预算下运行；本轮只执行其来源拒绝路径，不查询GPU、不读取inner labels做模型metric。
二者不是用户可用来绕过来源资格的替代入口。

## 预登记正式运行合同

| 项目 | fresh goal | fresh joint |
|---|---|---|
| 初始化 | 新 seed3101、无父权重 | 只加载实际合格 goal；history/diffusion fresh |
| 精度/优化器 | FP32 / Adam lr0.001 / clip1 | FP32 / Adam lr0.0001 / clip1 |
| scheduler | 每个完整训练+selection epoch 后 ExponentialLR gamma0.99 | 相同 |
| 训练/选择 | train175 packs/11122 exposures；inner391/24955 | 相同 |
| 选择 | agent-weighted masked goal BCE，strict-earlier tie | 原 ADE/native pixels，P20，strict-earlier tie |
| 最大epoch/更新 | 150 / 26250 | 250 / 43750 |
| 首次运行 wall 失败预算 | 24小时 | 48小时 |
| 续跑 | FRESH_START_ONLY；目录必须全新 | 相同 |

wall 在每个训练 pack 和 selection pack 读取前检查；partial epoch/selection 不推进 scheduler、不选择 best、不保存可作为父权重的完整 epoch。单个正在执行的 pack/原子 checkpoint I/O 不是可抢占硬实时边界。非有限 loss/梯度、分母不完整、hash/用途错误立即失败。完整运行 receipt 才能封存最终选中的 checkpoint；进行中的 best 不获得父权重资格。early stop 默认关闭（patience0）；没有 outer metrics 或自动 train_test。

## 已测速度与估计的边界

首轮预登记CUDA FP32原生64-agent两步（原始结果保留，不择优挑速度）：

- goal loss/backward/clip/step：4.1791秒冷启动、0.0500秒第二步；整个goal case 5.8965秒，包含输入、obs-only推理及状态保存。
- joint：1.5308秒、0.2339秒；整个joint case 7.6269秒，包含P20 train-only推理与状态保存。
- BF16各一步：goal1.2796秒、joint0.6246秒，只证明支持路径，不改变正式FP32配置。
- 首轮两个GPU子进程保守 wall 总计38.1436秒，峰值 reserved 1,826,619,392 bytes。最终合同复验也计入后，四个GPU子进程合计73.7793秒；全历史峰值见RESULTS/GPU账本。首步含初始化/编译等开销，不能当稳态速度。

两个固定 pack 不代表所有地图尺寸与 agent 尾包。仅作为规划范围：goal 175个训练pack按0.05—2.95秒/pack为约9—516秒；joint按0.234—3.82秒/pack为约41—669秒。这是从少量case外推的范围，不是完整epoch实测。

391个inner pack尚未获准真实模型预测，因此 selection wall **未测得**。若暂以goal0.05—2.95秒/pack、joint0.2—8秒/pack作为明确的规划假设（非已测结果），含selection的epoch粗范围分别约0.5—28分钟、2—64分钟，另加实际I/O；150/250epoch可能超过24/48小时失败预算，不能承诺跑满上限。完整CPU文件/packing I/O时间由最终结果单独报告，不能把纯输入准备速度当GPU selection速度。

## 条件性后续阶段

- `FRESH_JOINT_CONDITIONAL.yaml`：等待正式训练得到、且只读 qualification 工具实际核验的 goal 文件。不得使用本轮 smoke 或未更新初态。
- cache：等待合格完整base，并额外满足实际消费outer observation的semantic资格。deployment4249、teacher仅train2537；本轮不生成。
- `A0_CONDITIONAL.yaml` / `A1_CONDITIONAL.yaml`：相同base、共同未训练初态、seed3101、reference2537、H<=500；默认mean-energy与MC S4独立RNG。cap处progress=500/2537、beta≈0.09854158；新旧L1不是同一实验。
- L2默认关闭。`l2_inner_evaluate` 只按原生inner窗口先预测后读target，以原JADE/JFDE world-metre指标聚合；`L2Selector` 实现主JADE/次JFDE、min_delta、patience与strict-earlier tie。仅有synthetic参考测试，没有真实L2/outer评估。
- `tools/grouped_pipeline.py check` 逐级检查真实父产物/缓存/共同初态；不自动串行启动训练。`prepare-shared` 仅在真实依赖满足后创建独立未训练heads；本轮未执行。
