# GDTS Multiway Coupling V4

## 1. 方法目标

V4 把问题限定为：给定已经很强的 GDTS 单人边际轨迹集合，能否只通过重新组合各行人的样本编号，得到更一致的多人联合预测。正式 UNIV 配置固定使用 Stage0 epoch 100；它与当前 `best_model.pt` 的 112 个权重张量逐个完全相同，但采用显式固定 epoch 路径，避免 V4 启动配置继续依赖含糊的 `best` 标签。

因此正式配置使用 `training_stage=multiway_coupling`、`trajectory_coupling=multiway_v4`。默认上游是冻结的 Stage0 independent GDTS。V4 不改扩散网络、不连续修正坐标、不增加或删除轨迹；它只学习每个行人的候选轨迹到全局 joint slot 的双射。

## 2. 完整数据流

1. 从 synchronized scene-window cache 读取同一时刻的全部行人，所有图几何量统一使用 world metre。
2. 冻结的 Stage0（默认）或 Stage1（消融）产生真实的 `K=20` 条完整未来轨迹，形成 `Y[N,K,T,2]`。
3. radius+TTC 图产生稀疏规范边 `edge_index[2,E]`、观测几何 `edge_feat[E,D]` 和 edge weight。训练期所有边都参与；可选 top-k 仅允许在推理期使用，正式默认关闭。
4. 共享完整轨迹编码器对每个候选提取位移、速度、加速度和归一化时间序列特征，输出 `h[N,K,128]`。
5. 对每条稀疏边构造 `pair_geometry[E,K,K,12]`，包含端点/最小/平均距离、队形变化、位移及速度相似度、速度分歧、最近接近时刻、closing rate 等。距离只是证据，不被硬编码成“越近越坏”。
6. 候选轨迹对条件关系后验按 `softmax(log q0 + residual)` 计算。Stage0 上游没有 Stage1 社会先验时，`q0` 明确取均匀分布；不会冻结一个随机关系头冒充先验。
7. relation-conditioned trajectory energy 默认采用 rank-8 低秩因子，并在 compatibility 空间做 latent-relation `logsumexp` 混合。另保留可运行的 direct MLP 消融。
8. pair-specific gate 输出 `g[E,K,K]`，全程连续且不断梯度。训练期不 detach、不 hard top-k。
9. graph-wide synchronizer 在每个连通分量上同步更新所有 agent 的 `P[N,K,K]`。约定始终为 `P[i,candidate,slot]`，边目标为 `<C_ij, P_i P_j^T>`，所以消息方向严格是 `C_ij @ P_j` 与 `C_ij.T @ P_i`。anchor 只固定 permutation gauge；其余所有图边仍参与目标。
10. 每轮同步后使用 log-space Sinkhorn，保证行列和接近 1。component confidence 在训练时把同步解与 identity 做连续混合，因而 keep head 能收到 task/no-harm 梯度；推理时仍按 component threshold 做离散 identity bypass。
11. 默认用 Hungarian 将同步矩阵投影为严格双射；SciPy 不可用时才显式告警并记录 greedy fallback。最终 `perm[slot]=candidate`，输出严格由 `aligned[i,slot]=raw[i,perm[i,slot]]` gather 得到。

## 3. 边际保护与训练目标

硬推理后的每一行 `perm` 必须恰好包含 `0..K-1`。代码同时检查轨迹集合完全相同，以及 raw/aligned minADE、minFDE 在 `1e-6` 容差内一致；失败会直接抛错。

V4 联合训练以下新模块：trajectory encoder、candidate-conditioned relation residual、trajectory pair energy、pair gate、multiway synchronizer 和 component confidence head。默认冻结 Goal U-Net、history encoder、diffusion denoiser以及旧 Stage1 模块。

默认损失为：

```text
1.00 L_alignment
+ 0.50 L_pair_score
+ 0.50 L_pair_assignment
+ 2.00 L_no_harm
+ 0.02 L_perm_entropy
+ 0.02 L_relation_prior
+ 0.00 L_gate_reg
```

- `L_alignment` 直接在每个 joint slot 的 expected per-trajectory ADE+FDE cost 上做 soft-min，不对轨迹坐标做软插值。
- `L_pair_score` 的 oracle 同时含个体误差、完整相对路径误差和相对端点误差。
- `L_pair_assignment` 监督同步后 `Q_ij=P_i P_j^T`，把 pair oracle 信息传入全局同步器。
- `L_no_harm` 只保护 joint surrogate；边际由硬双射精确保护。
- 小权重 entropy 缩小 Sinkhorn 与 hard projection 的差距；relation KL 限制后验无证据偏离观测先验。

## 4. Stage0 / Stage1 上游切换

- `--upstream_generator stage0_independent`：论文主配置。使用 Stage0 的 TTST+diffusion 边际样本；V4 不依赖 scene-level social mode。
- `--upstream_generator stage1_joint`：加载并冻结 Stage1 joint-goal 生成器，供 2×2 消融使用。
- `--freeze_upstream_generator False` 已保留为后续消融，但不属于主定义。开启后，不再能声称相对于预训练 GDTS 的跨 epoch 边际集合不变。
- `--social_mode_scope off` 是 V4 默认；`scene/component` 仅为未来消融预留，V4 synchronizer 当前不再强化全场共享 latent z。

## 5. UNIV 模型选择协议

ETH/UCY 当前 cache 中原 valid 与 test 来源相同，不能一边选择 checkpoint 一边作为最终结果。V4 默认执行：

- `model_selection_split=internal_train`
- `internal_validation_strategy=source_block`
- 完整保留训练 cache 最后一个 raw source file 作为内部验证；不会把相邻重叠 window 随机拆到 train/valid 两侧。
- 当前 UNIV cache 对应训练 2,982 windows，内部验证 320 windows（`uni_examples.txt`）。
- `final_test_split=heldout_test`，仅在 best epoch 确定后运行。
- 精确 cache ID、source、数量和 seed 写入 `evaluation_protocol.json`。

## 6. 日志和诊断

每次验证记录 raw/aligned minADE、minFDE、JADE、JFDE，以及：alignment oracle/headroom/recovery、changed fraction、keep confidence、pair gate、relation/permutation entropy、Sinkhorn 行列误差、soft-hard gap、agent/edge/degree/component size。还会将 N 桶和 component-size 桶写入 `diagnostics/{split}_epoch_XXX.json`。

训练完成后 `tools/generate_v4_report.py` 自动生成 `docs/UNIV_MULTIWAY_COUPLING_V4_RESULTS.md`，包含 best epoch、边际保持、联合提升、分桶、Hungarian-vs-greedy 和下一步消融建议。

## 7. 文件与兼容性

新增核心实现：

- `src/models/trajectory_pair_relation.py`
- `src/models/trajectory_pair_energy.py`
- `src/models/permutation_synchronizer.py`
- `src/models/hungarian_projection.py`
- `src/models/multiway_trajectory_coupler.py`
- `src/multiway_coupling_loss.py`

V4 继续复用 baseline 与 Stage1 的基础编码、采样和评测组件。历史 Stage2/V3 启动入口与运行产物已清理，但兼容模块暂时保留，因为当前模型装配和回归测试仍会引用它们。V4 resume/test 使用 strict checkpoint load。

## 8. 运行入口

```bash
bash train_packed_multiway_v4.sh eth
bash train_packed_multiway_v4.sh univ
```

训练脚本使用 `setsid + nohup` 启动轨迹缓存、训练和最终测试流水线。PID、日志、配置、checkpoint 与诊断分别保存在 `output/{eth,univ}/joint/runs/multiway_coupling_v4_packed_cache_v2/`，不会覆盖基础模型权重。
