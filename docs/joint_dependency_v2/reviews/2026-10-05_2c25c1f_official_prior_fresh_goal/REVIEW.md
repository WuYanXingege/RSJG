# 官方语义先验独立协议与 fresh goal 验收

结论：**GOAL_TRAINING_READY**。本协议明确采用 GDTS/Goal-SAR 作者发布的语义栅格作为冻结环境输入；分割器训练范围仍为 `UNKNOWN`，因此本结论不是 strict-clean 语义来源认证，也不改变上一份 strict-clean 协议的 `EXTERNAL_EVIDENCE_BLOCKED` 状态。

协议身份为 `p2_grouped_univ_hotel_official_prior_v1`，data binding 为 `c17a6f031f0ffb1d94011f85bb8981aaef518b96863fbd80a51159a9d9361bff`。所有实验臂须使用完全相同、逐文件 SHA256 冻结的作者语义输入；只有 train 可产生梯度，inner 只在预测后读取 target 做选模，outer 不参与训练、调参或本次指标。

## 实际验收

| 项目 | 结果 |
|---|---|
| 数据角色 | train 2537、inner 1267、outer 445；原窗口、成员、轨迹划分和固定 order 不变 |
| 生产 loader | PASS；train 175 packs/11122 agent-window exposures，inner 391/24955 |
| CPU 回归 | PASS；119 项协议/loader 测试 + 50 项 observation guard 测试 |
| 真实小步更新 | 累计上限内 24 attempts/24 successes，其中本协议新增 CPU 2、CUDA 6；无失败/跳过 |
| CUDA | FP32/BF16 goal 与 joint 路径 PASS；峰值与逐项证据见 smoke receipts |
| reload | 独立进程 checkpoint、optimizer、scheduler、RNG 与固定输入 replay PASS |
| 完整 inner 接口 | goal 391 packs/24955，零优化更新，目标在预测后读取；仅工程门禁，不作为性能结论 |
| fresh 初态 | seed3101 未更新初态 state SHA `ff931446488399425f54ea689aaec0409c64319bd2a0ece74cf7366e65d7c8f0` |
| 正式入口 | `main.py --p2_launch_check` exit 0；所有当前协议收据与当前源码指纹一致 |

权威机器状态见 [READINESS_MATRIX.json](READINESS_MATRIX.json)，配置见 [FRESH_GOAL_CONFIG.yaml](FRESH_GOAL_CONFIG.yaml)，汇总见 [RESULTS.json](RESULTS.json)。

## 结论边界

- 作者 archive 到本地 `H.txt`、RGB、`pred_mask.png` 的文件字节绑定已核对；这不证明分割模型的训练 fold 或标签来源。
- 接受的是“官方语义先验设置”，不是“严格隔离训练出的语义预处理器”。论文与结果必须保留该披露。
- 完整 inner 检查使用的是 `SMOKE_ONLY` checkpoint，只证明 loader、预测、target 时序和聚合接口可执行，不证明收敛或性能。
- fresh goal 配置为 FP32、seed3101、batch64、Adam lr0.001、最多150轮、24小时 pack-boundary 预算、`FRESH_START_ONLY`。
- fresh joint、cache、A0/A1 都仍等待各自真实父产物；不会由本次启动器自动串行启动。
- 历史审查和 SDD 运行未修改。

首次 CPU 回归有 1 个反例触发了未结构化的 `AttributeError`；修复为类型检查后完整重跑通过。失败 attempt 和后续两次通过记录均保留，没有覆盖历史证据。

本归档先随训练源码提交并上传；正式训练启动后另写只读启动快照，不把运行中的日志或 checkpoint 提交到 Git。
