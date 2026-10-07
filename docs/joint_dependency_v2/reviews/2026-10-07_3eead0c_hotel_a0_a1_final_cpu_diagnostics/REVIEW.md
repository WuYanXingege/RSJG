# HOTEL A0/A1 最终审查：CPU 阶段性快照

## 结论

截至 **2026-10-07 18:35:05（Asia/Shanghai）**，原队列完成 4/6 臂；
`A0_3103` 可见 31/40 个完整 epoch，`A1_3103` 尚未开始。所有正式
`E_joint` baseline、六臂五推理种子结果和三项机制消融都尚未落盘。
因此目前**不能回答整体联合模块、A1 修订或不可约交互是否有效，也不能执行
A1 的预注册通过门槛**。这不是失败判定，而是 `PARTIAL_QUEUE_RUNNING`。

目前唯一可量化的配对信号来自模型选择用的 seed-2035 validation，证据层级低于
正式结果。已完成两对中，A1 相对 A0 的均值变化为：JADE
`−0.0005226 m`（降低 `0.2625%`）、JFDE `−0.0022868 m`、ADE
`+0.0005268 m`、FDE `−0.0005294 m`、CRmean `+0.0005914`，即增加
**0.0591 个百分点**。方向虽为 2/2，但 JADE 数量级远低于预注册的
`0.01 m` 且 `2%` 双门槛；这组 validation 数字不得代替最终判定。

## 队列事实

原始截止时间仍是 `2026-10-08T13:17:16+00:00`，即北京时间
2026-10-08 21:17:16，没有重置。快照时 `QUEUE_PROGRESS.json` 指向
`A0_3103`，日志和 epoch-31 原子 checkpoint 在持续更新。记录中的 manager PID
和 worker PID 在本轮 managed 进程命名空间内不可见，因此本报告不把 PID 存在性
写成已独立认证；进度仅取原子收据、完整 epoch 文件与日志时间。未停止、恢复、
重启或接管任何进程。

| 臂 | objective | 状态 | 可见 epoch | 选择 epoch | validation JADE/JFDE | 更新 |
|---|---|---:|---:|---:|---:|---:|
| A0_3101 | `mean_energy` | 完成 | 40 | 36 | 0.199096/0.336022 | 139,360/139,360 |
| A1_3101 | `expected_conditional_mc`, S=4 | 完成 | 40 | 29 | 0.198380/0.331743 | 139,360/139,360 |
| A1_3102 | `expected_conditional_mc`, S=4 | 完成 | 40 | 32 | 0.198705/0.338530 | 139,360/139,360 |
| A0_3102 | `mean_energy` | 完成 | 40 | 12 | 0.199034/0.338825 | 139,360/139,360 |
| A0_3103 | `mean_energy` | 运行中 | 31 | 当前最优 17 | 0.198574/0.337690 | 未封存 |
| A1_3103 | `expected_conditional_mc`, S=4 | 待运行 | 0 | — | — | — |

“选择 epoch”由完整可见曲线重新按 `(JADE, JFDE, earlier epoch)` 排序得到，
与四份完成收据和 `best_model.pt` 元数据一致；没有查看最终测试后重选。

## 身份与完整性

- 实际运行源码为 `664ad0410f5cf3b44d3a6c1892c98e37aa936b9e`；目录后缀
  `5b6e4e4` 仅是 cache 来源，未被误当成运行源码。
- 唯一 parent 是官方 HOTEL epoch 110，文件 SHA256 为
  `5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b`。
- 四个完成 selected checkpoint 中，parent 的 112 个 frozen tensors 均逐张量
  `torch.equal`；GDTS、corrector 与 scheduler buffers 的冻结状态没有漂移。
- A0/A1 的 seed 3101 共享 state hash
  `c9d2d621…d947`，seed 3102 共享 `c3d2601a…d334`。预先归档的配置差异审计
  只发现 objective、config/run/save/model 路径字段，状态为 PASS。
- 四个完成 checkpoint 的 `jdv2_corrector.output.2.{weight,bias}` 都严格全零。
  源码在 Stage-A eval 且末层精确为零时执行 literal bypass；它没有扩展到训练。
- 所有完成臂失败/跳过更新均为 0；last checkpoint 为 epoch 40、可恢复，scheduler
  `gamma=0.995`、last LR `8.183201210226742e-05`。
- 两个完成 A1 的独立 CPU MT19937 objective RNG 都有 139,360 draw calls、
  5,672,960 sampled-agent draws、`pending_backward=false`，初始 seed 分别为
  103101/103102；effective loss dtype 是 FP32，backend 是
  `cuda_fp32_bf16_v1`。

完整逐文件路径、字节 SHA、mtime 与 checkpoint 元数据见
[INPUT_PROVENANCE.json](INPUT_PROVENANCE.json) 和 [RESULTS.json](RESULTS.json)。

## 三类正式比较

| 问题 | 所需证据 | 当前状态 | 当前可说结论 |
|---|---|---|---|
| E_joint baseline → A0/A1 | 同步窗口的 baseline 五种子及六臂 6×5 | PENDING | 不可判定 |
| A1 → A0 | 三个训练 seed 内先平均五个推理 seed，再做三对差 | PENDING | 仅有 2/3 选择验证趋势，不能过门槛 |
| full → pair-off/physical/interaction-off | A1_3101 selected 的三项正式干预与 native evidence | PENDING | 静态入口已定位，指标与实际运行证据不可判定 |

官方独立验证 `ADE/FDE=0.13394/0.19140 m` 属于 `E_official`，且官方 valid/test
原始字节相同。它没有 JADE/JFDE，也不是同步 `E_joint` evaluator，故仅在
[FINAL_METRICS.csv](FINAL_METRICS.csv) 中以 `REFERENCE_ONLY_NOT_E_JOINT` 单列。

原 A1 门槛保持不变：JADE 平均至少降低 `0.01 m` 且 `2%`、至少 2/3 对同向；
ADE/FDE/JFDE/CRmean 分别满足原 guardrails。目前门槛状态为
`NOT_EVALUATED_FORMAL_RESULTS_PENDING`，没有事后下调。

## 训练与机制线索

四个完成臂的 total loss 约从 3.03 降到 2.871–2.873，但 validation JADE 在各轮
明显振荡，最优 epoch 为 12、29、32、36，并不随 surrogate 单调改善。动态关系的
日志条件熵从约 1.01–1.13 降到 0.010–0.019；将日志中的四个 mode mass 在非零
总质量上重归一后，最大 mode fraction 达 0.9965–0.9982。与此同时 energy std
从约 0.10 增至 1.24–1.41，说明 cost 非常量。

这些只能支持“关系输出高度集中、energy 尺度增长”的趋势：日志没有 edge-level
概率，且普通 diagnostics 是 window-equal 聚合，E=0 window 会贡献零 usage；因此
不能严格计算同权重的 `H_cond` 与 `H_agg`，也不能把 mode 编号跨 seed 解释为
相同物理语义。没有 pair-cost tensor 时，energy std 更不能证明存在不可约交互。

## 分项科研判断

| 命题 | 判定 | 证据等级 |
|---|---|---|
| 整体联合模块有效 | 不可判定 | 正式 E_joint baseline/arms 缺失 |
| A1 修订有效 | 不可判定；已有 validation 效应很小 | 趋势 |
| learned 不可约交互有效 | 不可判定 | 机制结果与 cost payload 缺失 |
| 四关系模式保持多样性 | 未观察到；日志显示强集中 | 趋势，不等于严格全局 collapse |
| candidate/oracle 支持是瓶颈 | 不可判定 | Goal_minFDE 恒定只证明 bank 固定 |

## 审计边界

本轮新增 GPU、模型 forward、训练、diffusion、sampler、cache 导出均为 **0**；
安全 CPU checkpoint loads 为 13，CPU solver calls 为 0，人工可算数学参考 4 个且
全部通过。没有保存的 edge-level/native tensor，因此 H_agg、I/a/b、同 checkpoint
Jensen gap 与 assignment margin 均明确 `NOT_RUN_MISSING_NATIVE_PAYLOAD`。

推荐暂不改结构或开新训练。队列完成后用 [COMMANDS.md](COMMANDS.md) 的同一 CPU
汇总命令刷新；只有正式指标和机制证据满足 [NEXT_DESIGN.md](NEXT_DESIGN.md) 的
条件分支，才把 M=1 简化作为下一轮单变量提案。
