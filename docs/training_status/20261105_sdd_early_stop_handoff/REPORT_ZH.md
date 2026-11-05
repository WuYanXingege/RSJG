# SDD Goal-pretrain 提前结束与早停交接报告

## 结论

SDD Goal-pretrain 已按用户授权在平台期提前结束，状态是正常完成而非故障。最后完整训练轮为 epoch 81/150；epoch 82 仅开始 2 个 batch，未形成完整 checkpoint，已放弃且未计入完成轮数。matched GDTS 已由单实例后台流水线接管，使用冻结的验证最佳 Goal 权重初始化。

## Goal-pretrain 冻结资产

- 完成原因：`USER_APPROVED_EARLY_STOP_PLATEAU`
- 完整停止轮：81/150
- 放弃的下一轮部分：epoch 82 的 2 个 batch
- 选中权重：`goal_pretrain_best_model.pt`
- 选中 epoch：35
- validation Goal BCE：0.0028039472735674562
- 完整 SHA256：`6f000e5c6f1642dd959107d1908e8b041706f09db1a135bc4f6a43d668145a88`
- epoch 81 last SHA256：`68f2241df97993f00ee16fba1a45a291b00b9949fdcb93a052074dae3a3bb1e0`
- strict-load：PASS

## matched GDTS 早停合同

- 最大轮数：250
- 原有 selector 与监控标量：validation ADE（`valid_ADE`），像素单位，越低越好
- 验证频率：每 10 epoch 一次
- 最早允许停止：epoch 50
- patience：20 次完整验证
- 有效相对改善阈值：0.001（0.1%）
- 优化器：Adam；学习率计划保持旧入口的 ExponentialLR，gamma=0.995
- 真实最佳与重置 patience 的有效改善分别记录
- early-stop 状态、optimizer、scheduler 与 RNG 写入 last checkpoint；旧 checkpoint 优先由完整验证曲线重建 early-stop 状态
- 只有原子完成收据存在且哈希匹配时，流水线才会推进到正式评测，避免中断时仅因 best 权重存在而误判完成

## EBJD 早停合同

- 最大轮数：30；最早允许停止：epoch 10
- patience：8 次完整验证；有效相对改善阈值 0.001
- 首次 eligible 前监控完整验证 H-JADE
- 首次 eligible 时切换阶段、用该轮 H-JADE 初始化基准并清零 patience
- eligible 阶段仅 eligible H-JADE 的有效改善重置 patience；ineligible 验证计作 stale
- 正式 selector 不变：优先 `best_eligible`，否则沿用 `best_jade` fallback 并明确合同未通过
- rollout 与 paired actual-step projection 保持关闭

## 验收与运行状态

- 定向测试：8 项通过
- Python 编译检查：通过
- `git diff --check`：通过
- 单实例锁：已接入；同一实验目录不能并行启动第二个 supervisor
- 后台 supervisor PID：920981
- matched GDTS PID：921007
- 源码绑定：`444487d1ccbcfbf748cc9920f73fa12341ad5370`
- 当前阶段：MATCHED_GDTS
- 首批 update：PASS；epoch 1 已推进到至少 38/2544 batches。旧 trainer 在每个 optimizer step 前检查 total loss 与全部梯度为有限值，进程持续运行且未触发异常。
- Goal 初始化核验：运行日志记录 strict Goal U-Net epoch=35，SHA256 与冻结收据逐值一致
- 首批训练显存：约 12952 MiB（RTX 5070 Ti 16 GB）；仅一个 GPU 训练进程
- 记录时磁盘可用：221,910,835,200 bytes（约 206.7 GiB）

运行日志：

`RSJG_EBJD_assets/experiments/sdd_jmm_unified_coupling_bridge_v1_20261103_155713/logs/matched_gdts.log`

资产收据：

`RSJG_EBJD_assets/experiments/sdd_jmm_unified_coupling_bridge_v1_20261103_155713/03_goal_pretrain/GOAL_PRETRAIN_COMPLETION.json`
