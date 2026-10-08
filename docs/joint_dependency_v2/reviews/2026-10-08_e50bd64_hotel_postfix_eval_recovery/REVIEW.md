# HOTEL A0/A1 正式评测 loader 修复与 post-only 恢复

状态：`TRAINING_COMPLETED_POSTFIX_EVALUATION_RUNNING`

快照时间：2026-10-08 09:28:53 +08:00

原正式训练源码：`664ad0410f5cf3b44d3a6c1892c98e37aa936b9e`

修复与恢复源码：`e50bd64f256dfb38e8457dc5fd048da0fcf129d2`

## 结论

六个 A0/A1 训练臂均已完成 40 epoch，训练收据全部为 `COMPLETED`。原队列在第一个 post-task `E_joint_baseline` 失败；失败发生在构造 loader 时，尚未产生 baseline 指标，也未启动后续 A0/A1 正式测试或机制消融。

根因是 resolved Stage-A YAML 中 `jdv2_active: null` 依赖主 CLI 做归一化，而独立 baseline evaluator 直接读取 YAML。loader 因而选择不存在的 legacy `data_batches/test_batches`，没有选择已经冻结且由 A0/A1 使用的 `data_batches_jdv2_v2/test_batches`。

修复只在 baseline evaluator 构造 loader 前显式设置 `loader_args.jdv2_active = True`。baseline 模型本身仍设为 `goal_model_type=independent`，JDV2、dynamic relation、joint energy、corrector、coupling 和 alignment 仍全部关闭；因此修复只对齐外部同步窗口，不改变原生 GDTS 推理语义。

## 失败证据与保护

- 原 manifest SHA256：`530d210f744b1576c69fe1f13d23441c1cf8517e5242d9bdb22317adb4b4ec19`。
- 原失败进度 SHA256：`9f98461552c4e799d441ac417cd4e29a9793b5976aba65dfe113555e5d073331`。
- 原 baseline 失败收据 SHA256：`ee044ee102a817483a32d621495f668cc64f761f3bd752e9525c73ba8a5d3036`。
- 原失败进度保存为 `QUEUE_PROGRESS_PRE_POSTFIX.json`；原失败收据保存为 `E_joint_baseline.pre_postfix.json`，均未覆盖。
- post-only 恢复器逐一核验六份训练收据为 epoch 40、return code 0，并重新计算每个 checkpoint SHA256。
- 恢复器不调用训练 worker，不重新进入任何训练臂。

## 验收与启动

路径探针确认：

- 修复前：`.../data_batches/test_batches`，不存在；
- 修复后：`.../data_batches_jdv2_v2/test_batches`，存在，共 445 个同步窗口。

post-only preflight 返回 `POSTFIX_PREFLIGHT_PASS`，认证六个完成臂、原失败状态、原 manifest 和新源码提交。恢复队列于 2026-10-08 09:27:33 +08:00 后台启动：

- manager PID：`1192288`；
- 当前 baseline worker PID：`1192316`；
- GPU 显存：504 MiB；快照利用率 22%；
- loader 已打印 `Test dataset contains 445 data batches.`；
- 当前未见 traceback 或新合同失败；
- 保留原总截止时间：2026-10-08 21:17:16 +08:00。

## 不变合同

- 六个 selected checkpoint、训练结果和收据不变；
- 推理种子仍为 2035--2039；
- P20、窗口、坐标、mask、指标与聚合规则不变；
- post-task 顺序仍为 E_joint baseline、六个 A0/A1 正式评测，以及 `off`、`physical`、`pure_interaction_off`；
- 不增加训练、不进行超参数搜索、不删除或覆盖负结果。

当前只是恢复成功的运行快照，不是最终结果。正式收益仍须等待所有 post-task 收据和 `RESULTS.json` 完整生成后判定。

机器可读启动记录见 [RELAUNCH_RECEIPT.json](RELAUNCH_RECEIPT.json)。
