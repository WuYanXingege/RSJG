# v2 入口

在代码提交 `04ecb3b0ea8ab347f95e36675cc90972b4f1139b` 的独立工作区运行。使用项目已有 Python 环境。`PLAN` 指向正式 JMM 五折的只读 RUN_PLAN.json，`FOLD` 取其官方折名。输出自动落在既有实验的同级独立 v2 目录。配置默认关闭真实执行。

## 当前可以执行的只读检查

```bash
python -m bcep_v2.workflow inspect --fold zara2 --plan "$PLAN"
CUDA_VISIBLE_DEVICES= python -m bcep_v2.check_interfaces --plan "$PLAN"
```

第一条核对拆分、成员、坐标与源码。第二条只读 ZARA2 的一人、两人各一个窗口和已冻结旧权重，运行不涉及更新的兼容检查。合成测试可由 `python -m pytest -q tests/test_bcep_integrated_v2.py` 单独复现，恢复用例自身执行 8 次合成更新。

## 未来获得训练授权后

授权副本 `V2_CONFIG` 应与本次已解析配置一致，仅将顶层 `allow_real_training` 设为 true。该配置和显式 `--execute` 同时生效。三项 `*_CAP` 都是该阶段获准达到的**累计绝对更新数**；按折的每轮更新数在配置中列出。以下为单阶段入口示例，本次没有执行这些命令。

```bash
COMMON=(--plan "$PLAN" --config "$V2_CONFIG" --fold "$FOLD" --device cuda)
python -m bcep_v2.workflow density "${COMMON[@]}" --until-update "$DENSITY_CAP" --execute
python -m bcep_v2.workflow base-cache "${COMMON[@]}" --execute
python -m bcep_v2.workflow head "${COMMON[@]}" --until-update "$HEAD_CAP" --execute
python -m bcep_v2.workflow banks "${COMMON[@]}" --execute
python -m bcep_v2.workflow joint "${COMMON[@]}" --until-update "$JOINT_CAP" --execute
python -m bcep_v2.workflow evaluate "${COMMON[@]}" --execute
```

按顺序单独执行，每个训练阶段返回 COMPLETE 并持久化冻结收据后进入下一阶段。会话预算只允许达到一个中间更新位置时，用同一阶段、同一配置和 `--resume` 恢复；有新增预算时传入新累计上限。SIGTERM/SIGINT 请求在逻辑更新边界保存；收到信号时正在执行的验证会先结束。突然断电可恢复至最近已落盘断点。模型、AdamW、学习率、排名、游标和所有已登记的 RNG 一起恢复。相同已用尽的上限不会提交第二次更新。

基础候选缓存按两个训练视图和五个验证 seed 分开。精修 bank 只有经过 A、B 两次选模冻结后才能写入；联合权重冻结后才允许生成留出测试 bank。阶段末尾检查 `TOP5.json`、`FROZEN.json`、`EPOCH_METRICS.json`；最终指标位于 `evaluation/PER_SEED_METRICS.csv`。真实 GPU 容量、速度与精度仍待单独验证。
