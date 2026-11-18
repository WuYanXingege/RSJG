# BCEP–EBJD v2 五折正式运行合同

采用 2026-11-18 两大阶段整理版。工程验收源码为 `51cdc5e84c3601b75482efeebea374f7ffa260f9`，验收报告快照为 `d371c8903bd48470f8c140191873b91f1e04110f`。五折按 ZARA2、ZARA1、HOTEL、UNIV、ETH 的顺序串行。每折独立初始化，模型只在该折官方 train 上拟合。官方 dev 决定调度、早停与排名。项目早期开发曾参考既有 ZARA2 测试诊断，本轮训练规则、配置和测试矩阵在新一轮测试前冻结。

| 整体阶段 | 既有子流程 | 最少 / 最多完整轮数 | 早停耐心 | 排名依据 |
|---|---|---:|---:|---|
| 1A · Provider 密度 | density | 20 / 100 | 15 | dev 终点 NLL，有效行人加权 |
| 1B · Provider 候选校准 | head | 5 / 20 | 5 | dev 有效行人 minFDE，五推理种子等权 |
| 2 · 联合轨迹 | joint | 30 / 100 | 15 | dev H-JADE |

1A 后用选中的 density 生成基础候选。1B 冻结该 density，仅训练 0.5 m 有界修正头。Stage 1 需完成两个优化子流程并核验精修 train/dev bank。Stage 2 冻结完整 Provider，联合学习配对、单轮反馈与六层条件桥。两次缓存操作的 optimizer 更新数均为零。三个优化子流程分别记录自己的 epoch 和 update。

| 折 | 训练窗口 | density 更新上限 | head 更新上限 | joint 更新上限 |
|---|---:|---:|---:|---:|
| ZARA2 | 2681 | 67100 | 13420 | 16800 |
| ZARA1 | 2889 | 72300 | 14460 | 18100 |
| HOTEL | 3118 | 78000 | 15600 | 19500 |
| UNIV | 2719 | 68000 | 13600 | 17000 |
| ETH | 3283 | 82100 | 16420 | 20600 |

逻辑 batch 为 4 / 4 / 16 个场景，各折上限合计 533,000 次。density/head 训练种子为 3102，joint 为 3201。四层 Flow 生成 M256，使用已验收的 KMeans20。硬排列同步作用于终点和桥噪声，联合桥使用 20 步 DDIM。8 帧观测、12 帧未来、间隔 0.4 秒，坐标单位为米。使用 PyTorch 2.8.0+cu128、FP32、严格确定性算法和 math SDPA。基础配置 SHA256 为 `0d4034a54443a40f4b046a8e88a423c64f2b7d3ad4e7f14f2b7de0d0be535985`。正式授权副本仅将顶层开关设为 true，嵌套开关维持 false。

## 留存与测试

每个优化子流程保留全部完整 epoch、best、可恢复 last、逐轮指标和 dev Top5。排序同分取较早完整 epoch。对外 Provider Top5 由固定 density dev rank1 与五个不同 epoch 的 head 组合；Joint Top5 绑定相同的 Provider dev rank1 和精修 bank。每项索引可定位源权重、上游、配置、成员和 SHA256。

统一冻结五折后，先执行所有正式 rank1 的 G/H，再执行诊断。推理种子 2035–2039，K=20。正式 G 共 25 组，H 共 125 组（其中 25 组复用正式 rank1），Provider 终点评估共 125 组。每组覆盖整折测试成员。只读加载非首位权重，同折同种子共享候选。非首位 joint 仅运行 H。Provider 复用确定性基础 K20、单窗口编码并逐值核对 rank1 与正式候选。生成端仅接触观测和地图；预测落盘后由评分端读取未来。所有正式表始终用 dev rank1。测试后的赢家单列，四项指标来自同一权重。

minADE/minFDE 按有效行人，JADE/JFDE 按场景。跨折先按每个推理种子计算五折等权平均，再对五个均值计算样本标准差。固定精修 K20 的 scene_weighted_minFDE 给出真值辅助排列的联合终点下界，coordination_gap_FDE 衡量 H 与此下界的距离。相应 ADE 分解仅适用于已生成的路径。公开报告区分推理波动与独立训练重复。

## 运行与恢复

在验收环境及已提交的源码工作区设置以下变量。RUN_PLAN 指向既有 JMM 官方计划，REPORT_CHECKOUT 是独立报告分支的工作区。准备命令打印正式输出根目录，将其赋给 RUN_ROOT。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m bcep_v2.formal_run prepare --plan "$RUN_PLAN" --report-checkout "$REPORT_CHECKOUT"
python -m bcep_v2.formal_run start --root "$RUN_ROOT"
python -m bcep_v2.formal_run status --root "$RUN_ROOT"
```

单一后台控制器建立独立会话，控制器和子进程共享同一文件锁。核对 STATUS.json 中的 PID 和命令后，可向该控制器发送 SIGTERM。退出码 75 表示可恢复暂停，其他非零退出写明故障。恢复入口如下。

```bash
python -m bcep_v2.formal_run resume --root "$RUN_ROOT"
```

恢复逐项核对已完成子流程的收据，在原 fold 和实际 substep 继续。训练读取本阶段 last.pt 中的优化器、scheduler、RNG、游标、部分轮次统计及选模状态。缓存和测试复用已校验的行收据。折完成和测试完成时由独立报告工作区推送小型结果；传输状态保存在 publishing 目录。公开报告分支为 `report/bcep-ebjd-v2-formal-top5-20261118`。
