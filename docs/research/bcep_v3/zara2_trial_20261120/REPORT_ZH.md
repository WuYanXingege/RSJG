# BCEP–EBJD v3：唯一 ZARA2 Stage2 正式训练最终报告

**本折未同时通过三项门槛，保留旧 coordinate 基线。** 原生停止原因 `early_stop`；完整 **30轮**；有效真实optimizer更新 **5040**；记录到的物理step事件 **5040**（物理总数精确可核对：True）。唯一主结果按全体dev H-JADE选为 **epoch 12 / rank1**，没有从Top5改挑JFDE最低权重。

| 固定验收指标 m | 旧coordinate epoch5 | 新v3 rank1 | 新−旧 cm | 通过 |
|---|---:|---:|---:|---|
| 全体 H-JADE ≤ | 0.4097670185111337 | 0.4124557339160119 | +0.268872 | False |
| 全体 H-JFDE < | 0.7841791655426308 | 0.8049277728581328 | +2.074861 | False |
| N≥9 H-JFDE < | 1.2968434301886971 | 1.3103605776321230 | +1.351715 | False |

三项按用户预先声明的归档未舍入值判断，必须同时通过。旧正式rank1 epoch26仅作背景：H-JADE 0.4084707879413993 m、H-JFDE 0.8016974424154072 m、N≥9 H-JFDE 1.3188941478729248 m。完整v3一次训练只能验证组合效果，不把任何变化直接归因于单个模块。

## 研究问题的明确回答

1. 是否同时通过全部门槛：**False**；每项差值见首页。
2. 初配和终配如何变化：下列均为同窗口新减旧，误差负值表示改善，反馈收益正值表示帮助增加。
- all：初配I-JFDE新−旧 +0.326496 cm；反馈收益I−H新−旧 -1.748365 cm；最终H-JFDE新−旧 +2.074861 cm。新模型平均反馈收益为 -1.254817 cm，旧模型为 +0.493548 cm。
- multi：初配I-JFDE新−旧 +0.384496 cm；反馈收益I−H新−旧 -2.058952 cm；最终H-JFDE新−旧 +2.443449 cm。新模型平均反馈收益为 -1.477728 cm，旧模型为 +0.581224 cm。
- N_ge_9：初配I-JFDE新−旧 -0.756039 cm；反馈收益I−H新−旧 -2.107752 cm；最终H-JFDE新−旧 +1.351714 cm。新模型平均反馈收益为 -3.194520 cm，旧模型为 -1.086768 cm。
3. HCRA反馈是否正向：只能直接判断“旧反馈+HCRA增量”的合成反馈。多人组I−H均值为 **-1.477728 cm**，因此合成反馈在多人组为 **负向**。正值说明最终配对平均优于初配，负值说明平均抵消。该数值不能隔离HCRA本身的因果贡献。
4. CSEE是否得到真实参数更新：**True**；实际step梯度/增量和rank1相对初态变化见下表，零初始化输出权重的非零变化排除了仅凭其他权重衰减推断学习。梯度小不作为失败条件。
5. GRG-CTF启用后的组合模型完整路径JADE是否改善：**退步**。全体新−旧 **+0.268872 cm**，N≥9 **-0.611294 cm**；负值为改善。这是完整v3的结果，不能单独归因于GRG-CTF。
6. 可直接证明的是组合模型指标、配对分解、三模块实际梯度/参数更新、候选边缘保持和完整恢复状态的保存；CSEE、HCRA、GRG-CTF各自独立效果、协同贡献、跨seed稳健性与跨折泛化均未被本轮证明。没有追加消融或测试集评测。

## 同窗口配对分布

差值Δ=新−旧，I/H/JADE负值是改善，反馈收益正值是改善；按严格零统计相等，另在CSV列出|Δ|≤1e-6数量。all 590窗、multi 501窗、N≥9 127窗；没有使用行人数加权替代场景等权。

| 分组 | 指标 | 窗数 | 均值差 cm | 中位数差 cm | 改善窗 | 退步窗 | 相等窗 |
|---|---|---:|---:|---:|---:|---:|---:|
| all | H_JADE | 590 | +0.268872 | +0.483498 | 281 | 309 | 0 |
| all | I_JFDE | 590 | +0.326496 | +0.000000 | 260 | 236 | 94 |
| all | H_JFDE | 590 | +2.074861 | +0.000000 | 210 | 281 | 99 |
| all | feedback_gain | 590 | -1.748365 | +0.000000 | 230 | 270 | 90 |
| multi | H_JADE | 501 | +0.340072 | +0.571644 | 237 | 264 | 0 |
| multi | I_JFDE | 501 | +0.384496 | -0.770009 | 260 | 236 | 5 |
| multi | H_JFDE | 501 | +2.443449 | +2.086532 | 210 | 281 | 10 |
| multi | feedback_gain | 501 | -2.058952 | -1.780301 | 230 | 270 | 1 |
| N_ge_9 | H_JADE | 127 | -0.611294 | -0.076377 | 64 | 63 | 0 |
| N_ge_9 | I_JFDE | 127 | -0.756039 | -1.360190 | 67 | 60 | 0 |
| N_ge_9 | H_JFDE | 127 | +1.351714 | +2.086532 | 51 | 76 | 0 |
| N_ge_9 | feedback_gain | 127 | -2.107752 | -0.960231 | 58 | 69 | 0 |

逐窗检查 ΔH=ΔI−Δ(I−H) 在1e-6 m内成立。与旧epoch5复用完全相同候选bank时，逐人minFDE和逐窗oracle差均小于1e-6 m；全体边缘minFDE仍为0.3520422351797007 m、逐窗等权oracle为0.33942601943420153 m。固定端点下桥不能直接改变JFDE。

旧epoch5匿名逐窗结果从固定归档 `76d80b508ad79f52965c15109b58d2d835473fd7` 读取，核验CSV/权重/split/bank身份，复用已归档20步开发集推理，本轮没有再次训练旧模型。旧参考聚合与门槛的浮点差记录在REFERENCE_IDENTITY.json；门槛判断仍使用用户归档值。[匿名逐窗差值](RANK1_VS_OLD5_WINDOWS.csv) · [配对汇总](PAIRED_SUMMARY.csv)

## 模块梯度与参数更新

| 模块 | 非零梯度step数 | 非零参数增量step数 | rank1相对初态参数L2 | 零初始化输出权重变化L2 |
|---|---:|---:|---:|---:|
| coupling.csee. | 5039 | 5039 | 0.226174349 | 0.219862898 |
| coupling.hcra. | 5040 | 5040 | 0.631598026 | 0.00078263898 |
| bridge.denoiser.grg_ctf. | 5040 | 5040 | 1.29805419 | 0.119549074 |

来自已验收入口的实际AdamW step后hook；梯度是裁剪前范数。各epoch统计见[EPOCH_MODULES.csv](EPOCH_MODULES.csv)，不以某个范数小直接判模型失败。CSEE初始下游零出口造成第一步无梯度是已验收行为。若技术恢复重放更新，逐轮统计按最终有效update取最后事件，物理事件总数单列。

## 固定协议、耗时和保留

正式训练源码固定提交 `2e96b6f35554ceceb5a04fc08b6259dac75e486b`，配置规范化SHA `294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43`。实际启动只使用已验收CONTROLLED_COMMAND；准备态allow_real_training=false归档保持原样，本次用户授权独立记录于主机AUTHORIZATION/EXECUTION_AUTHORIZATION。

Provider density epoch19、head epoch9冻结；train 2681窗、dev590窗（N≥9 127），candidate seed2035、训练随机初态seed3201。CSEE+HCRA+GRG-CTF全部启用，final_coordinate旧有效损失、6层桥、K20、20步DDIM、FP32、AdamW、5轮warmup/100轮调度、168更新/epoch、minimum30/patience15/max100均未更改。没有重训Provider、改变终点、增加注意力层或调整权重。

原生计时合计：训练 **7.832小时**，逐轮验证 **2.469小时**。计时含原控制器定义的范围，额外启动/保存开销不伪装成纯GPU时间。逐轮详情：[EPOCH_GROUPS.csv](EPOCH_GROUPS.csv)、[EPOCH_RESOURCES.csv](EPOCH_RESOURCES.csv)。

显存旁路每5秒读取本次PID的nvidia-smi进程used_memory，记录driver/context/reserved口径的采样峰值，同时记录RSS。**固定入口未采集PyTorch allocated峰值；旁路采样可能漏掉瞬时峰值，阶段边界为最后step与原生epoch日志近似划分。** 未修改冻结训练源码来插入计时或额外前向，不能把采样显存称为精确分配器峰值。资源口径和进程尝试记录见RESOURCE_PROTOCOL.json。

保留全部 **30个不同完整epoch**、best、可恢复last及按dev H-JADE排序的Top5 **[12, 25, 10, 22, 23]**；逐个核验完整模型/optimizer/scheduler/RNG/数据游标与选择状态、epoch更新计数和SHA。[ALL_CHECKPOINTS.csv](ALL_CHECKPOINTS.csv) · [TOP5.json](TOP5.json)。权重与候选库保留训练主机。旧v2/candidate_cost/sharedworld共277个保护文件大小与SHA未变，旧队列没有恢复。

## 复现与停止边界

```bash
RUN_ONE_ZARA2_V3=true OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python -m bcep_v3.formal train --frozen-sha 294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43
```

本次已经COMPLETE，相同命令会返回，不另起训练。技术中断仅从同一运行last恢复，未落盘更新可能重放；若发生恢复，不能只根据checkpoint有效计数推断所有物理step，物理事件记录与精确性已在首页分列。

本轮结束后不启动第二seed、其他折、模块消融、测试集、旧五折、candidate_cost或sharedworld。三项全部通过也只支持本折后的独立审阅决定；否则保留旧coordinate基线。当前结论不推广为跨折性能保证。
