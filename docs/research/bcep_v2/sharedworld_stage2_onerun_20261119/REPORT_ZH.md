# ZARA2 shared_world_bounded：一次 Stage 2 受控训练

**本折未通过，保留旧 coordinate 基线。** 停止原因 `early_stop`，完整 **30 轮**，有效真实更新 **5040**，唯一主结果为按 dev H-JADE 选出的 **epoch 12 / rank1**。训练 7.21 小时、逐轮验证 2.22 小时；原五折与 candidate_cost 保持暂停。没有测试集选模、第二个训练 seed、额外折或后续队列。

| 预先固定指标（m） | 旧 epoch 5 | 新 rank1 | 新−旧（m） | 新−旧（cm） | 通过 |
|---|---:|---:|---:|---:|---|
| 全体 H-JADE ≤ | 0.4097670185111337 | 0.4116103804591348 | +0.001843362 | +0.184336 | False |
| 全体 H-JFDE < | 0.7841791655426308 | 0.8017813523757761 | +0.017602187 | +1.760219 | False |
| N≥9 H-JFDE < | 1.2968434301886971 | 1.3138810612085297 | +0.017037631 | +1.703763 | False |

三个门槛均来自旧 epoch 5 的归档未舍入数值；必须同时通过。微小差值只能作为本折观测，不能称稳定提升。Top5 只按 dev H-JADE 排序，没有改取 JFDE 最低权重。

all：初配 I-JFDE 新−旧 **+2.9212 cm**；邻居反馈收益变化 **+1.1610 cm**（正为反馈更有帮助），最终 H-JFDE 新−旧 **+1.7602 cm**。分解为 ΔH=ΔI−Δ反馈收益。完整轨迹 H-JADE 新−旧 **+0.1843 cm**（负为改善）。该分解描述已训练模型间的观测，不把损失标量系数当作参数梯度比例。

N_ge_9：初配 I-JFDE 新−旧 **+4.7005 cm**；邻居反馈收益变化 **+2.9967 cm**（正为反馈更有帮助），最终 H-JFDE 新−旧 **+1.7038 cm**。分解为 ΔH=ΔI−Δ反馈收益。完整轨迹 H-JADE 新−旧 **-0.7348 cm**（负为改善）。该分解描述已训练模型间的观测，不把损失标量系数当作参数梯度比例。

## 匿名逐窗配对分布

旧 epoch 5 缺少逐窗完整 H-JADE，故只读补算一次相同开发集的 20 步 DDIM，并从同一前向提取 I/H。没有补训旧模型；旧参考与归档的数值误差记录于 REFERENCE_REPLAY.json。Δ=新−旧；I/H/JADE 的负差为改善，反馈收益的正差才是改善。计数按严格零划分，另列 |Δ|≤1e-6 的数量作为浮点近零提示。

| 分组 | 指标 | 窗数 | 平均差 cm | 中位差 cm | Δ<0 | Δ>0 | Δ=0 | 近零 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | I_JFDE | 590 | +2.92123 | +0.87389 | 187 | 304 | 99 | 99 |
| all | H_JFDE | 590 | +1.76022 | +0.00000 | 236 | 260 | 94 | 94 |
| all | H_JADE | 590 | +0.18434 | +0.08660 | 292 | 298 | 0 | 0 |
| all | feedback_gain | 590 | +1.16101 | +0.00000 | 238 | 260 | 92 | 94 |
| N_ge_9 | I_JFDE | 127 | +4.70045 | +5.75845 | 44 | 83 | 0 | 0 |
| N_ge_9 | H_JFDE | 127 | +1.70376 | +1.67445 | 53 | 74 | 0 | 0 |
| N_ge_9 | H_JADE | 127 | -0.73485 | -0.21121 | 64 | 63 | 0 | 0 |
| N_ge_9 | feedback_gain | 127 | +2.99669 | +2.03623 | 57 | 70 | 0 | 0 |

逐窗结果：[RANK1_VS_OLD5_WINDOWS.csv](RANK1_VS_OLD5_WINDOWS.csv)；逐轮分组：[EPOCH_GROUPS.csv](EPOCH_GROUPS.csv)；训练目标与系数日志：[EPOCH_TRAIN.csv](EPOCH_TRAIN.csv)。固定候选下逐窗 minFDE/窗口 oracle 新旧差均小于 1e-6 m，每轮全体 agent-micro minFDE 核对为 0.3520422351797007 m、窗口等权 oracle 为 0.33942601943420153 m（容差 1e-6）。

## 历史参照与选择边界

| 模型 | dev H-JADE | dev H-JFDE | dev N≥9 H-JFDE |
|---|---:|---:|---:|
| 旧 coordinate 正式 rank1 / epoch 26 | 0.4084707879413993 | 0.8016974424154072 | 1.3188941478729248 |
| 旧 coordinate epoch 5 | 0.4097670185111337 | 0.7841791655426308 | 1.296843430188697 |
| 旧 coordinate epoch 3，仅上下文 | 0.4218764735164784 | 0.7792569560204017 | 1.2382272847994107 |
| 新 shared_world_bounded / epoch 12 | 0.4116103804591348 | 0.8017813523757761 | 1.3138810612085297 |

这些是开发集结果，未使用历史 test 数值作门槛，也没有本轮测试集评测。一次训练仅检验该目标相对旧基线的整体效果，不声称动态系数优于未实测的固定混合。

## 公平对照、身份与保留

报告基点 `7cea557556386d9fbc6e905e02cf3b24640a8d03`，已验收实现 `f17571c6836287408514736855e156fd5b825828`，运行代码 `b53185d5ed0541011f733f807ec360fea20e2033`。原实现文件逐项哈希匹配审计；只增加单折入口，模型/损失未再修改。配置 SHA256 `d8cfc7679f90ed46031b39b98eb59be62cb7d5f6ef67334fd9fe7f6ce5f40cc3`。审计档案保持 RUN_ONE_ZARA2=false；本运行独立授权 true，上限 16800。

Provider/候选修正头沿用原 dev rank1，seed 2035 的 train/dev bank、成员与顺序哈希通过；仅 ZARA2 train 2681、dev 590（其中 N≥9 127）窗。训练 seed **3201**，候选/验证 seed **2035**。初态参数 hash `fcd1967d91712784ee963948b76fa3ba05c812224c81eeb0023db0751dc062c7` 与旧 coordinate 完全相同，无旧权重微调。首个真实 step 的损失与梯度范数和旧首步完全相等，优化器计数及参数变化见 [FIRST_UPDATE_VERIFICATION.json](FIRST_UPDATE_VERIFICATION.json)。

沿用原 joint_callbacks/run_stage：共享编码、配对头及六层桥；FP32、math SDP、原增广和数据顺序；微批 4、累积 4、尾批原除数，AdamW lr 1e-4、weight decay 1e-4、clip 5；5 轮 warmup、100 轮调度、minimum 30、patience 15、最大 100 轮。τ=0.10 m、a≤0.5、pair=1，其他损失原值。验证只跑一次原 20 步 DDIM，同时捕获初配端点，不额外跑一遍配对或轨迹。

完整保留 30 个不同完成轮次、best 与可恢复 last；[TOP5.json](TOP5.json) 按 dev H-JADE、同分早轮排序，文件与 SHA256 可核验。Top5 轮次为 [12, 24, 4, 28, 26]。完整状态检查包括 optimizer、scheduler、RNG、数据游标、早停与统计分子分母。全部 checkpoint/候选库留在主机；公开包仅相对文件名、哈希和匿名统计。新入口冒烟 0 次真实/合成优化器更新；沿用上一轮 15 项验收。原五折 update 7097、candidate_cost update 1903 的 last、状态、Top5 哈希保持不变。[PRESERVATION.json](PRESERVATION.json)

训练峰值 allocated 2.373 GiB，验证峰值 1.292 GiB。旧参考补算 269.75 s，不计作训练更新。原生结束后退出；通过则仅支持本折后的独立验证决策，未通过则保留旧 coordinate 基线，本程序不启动任何后续任务。

## 复现

在本次代码 checkout 与原主机环境中设置 WORKSPACE 为资产所在工作区。入口路径限定为本实验根目录；同一身份重复 train 调用只恢复本次 last.pt，已 COMPLETE 则直接返回，不会另起训练。先确认数据与 RUN_ONE_ZARA2=true 授权仍有效。

```bash
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/shared_world/one_run.py prepare --workspace "$WORKSPACE"
python scripts/shared_world/one_run.py train --workspace "$WORKSPACE"
# 已启动/中断时仅执行上一条 train；不要对已启动运行再 prepare。
python scripts/shared_world/one_run_report.py --workspace "$WORKSPACE"
```

原五折和 candidate_cost 必须保持暂停。入口同时持有本运行锁与两条旧运行的控制锁，结束后释放；运行身份不符会拒绝恢复。训练源码与报告提交身份分开记录，最终发布以固定 GitHub SHA 为准。
