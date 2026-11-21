# SSEC Energy-only 独立合成非零更新恢复验收

**结论：PASS_SYNTHETIC_NONZERO_RECOVERY。** 连续4步、2＋2恢复以及更新日志重放的完整数值状态逐值一致，比较容差为0。没有未解决的本轮工程阻塞。

本轮共实际执行 **11次合成 AdamW 更新、0次真实数据更新**：连续路4次，正常断点路2＋2次，故障重放另外1＋2次。三条路径的最终逻辑更新均为4。[逐次实际更新事件](OPTIMIZER_EVENTS.jsonl)保留全部11次调用。**未启动SSEC正式ZARA2训练，没有完成任何训练epoch、产生选模结果或启动其他实验。**

基点：`ece95ce6f1f59d58a8119fef7cdd79bf9d04ee86`。所有新增代码仅在独立worktree的`scripts/ssec_nonzero/`及审计测试/报告目录；网络、原损失、优化器、恢复控制器和正式冻结配置均未修改。

- [机器收据](READINESS.json)、[独立审计配置](AUDIT_CONFIG.json)、[合成输入收据](SYNTHETIC_INPUTS.json)
- [完整状态逐项对照](FULL_STATE_COMPARISON.csv)、[逐步一致性](PER_UPDATE_EXACT.json)、[实际梯度和更新](MODULE_GRADIENTS_AND_UPDATES.csv)
- [日志重放收据](JOURNAL_REPLAY.json)、[端点与DDIM噪声](DDIM_IDENTITY.json)、[资产保护](ASSET_PROTECTION.json)
- [复现命令](REPRODUCE.md)、[审计主程序](../../../../scripts/ssec_nonzero/audit.py)、[GPU执行程序](../../../../scripts/ssec_nonzero/worker.py)

## 1. 身份、数据和授权范围

SSEC Energy-only、无CSEE、FP32、原6层条件桥、K20、20步DDIM、proposal-ST、final_coordinate及全部损失系数保持不变。使用原AdamW参数分组、betas=(0.9,0.999)、eps=1e−8、基础学习率1e−4、weight_decay=1e−4/0，原clip=5和5轮warmup/100轮调度。4步均处于epoch1，实际学习率2e−5。

原正式冻结配置文件SHA256仍为`3ff36b653703307be4f38457d9cb9395c264ea7647c0df339913376f38b9d842`。初态指纹、7,627,950个参数及全部6个buffer均来自同一seed3201初始化，没有载入历史训练后权重。

合成数据由固定独立命名空间生成，观测8帧、未来12帧、dt=0.4s、6×256×256合成语义图；候选只用合成观测运动与独立随机偏移构造。全程没有把真实ZARA2或旧bank数据送入优化器。

保留2681个**合成索引**，从而保持原每轮168更新的控制器长度；只实例化本轮需要的64个独立样本。native seed3201 shuffle的前四个逻辑批次，每批人数为`[1,2,9,57,2,2,2,2,2,2,2,2,2,2,2,2]`。每批16窗，原4窗微批×4累计、group_max_slots=32；57人场景按原规则单独处理，不降低人数或地图容量。

GPU worker安装跨工作区文件访问拒绝器，仅可访问本工作区及运行库；其他RSJG资产/工作区均不能作为输入来源。历史资产哈希由独立父进程只读核验。没有设置`RUN_ONE_ZARA2_SSEC=true`，没有调用正式训练入口。

## 2. 连续与恢复实验

1. 连续路：一个新进程运行原`bcep_v2.state.run_stage`至update4。
2. 正常断点路：新进程运行至update2，原子保存完整`last.pt`；另一进程恢复至update4。
3. 恢复前刻意扰动目标模型参数及CPU/CUDA/Python/NumPy RNG，随后完全由原严格恢复路径重置；因此结果一致并非因为目标实例碰巧保留原状态。
4. 额外故障路：复制本次合成update2 checkpoint，在真实update3与原`MODULE_UPDATES`落盘后、checkpoint提交前注入异常；该update3计入实际执行次数。原`last.pt`字节哈希仍为update2。再从它恢复，重放update3并执行update4。

每条路径均复用原分组损失、backward、梯度裁剪、AdamW、scheduler、checkpoint持久化与恢复代码；没有手工模拟参数更新或AdamW动量。

## 3. 严格一致性结果

连续、2＋2、重放三路最终原生checkpoint的 **23个顶层字段全部逐值一致，无跳过字段**。模型参数、buffer、326组实际非零AdamW一二阶动量及step=4、param groups、scheduler、CPU/CUDA/Python/NumPy/order/view RNG、2681项完整数据顺序、cursor=64、update=4、epoch_steps=4、累计训练统计和最后一步指标均一致。

每一步还分别比较完整模型、buffer、优化器、全部RNG、裁剪后梯度哈希、损失、实际参数变化、输入组、评分、硬排列与逐人公共能量决策。update1–4均精确一致；故障路的两次update3也与连续路一致。没有用allclose或舍入掩盖差异。

为了让**数值训练状态**整体可逐值比较，两路统一使用原控制器参数`record_progress=False`；该开关关闭墙钟耗时、进度文件及第1步额外的进度checkpoint，不改变训练图、更新、数值统计、数据顺序或完整状态保存/恢复函数。第2步和第4步的本轮断点仍由原控制器实际保存。正式冻结配置和正式入口的`record_progress=True`未改；本轮不声称验证了这些额外进度I/O的故障恢复。真实耗时/显存/进程会话仍由原MODULE_UPDATES日志保留，它们是运行元数据，本就不会跨进程逐值相同；其差异不冒充训练数值差异。

## 4. φ/ρ的真实学习路径

以下为连续路原post hook测得的梯度L2及**真实参数前后差L2**，不是梯度×学习率的估计。恢复路及重放路同一步逐值相同；共享编码器梯度和参数变化也均有限、非零。

| 逻辑步 | φ梯度L2 | φ真实更新L2 | ρ梯度L2 | ρ真实更新L2 |
|---:|---:|---:|---:|---:|
| 1 | 0.210144081 | 0.00235647395 | 0.0366440729 | 0.00323255507 |
| 2 | 0.219902756 | 0.00193414766 | 0.0369490575 | 0.00273862939 |
| 3 | 0.208908736 | 0.00190996663 | 0.0337036174 | 0.00269593726 |
| 4 | 0.103497646 | 0.00176484334 | 0.0169686586 | 0.00251586057 |

这些记录证明当前合成输入下梯度能通过proposal-ST到达φ/ρ并实际改变参数；不证明真实开发集误差会改善，也不解决既有Sinkhorn近均匀、软硬偏差或顺序依赖。

## 5. 候选身份、端点和公共能量

每次训练配对检查每人20候选双射、canonical到原槽位逆映射、硬输出等于原候选gather、输入候选不变、评分有限、被接受更新的FP64可靠下降。连续路共有368次协调提案，接受316、拒绝52、严格并列0、舍入不确定0，公共能量单调性检查全部通过。

4步结束后，对同一组1/2/9/57人合成窗口实际运行完整G/H 20步DDIM，捕获原sampler收到的噪声：原(scene,agent,bank-slot,seed2035) keyed noise及配对gather逐值吻合，末帧严格等于冻结端点。连续/恢复/重放的完整G/H输出、最终排列与噪声哈希均一致。训练扩散噪声仍按原训练源码随机生成，没有改成推理keyed noise。

公共能量下降只是机制不变量，本轮不计算真实数据效果、不用能量下降替代JFDE改善证据。

## 6. MODULE_UPDATES断点与重放

连续和正常恢复日志逻辑序列均为`[1,2,3,4]`；故障路为`[1,2,3,3,4]`。两个update3记录保留不同session，均没有覆盖或删除。checkpoint尚在update2时，只把`[1,2]`视为已提交；恢复完成后，按原“同一逻辑update的最新尝试生效”规则选择`[1,2,3,4]`，全部数值监测与连续路一致。

原`epoch_summary`要求完整168更新，本次4更新日志会正确拒绝生成完整轮摘要。本审计使用checkpoint提交游标做局部核查，不伪造其余164条记录。未测试写到半条JSONL时的断电或存储损坏；本次注入点是完整记录写入后、checkpoint提交前。

原始日志：[连续](MODULE_UPDATES_continuous.jsonl)、[正常恢复](MODULE_UPDATES_split.jsonl)、[故障重放](MODULE_UPDATES_replay.jsonl)。`session/unix`、实际时间和显存记录保留原值；数值梯度、参数更新、能量/评分/gap/熵全部严格比较。

## 7. 资源、保护与停止

连续路四个逻辑批次实际前反向累计约19.133秒，allocated峰值约2310.6MiB。完整各阶段耗时与运行版本见[PHASE_RECEIPTS.json](PHASE_RECEIPTS.json)；这不是稳态正式训练吞吐预测。

历史受保护资产 **45,671个文件、13,154,566,994字节**，前后逐文件SHA一致，覆盖v2、v3、candidate_cost、sharedworld、既有审计、已准备的SSEC正式运行目录与零更新收据。全部本次合成输入、逐步权重、断点和故障现场留在新worktree的`runs/ssec_synthetic_nonzero_20261121/`，不会上传权重或候选数据。

6项审计工具单元测试通过，包括1 ULP差异检测、额外状态字段不能被忽略、日志提交边界/重放及数值日志误差负例；这些单元测试本身没有执行optimizer.step。

本轮补上了此前“未验证非零AdamW更新后的连续/恢复等价性”这一工程证据。结论限于同一GPU/软件/确定性FP32下的4步合成数据恢复，不是跨设备逐位保证，也不是完整轮、早停或训练效果实验。本轮结束，等待独立正式训练授权。
