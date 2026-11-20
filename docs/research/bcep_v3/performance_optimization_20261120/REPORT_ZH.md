# BCEP–EBJD v3 独立性能审计与可选优化

**CPU_READY_GPU_PROFILE_PENDING。GPU加速率 UNMEASURED，没有GPU置信区间，也没有BENCHMARK_PASS。** 正式训练状态未被本任务修改；本审计向原训练注入的真实optimizer更新为 **0**。独立合成CPU optimizer更新 **78**，合成GPU更新 **0**。原授权任务在两次资产收据之间自然从update **332→450**，PID仍为1978816、状态RUNNING；这部分正常增长不伪称“计数完全不变”。

当前可以提交的是默认关闭的 **P0遥测批量读回、P1诊断集中读回、P2 CPU身份检查**，作为下一次独立实验的候选。它们须先完成空闲GPU上的逐更新精确性和同输入速度/显存验收，再取得新实验授权及重新冻结，不能直接替换当前任务。**Checkpoint关闭开关只用于基准；几何化简不具备逐位等价性；noise缓存本轮不接入。**

## 固定身份与边界

- 报告/代码分叉锚点 `03158af4a56e75e1cc169b7e8e096048b54ba0bc`；正式运行源码 `2e96b6f35554ceceb5a04fc08b6259dac75e486b`；正式冻结配置规范化SHA `294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43`。
- 分支 `perf/bcep-v3-speed-audit-20261120`，候选与未修改baseline分别位于独立worktree。所有原有正式源码文件与冻结manifest、baseline逐文件一致；只新增perf命名空间、审计脚本和测试。原formal入口未导入perf，授权与源码SHA检查未放宽。
- 首次只读检查确认正式进程和GPU占用，故本轮没有GPU基准、attach profiler、等待GPU、自动预约、信号、重启或热替换。数值验收和CPU噪声实验隐藏CUDA并使用单线程/低优先级。
- 42429个静态受保护文件前后大小/SHA相同，包括运行worktree、冻结配置、既有完整epoch、历史checkpoint和bank。活动日志、last/best/Top5及新epoch由原任务正常推进，另存前后收据，不纳入“静态文件完全相同”的伪断言。[保护收据](PROTECTION.json)
- v2五折仍PAUSED_RESUMABLE，candidate_cost仍SESSION_BUDGET_PAUSED，sharedworld仍COMPLETE。未打开真实轨迹或test/heldout作训练/评测；P5仅读取已冻结开发集身份元数据和生成CPU随机噪声。

## 证据能说明什么

固定11:45快照的160个更新间隔平均5.8647998765秒，估算训练16.42分钟/轮、加历史验证约21.21分钟/轮；它们不是此次优化收益的测量。人数、padding和分组数高度相关，r≈0.9707不能给某个模块分配因果耗时。本轮确认了同步位置与计算规模，**尚未测得GPU时间占比或最主要瓶颈**。

P0原路径在全部34个新增参数张量都有梯度时有约136次标量/布尔host边界：梯度finite、平方和、参数差平方和、参数差最大值各一组。候选保留相同FP64定义，在GPU归约后每逻辑更新集中读回两次。参数快照仍克隆约1372136字节；未靠删除审计、降精度或减少写盘取得假提速。边界计数下降不等于相同比例的时间下降，FP64计算和kernel launch仍存在。

P1每物理组约11个CUDA诊断标量改为一个逻辑更新末集中搬回。保留原Python双精度乘除/累加次序，**原loss构造、agent/scene权重、backward调用顺序、裁剪、AdamW与RNG消耗没有更改**。只缓存detach后的标量，不持有autograd图。非有限诊断在本次step前拒绝；原裁剪和参数finite检查保留。

P2把合法成员/压紧mask检查放在搬运前；没有删除验证或重排成员。`counts=batch.valid.sum(1).tolist()`来自CPU collation，不能算CUDA同步。

其他路径仍需独立profile：两次CPU Hungarian及score/result传输、HCRA的E×20×20×12候选对几何、incidence/dense attention、CSEE候选注意力、GRG路径几何、地图U-Net、六层FJAB及重算。原生`finite_tensors`已经统一归约再检查，不再重复优化。v3输入验证使用CUDA异步assert，保留其安全边界。

**训练桥实际使用P=1的目标条件世界，配对仍为K20；验证才是K20、20步H-DDIM。** 验证共享context只编码一次；activation checkpoint仅在训练且开启梯度时生效。因此不能把每次训练步误算为20世界DDIM，也不能指望关闭checkpoint直接减少验证的20步。

逐位置的原计算、候选原理、浮点风险和证据见[CHANGE_MAP.md](CHANGE_MAP.md)，包括文件、函数与行范围。[TRACE_SUMMARY.json](TRACE_SUMMARY.json)没有填写虚构耗时占比。

## 两类等价性

**P0/P1/P2：CPU训练状态精确。** 共36项不同测试通过，共执行38个测试用例（含增强验收复跑）。覆盖1、2、9、57人、padding、空邻居、同位邻居、逐人独立K20候选重排、行人重排、混合场景、完整16窗及3窗尾批。比较优化前后同一输入，而不把有并列的Hungarian说成唯一解。

每个完整逻辑update的optimizer前后模型/buffer、梯度、AdamW、CPU/Python/NumPy RNG、顺序/增广生成器及组成员SHA完全相同；原生连续4步、独立P0、P012与1+3步恢复的完整状态相同。scheduler、cursor/order、history/stale与计数也比较。原minimum30/patience15、相同分数选早epoch、Top5与完整原子恢复状态的原生测试通过。统计字段单独容许rtol1e-12/atol1e-15，最大观察差为 **4.336808689942018e-19**，不拿统计容差掩盖模型差异。

额外零optimizer检查直接捕获初/终Hungarian、世界score、硬端点、桥final/coarse/coarse path/coarse goal和全参数梯度：P012在1/2/9/57人合成样本逐张量相同。所有这些CPU样本使用**32像素合成地图**和原192宽度/六层结构，不能代替正式256像素GPU容量与CUDA RNG验收。[EXACTNESS.json](EXACTNESS.json)

P0日志仍每step写完整字段，内存仅保留最近64行；70行压力检查确认不积累历史Tensor。恢复从checkpoint的update编号继续，保留已经存在的物理重放事件而不改写它们；重放测试序列1/2/3/2的统计对应正确。半行尾记录拒绝继续拼接，需要明确离线修复。close不是fsync，突然断电可能丢末尾记录；本轮没有宣称日志提供断电级持久性。70次及4次日志压力/重放中的参数扰动不属于optimizer.step，已与78次合成optimizer更新分开。

**P3/P4：可选计算路径，不能由CPU局部检查外推GPU无损。** Checkpoint开/关在2人32像素CPU样本输出、梯度及RNG一致；map、bridge、both开关都会恢复，默认仍开启checkpoint。没有allocator peak和57人256像素GPU结果，不允许默认关闭。

几何代数化简在实数域等价，但本轮FP32已经出现非零输出/梯度差，状态标为 **NON_BITWISE_OPTIONAL**，只留在诊断脚本中，正式geometry没有改。

| 人数 | 参考路径最大差 m | 12维关系最大差 | argmin时刻改变元素 | 世界score最大差 | 梯度最大绝对差 | 初配/终配相同 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.38418579e-07 | 0 | 0 | 0 | 0 | True/True |
| 2 | 4.02331352e-07 | 5.96046448e-07 | 0 | 2.98023224e-08 | 1.86264515e-08 | True/True |
| 9 | 7.15255737e-07 | 5.64171933e-06 | 0 | 2.98023224e-08 | 4.65661287e-09 | True/True |
| 57 | 2.86102295e-06 | 0.0833333135 | 2 | 3.27825546e-06 | 1.16415322e-09 | True/True |

57人有2个候选对的最小距离时刻跳变；时间特征约变化1/12，说明微小坐标舍入可经argmin放大。测试样本的两次Hungarian仍相同，硬端点、noise逆gather和边缘minFDE保持。另构造score仅变化1e-8、满足1e-6 allclose却翻转Hungarian的负对照，证明不能用score接近代替排列一致。逐维CPA/关系误差、相对梯度误差分母下限1e-8及负对照说明见[GEOMETRY.json](GEOMETRY.json)。无profile证据支持改动incidence、分块或关系组装，本轮保留原实现。

## Keyed noise与资源成本

按官方590窗、4262个有效行人、K20、seed2035，三次纯CPU生成中位数 **1.9060秒**；缓存clone中位数 **0.004228秒**。缓存Tensor需8183040字节，另有Python键/字典开销。三次输出逐字节相同且CPU全局RNG不变；缓存键涵盖算法/torch版本、scene、agent、world、seed、有效成员顺序、frame IDs和坐标身份。

这只是同进程CPU组件的探索性三次计时，未测整网加速。约1.9秒与历史287.28秒验证相比只占约0.7%的量级，且两者并非同期隔离A/B，故不把缓存接入生产推理。[CPU_NOISE.json](CPU_NOISE.json)

P0保留全部参数克隆，额外GPU归约标量很小但未实测peak；P1保留detach标量至逻辑步末，16个物理组时最多约176个Tensor标量，仍有Python容器与allocator开销。CPU等价性样本不证明关闭U-Net/FJAB重算后能装入显存。原约5.5GiB的进程采样含reserved/context且会漏峰值，不能作为关闭checkpoint的容量证明。

## 待执行的独立GPU基准

当前 **GPU_PROFILE_PENDING**，没有生成BENCHMARK.csv或GPU trace。已提供可执行入口及通过的准入负例测试；**GPU执行分支本身仍未运行**。正式任务未停止、GPU存在其他计算进程、缺少独立授权环境变量、输出路径不在性能目录或目录已存在时均拒绝；不发信号、不等候、不预约。只接受内置合成数据，没有正式训练/数据入口。

未来手动吞吐命令使用同一初态和RNG序列、FP32/math-SDPA、AdamW及原epoch1学习率，形状含1×1、4×8、1×57和混合16窗，地图256像素，2次预热和每形状5次测量。比较baseline、P0、P01、P012，再单列checkpoint开关和关闭telemetry的开销对照。CPU整状态已单测P0与P012组合；P01中间臂和所有GPU臂的独立基准仍待执行。每次调用合成更新上限300；profile独立运行，不能把重profile时间用于速度结论。

记录墙钟、CUDA跨度、p50/p90/方差/范围、max_memory_allocated及实际合成step ledger；逐step完整状态hash在计时区外核验。CUDA跨度含CPU供给造成的空隙，不等于kernel纯占用时间。profile通过独立scope区分数据/增广、两次Hungarian、几何、CSEE/HCRA、map U-Net、六层FJAB前向/重算、GRG、backward、遥测/日志和AdamW；重叠scope不能直接相加。速度差只能取同环境同输入的100×(Tbase−Topt)/Tbase，5次单次调用也不足以声称跨运行稳定收益。

[COMMANDS.txt](COMMANDS.txt)列出手动命令。本轮没有运行这些GPU命令，也没有安排训练结束后自动执行。

## 后续适用范围

保留旧实现为默认。优先在**下一次获准的独立验证**中检验P0，再加入P1/P2；只有GPU状态逐位相同且收益/显存可接受，才考虑新的冻结训练配置。P3保持诊断开关；P4拒绝视为无损；P5不接入。人数重排、BF16、TF32、compile、多进程DataLoader、GPU并行均未实施。

所有正式权重、候选和既有报告保留原位；本任务到此结束，不自动应用补丁、恢复队列、扩折、换seed、消融或测试集选模。
