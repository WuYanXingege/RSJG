# 改动与保留位置

原始位置相对于固定报告提交；候选均是隔离分支的新文件。现有正式文件未改。

## P0

- 原始：`bcep_v3/formal_telemetry.py:ModuleTelemetry.capture_backward:L12–L20`
- 候选：`bcep_v3/perf/telemetry.py:BatchedModuleTelemetry.capture_backward:L38–L56`
- 原计算：逐张量 finite 布尔读回与 FP64 平方和 float 读回。
- 具体改动：每模块设备端归约，三个模块的 finite/平方和一次读回；仍在 clip/step 前拒绝 NaN/Inf。
- 数值边界：不接入损失图；统计跨张量求和顺序可变化，模型/梯度不得变化。
- 证据：逐step状态、非有限负例、缺失梯度、参数对象身份。

## P0

- 原始：`bcep_v3/formal_telemetry.py:ModuleTelemetry.step_done:L21–L31`
- 候选：`bcep_v3/perf/telemetry.py:BatchedModuleTelemetry.step_done:L59–L79`
- 原计算：逐张量参数差 FP64 范数与最大值读回；rows 无界增长；每步 JSONL。
- 具体改动：每模块归约后一次读回；保留每步字段/编号/写盘；内存仅留64行。
- 数值边界：统计允许独立舍入容差；参数克隆与实际step对应不变；没有 fsync 断电保证。
- 证据：正常恢复、重放编号、完整日志、70行内存界限及半行拒绝。

## P1

- 原始：`bcep_v2/stages.py:joint_callbacks.backward:L134–L177`
- 候选：`bcep_v3/perf/stages.py:joint_callbacks.backward:L77–L110`
- 原计算：每物理组读取 loss、pair metrics 和 loss components 到 Python。
- 具体改动：detach 后保留标量，逻辑更新末集中读回，按原次序在 Python 进行同样乘除/累加。
- 数值边界：损失构造、agent/scene 权重、backward顺序、RNG和optimizer完全不变；非有限诊断在step前拒绝。
- 证据：完整16窗+3窗尾批，逐step前后SHA，恢复和RNG，所有日志字段。

## P2

- 原始：`bcep_v2/stages.py:joint_callbacks.backward:L134–L177`
- 候选：`bcep_v3/perf/stages.py:compact_checked:L53–L66`
- 原计算：physical.to(cuda) 后再 physical.valid.cpu() 与CPU掩码比较。
- 具体改动：先验证CPU压紧后掩码相同，再搬运同一 SceneBatch；保留非前缀mask拒绝。
- 数值边界：没有数值运算或成员重排；counts.tolist本来就在CPU，不记作CUDA同步。
- 证据：坏padding在.to前拒绝、混合人数、顺序和完整状态精确相等。

## P3

- 原始：`ebjd/encoders.py:MotionSceneContextEncoder.forward:L139–L216`
- 候选：`bcep_v3/perf/options.py:checkpoint_arm:L6–L14`
- 原计算：训练时按map_agent_chunk对U-Net activation checkpoint重算。
- 具体改动：仅独立性能臂显式切换encoder/denoiser原有开关，并finally恢复。
- 数值边界：CPU小图2人样本输出/梯度精确；GPU全分辨率57人显存和数值均未验收；默认仍开checkpoint。
- 证据：独立map/bridge/both臂，输出/梯度/RNG与未来allocator peak。

## P4

- 原始：`bcep_v3/geometry.py:reference_paths:L5–L11`
- 候选：`scripts/perf_audit/mechanisms.py:simplified:L27–L30`
- 原计算：CV baseline + 比例×端点残差。
- 具体改动：仅诊断脚本实现代数化简；生产geometry、incidence和attention均未改。
- 数值边界：FP32非逐位等价；57人发生argmin时间跳变，拒绝默认替换。
- 证据：12维逐项差、CPA、世界分数、两次Hungarian、梯度与近并列负对照。

## P5

- 原始：`bcep_joint/inference.py:keyed_noise:L8–L17`
- 候选：`scripts/perf_audit/mechanisms.py:noise_audit:L92–L125`
- 原计算：590窗4262人×20次CPU Generator初始化。
- 具体改动：只读CPU缓存可行性实验；键包含scene/agent/world/seed/有效顺序/帧坐标身份；不接入generate。
- 数值边界：缓存副本逐字节相同，CPU RNG不变；小成本不等于整网显著加速。
- 证据：590窗三次原生/缓存对照、字节hash、缓存8MB成本。

## 保留的科学和安全路径

- `ebjd/denoiser.py:GeometryConditionedJointDenoiser.forward:L282–L394`：six FJAB checkpoint recomputation; training only。
- `ebjd/encoders.py:MotionSceneContextEncoder.forward:L139–L216`：map U-Net checkpoint recomputation; context encoded once for validation。
- `ebjd/unified.py:hard_assignment_indices:L41–L60`：scores GPU-to-CPU, CPU Hungarian twice, per-agent result copies; unchanged。
- `bcep_v3/coupling.py:RelationalAssignment.relations:L24–L50`：edge-chunk candidate pair geometry and dense incidence/matmul; unchanged。
- `bcep_v3/coupling.py:RelationalAssignment.forward:L52–L70`：dense world social attention and relation values; unchanged。
- `bcep_v3/gate.py:GeometryReliabilityGate.forward:L12–L38`：actual path pair distances and bounded CTF gate; unchanged。
- `bcep_v2/objective.py:bridge_and_pair_loss:L7–L65`：training pairing K20 but conditional bridge uses P=1 target-conditioned world。
- `bcep_joint/inference.py:generate:L26–L44`：validation K20, DDIM20; one shared context; no change。

物理组不按人数重排；不使用BF16、TF32、torch.compile、新DataLoader进程或GPU并行。CSEE/HCRA/GRG-CTF、旧损失、Provider、候选、硬排列和DDIM语义保持。
