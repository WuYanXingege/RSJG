# P2 clean base / split / objective preflight

源码：`4f11f44c3a490c491eb310b4fa7529806c71c1ed`；日期：2026-10-05；分支：`research/joint-dependency-v2-clean`。

## 1. 结论与状态

**AUDIT_COMPLETE；CLEAN_PILOT_READINESS = BLOCKED_BASE_PROTOCOL_VIOLATION。**

在限定清单内，合格 GDTS base **0**，明确违规 GDTS base **2**；另有继承违规基座的 Stage-A checkpoint **3**（不是五个独立基座）。这五个文件均有正面违规证据，不能降格为仅来源未知。限定搜索不等于全盘不存在其他合格模型。map/semantic 上游训练来源、跨文件 recording 身份仍 UNKNOWN。

[A0](A0_BLOCKED.yaml)/[A1](A1_BLOCKED.yaml) 是 TEMPLATE_BLOCKED；base/cache/共同初态的路径及 SHA 为 null，不能启动。只读检查入口拒绝它们（退出码 2）。次级阻塞为 P2 loader/安全 step-cap 未实现、旧 cache producer 违规、protected split teacher 生成策略不合格，以及 recording/agent-ID 命名空间未认证。**不启动 A1，也不以旧 canonical A warm start 代替 clean 对照。**

## 2. 三个协议不是同一认证

| 协议 | 成员和用途 | 证据边界 |
|---|---|---|
| P0 历史 ETH | valid/test 同一 biwi_eth 139 窗；canonical A epoch13 在此选择 | 139/139 重合；历史开发/兼容性证据，不是独立最终测试 |
| P1 旧 clean 基础设施 | train3790 / uni_examples inner320 / ETH test139 | 认证窗口、cache 地址、test lock；未认证冻结 base 的训练/选择来源，也不支持任意 P2 role |
| P2 本次保持预登记 | train3484 / uni_examples inner320 / HOTEL outer445 | 整 source 家族分角色；无随机拆窗、无换 fold；基础设施和来源仍被阻断 |

P2 “其余合格来源”解释为包含旧 ETH 开发的 139 窗作为 **train**，不是最终测试：3345 个原物理 train 窗 + 139 个原物理 test 窗。4388 个旧物理记录减去 ETH valid/test 的 139 个重复，得到 4249 个唯一窗口；不截断 agent、不改图、不重编号 cache。旧 P1 manifest 和“ready/untouched ETH”历史措辞未改。

HOTEL 不是从未接触：旧 ETH/UNIV 基座及 canonical/P1 heads 在其上训练；上轮 R1–R4 使用 HOTEL frame0 N3/E3、frame620 N1/E0，读取 future teacher、loss 并 backward，但该 smoke 的真实 optimizer 更新为 0。旧训练与此零更新不能相互抵消。HOTEL 未找到直接 checkpoint-selection 记录，不代表从未用于任何方法选择。未来新模型可做到前瞻隔离，不能撤销已有开发暴露。

## 3. 实际基座链

| 候选（文件 SHA 前缀；完整见 JSON） | 实际 epoch | P2 判定及证据 |
|---|---:|---|
| GDTS ETH best，126acf2a | 90 | train_test/test_set=eth 的原始 train 成员包含 HOTEL、uni_examples；完成日志 goal/diffusion loss 为正，best valid ADE 的 epoch 与文件一致 |
| GDTS UNIV best，ebfbae9d | 100 | test_set=univ 的 train 同样包含 HOTEL、uni_examples；日志与 best epoch 一致 |
| canonical A，699336d4 | 13 | 53430=13×4110 更新；heads 训练两个保护 source，ETH139 选择；冻结组件继承 ETH base |
| P1 A best，95f32cc2 | 1 | heads 更新3790次，包含 HOTEL；uni_examples 为 head valid，但 base 已对其训练 |
| P1 A last，87ae976b | 1 | 同3790更新；参数组件与 best 相同，checkpoint 文件身份不同 |

[BASE_PROVENANCE.json](BASE_PROVENANCE.json) 保存每个 state key、shape、numel、组件 tensor-byte SHA、完整文件 SHA、配置和父来源。三个 Stage-A 的 goal / history registrar / diffnet 分别与 ETH base **实际权重逐组件同 SHA**，不是仅凭路径或配置推断。完整历史 SGD 样本事件轨迹和更早父权重证据不齐，明确 UNKNOWN；当前正面 run-family 训练证据已足够判违规。

地图读取 semantic mask/homography；已检查路径未发现依赖未来坐标拟合的归一化，但外部 segmentation 的训练来源未认证。不能据此声称所有可学习上游组件已经 clean。

## 4. source、重叠与保护

[SPLIT_ROLE_MANIFEST](SPLIT_ROLE_MANIFEST.json) 和 11 个 windows JSONL 分片提供 source 原路径/内容别名、physical split/cache index、logical role、20帧与 agent 身份、既有内容指纹。8 个 byte-distinct 文件、5 个 scene family，**不是已证明的 8 个独立 recording**。原始文件只做流式 SHA，没有解析保护分区的坐标；逐窗口 tensor 指纹继承已认证历史 metadata，本轮没有重开所有 source batch 重算它们。

| source | P2 role | 窗数 |
|---|---|---:|
| biwi_hotel | outer | 445 |
| uni_examples | inner_valid | 320 |
| biwi_eth | train | 139 |
| crowds_zara01 / 02 / 03 | train | 705 / 998 / 695 |
| students001 / 003 | train | 425 / 522 |

按 source SHA 命名空间，跨角色 agent-time 交集为 0、缺 metadata 窗为 0。共享 agent-time 的窗口数 train3482/3484、inner318/320、outer438/445，不能把窗口当独立样本。

额外 [CROSS_RECORDING_IDENTIFIERS](CROSS_RECORDING_IDENTIFIERS.json) 显示同 scene 跨文件 ID/frame 碰撞：students001–uni_examples 76、students003–uni_examples 29；train 内 students001–003 为2984、zara02–03 为1297。**这是 identifier collision，不是已确认轨迹重复；也不能当作独立性的证明。** 下一步应取得 recording/ID namespace metadata，不解封保护未来标签，不擅自合并/换源。P2 readiness 保留此 split provenance 阻塞。

## 5. cache、loader 与接口缺口

[Cache 检查](CACHE_COMPATIBILITY.json)：仅反序列化非保护 train 的 deployment 000445 和 teacher sidecar 各1次（N6/E15）。deployment 无 future/learned h/u，teacher descriptor[15,6] 有未来信息；顶层 contains_future_supervision=true 不等于 deployment 泄漏。两个旧 cache 的 producer 都是上述违规 base，不能复用旧 G/p0 然后改写 SHA。

当前 builder 为所有 physical split 算 future teacher（src/joint_dependency_v2_cache.py:293）；未来必须 train-only teacher、protected observation-only candidates。预计新 producer 需4249 deployment records、3484 train teacher records、保护 teacher 0，不导出完整 trajectory bank。

生产 clean loader 有3790/320/139硬约束，inner 必须单一末尾 block，outer 固定 physical test；P2 outer 在 physical train、P2 train 包含旧 physical test，不能用换 test_set 或旧 P1 manifest 表达。其 source-identifier helper 会 load 完整 batch，本轮未调用。未来 role 必须先由独立 metadata 判断，再授权 payload 读取；augmentation/teacher 跟随 logical role，不能跟随物理目录。

trainer 无 H/attempt 安全 cap；L1 必须跳过 validation 的真实执行分支。L2 Stage-A auto best 是 JFDE，现有 JADE 选项不自动实现 JFDE tie-break/min_delta；未来需要最小选择适配。A0 现有 joint_diagnostics 会在 prediction 后额外读取 teacher loss，MC 跳过：未来 L2 两臂都须关闭这类诊断或隔离预测/指标 writer，保持推断噪声相同。不能靠修改 YAML 假装这些能力已存在。

## 6. 固定结构与公平性

见 [ARCHITECTURE_AND_INTERFACE](ARCHITECTURE_AND_INTERFACE.md)、[配置差分](PAIRED_CONFIG_DIFF.json)、[初始化/RNG](INITIALIZATION_RNG_PLAN.md)。11个架构模块 AST 与79ea3fa定义相同。Stage-A 注册331845、endpoint有效路径331781（不等于逐次全参数非零梯度）；base参数7826588、实际forward参数5723804、unused template2102784，corrector30851冻结。state buffer12793单列。新增参数/层/buffer均0。

A0 mean_energy 对 A1 expected_conditional_mc/S4；目标、backend、MC seed、run输出是唯一允许差异。旧 loss/teacher 梯度/mixing/KL、K21/P20、两轮 sampler、graph/精度/训练预算均不变。本轮不产生共同真实初态：NOT_CREATED_AUTH_REQUIRED。

## 7. 待授权试验与最小下一动作

[L1/L2预登记](PILOT_PREREGISTRATION.md) 和 [资源依赖](RESOURCE_AND_DEPENDENCY_PLAN.md) 分开。L1仅3101，两臂H500次尝试；Adam1e-4/clip1/gamma.995，native epoch3484，参考总步3484，beta末次0.0716130884，部分epoch不调度、不验证。每臂900s、总2700s、reserved10GiB都是停止上限，不是预测。任何skip/非finite/pending/身份不符立即停止整个pair。

**最小下一动作**：先补 recording/semantic 来源 metadata 和 P2 role/teacher/step-cap CPU 合同；形成经授权的 clean goal→joint base 训练/inner选择产物；认证完整 base 链后生成新 cache；最后另行授权 seed3101 L1。当前没有合格 base，不能把下一步简化为授权 A1。旧204 GPU-hours不带入执行许可。

## 8. 本轮真实账本与限制

审计主进程5.1023s：checkpoint 5个各一次 CPU load（上限20），非保护 deployment/teacher各1；75次文件hash共198351497 bytes，JSON metadata8、text16。[EXECUTION](EXECUTION.json) 是该进程的精确账本，不是全会话所有 shell read 的总数。补充 metadata脚本31次read/hash，跨recording脚本12份 metadata；交互式源码/文档读取未统一计数，不能把主进程计数冒称全会话总量。一次旧 training.log shell 输出被截断，随后脚本完整读取并提取所需证据，无丢弃失败尝试。

针对性 CPU 测试：首次28/28通过0.18s；新增 recording 检查2/2通过0.04s（28 deselected），合计30个不同测试、30次、0失败；没有重跑旧172项。见两份 CPU_TESTS 记录。

保护 future tensor读0、outer指标0、真实网络构造/调用0、CUDA初始化/枚举/同步0、optimizer update0、训练/评估/sampler/diffusion0。没有启动/停止 SDD、改变 Skills/Apps、修改生产 src/default YAML、重写旧 artifact。工具导入无 torch/model side effect；CPU audit 才延迟导入 torch。工具不构成完整训练 launcher，静态 plan 检查通过也不代替未来运行时权限和身份 guard。

收益、创新、pair 因果、真实CUDA续跑、GPU观察中立性仍 OPEN。本轮是来源/准备审计，不是 efficacy 实验。

## 9. 文件入口

[RESULTS](RESULTS.json) · [来源暴露矩阵](SOURCE_EXPOSURE_MATRIX.csv) · [补充源证据](SUPPLEMENT_METADATA.json) · [本轮与未来命令](COMMANDS.md) · [哈希清单](EVIDENCE_MANIFEST.json)。
