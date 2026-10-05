# P2 role loader / step-cap CPU 验收

输入源码：`28728bf96a290ccd83c5c637769deb987f477b07`；日期：2026-10-05。发布版本以包含本报告的 Git commit 为准。

## 1. 结论与边界

- ENGINEERING_CPU_ACCEPTANCE = **PASSED**：63 个 P2 用例，连同相关旧合同共 157 个唯一用例的最新结果通过。22 个旧有 P2 通过项由首轮 33 passed / 16 failed 汇总和固定收集清单重建，未伪造逐项原始 trace；其余有逐项 JSON。不是一次 157/157 的单次执行。
- DATA_BASE_READINESS = **BLOCKED_RECORDING_IDENTITY_UNVERIFIED**；semantic 来源仍 UNKNOWN，4249 个 obs-only 产物缺失，合格 clean base = 0，新 cache/真实初态未生成。
- TRAINING_EXECUTION = **NOT_AUTHORIZED_NOT_STARTED**。真实神经网络构造/前向、真实数据更新、保护 future 读取、outer 指标、真实 candidate/bank 生成均为 0。
- 执行合规有历史例外：旧 CPU resume 子进程的 Adam 健康检查未拦截 CUDA availability 查询。由源码与 6 次 CPU step 推定 6 次查询；底层枚举/驱动初始化未记录，**不能宣称本轮 CUDA 初始化/枚举均为 0**。没有 GPU tensor/计算。已补齐子进程护栏，仅将 CPU optimizer 的诊断检查替换为带 CPU 参数断言的 no-op，实际更新算法不变；定向 9 vs 3+6 resume 复测通过。

详见 [RESULTS](RESULTS.json)、[CPU_RUN](CPU_RUN.json)、[来源证据](RECORDING_AND_MAP_PROVENANCE.json)。历史失败和记录缺口不被最新通过状态覆盖。

## 2. 已实际接线

新增显式 `p2_hotel_uni_examples / rsjg-p2-role-v1`，默认 off。原 P1 3790/320/139 与 source_block/final-test lock 保留，未改旧 clean_split_protocol。

运行 registry 固定原 4249 行 metadata 的 SHA，验证完整 physical 地址、稳定 window 身份、source aliases、跨角色 recording、成员/顺序及 obs8/future12 边界。Stage-A seed3101 顺序保持。train = physical train3345 + physical test ETH139；inner uni_examples320；outer HOTEL445。

RoleDataset 在 provider 前授权；obs/target/teacher 独立 typed/hash-bound 文件。没有 obs-only 文件即拒绝，不读原 full pickle 再删 future。只提供 per-window 分离纯函数，未对真实文件运行。训练标签和 teacher 跟 logical role；inner prediction 仅获得 obs8，target 只进入之后的 metric writer；outer metric 永久锁在当前入口。

main、parser、直接 trainer/goal_pretrainer/cache builder 均先检查 metadata/来源/产物，再进入设备或初始化。P2 goal 只调用 train（训练内部含 inner selection），不调用额外 train_test；原 goal train_test 的 test 复用 valid，本报告没有据此虚构历史 outer 泄漏。L1 仅建 train loader。

## 3. Cache 和 L1

生产 cache 的共同 task/execution 分支已用假 candidate、真实纯张量 future descriptor 和 atomic serializer 测试。计划 deployment4249、teacher3484，仅 train；保护 teacher 计算/写入0。新 schema v2/new root，不覆盖/移植旧 cache；dataset-level future 标记表示存在 train sidecar，不表示 deployment 含 future。

L1 controller 已嵌入真实 trainer._train_loop/_train_epoch 的 A0/A1 分支。cap 在 next(loader) 前检查；num_workers0、accumulation1。跳过/非有限/异常/身份不符停止整对，无补窗。部分 epoch 不 scheduler/validation/best/ordinary-last；cap 恰好完整 epoch 才 scheduler 一次，仍不 validation/best/ordinary-last。

正常 bounded stop 保存独立原子 weights-only snapshot、resumable=false；旧 resume 入口拒绝该 role。异常保留状态/ledger，不回滚 RNG、不清 pending。资源只在算子边界探测，不声称可中途抢占长算子。具体见 [合同](BOUNDED_UPDATE_AND_STOP_CONTRACT.md)。

## 4. Fresh goal → joint

新增真实 `--p2_config FILE --p2_launch_check` CLI。四份 YAML 实际 launch-check 都 exit2/recording BLOCKED，未执行设备/网络。fresh goal 不依赖已有 base；joint 必须匹配合格 P2 goal；cache 必须完整 clean base；L1 再要求新 cache、共同未训练初态和共享 pair budget。

新的独立 base packing 明确为原 source/window 顺序下按 agent64 装包、source 边界清包、无丢弃：train35456 agent-window exposures → **556 pack/epoch**；inner621 → **10 pack**。这是明确新增 P2 装包规则，不是对旧 loader 的吞吐认证；Stage-A 仍是3484 scene batch。goal150/joint250 上限分别83400/139000更新，不执行。

P2 base 原子 checkpoint 写 source/provenance/初态/曝光/选择信息及实际 SHA sidecar；sidecar 默认 PROVENANCE_REVIEW_REQUIRED，不自动 QUALIFIED，不宣称 CUDA exact resume。神经网络/loss 定义 AST 不变，新增参数/buffer0。详见 [基座计划](CLEAN_BASE_LAUNCH_PLAN.md) 和 [源码范围](SOURCE_SCOPE.json)。

## 5. 测试与保留的失败

首轮49项：33通过/16失败，失败是 constructor-free fixture 调用了要求完整模块的 train-mode helper；修复 fixture，不改网络。第二次 JSON 遇有意 NaN 写出失败，pytest 结果 UNKNOWN；当时只读检查了16份 tmp ledger，但归档前已被 pytest 临时目录保留策略清除，没有保留完整副本，派生计数不能冒充原始逐项证据。第三次10通过/6失败，CPU Adam 的隐式 CUDA 健康查询被父进程护栏拒绝；修复测试诊断护栏后6个 cap 用例通过。

相关旧合同一次范围回归：93通过/1失败，失败为既有 SDD 测试 stub 未接受生产已存在的 weights_only 关键字；仅修 stub，定向复测通过。后续只跑修复/新增/受影响选择，没有通过后全仓库重跑。真实 payload/真实 checkpoint/真实 corrector/sampler 构造类测试未执行，过滤条件在 COMMANDS/CPU JSON。

已知实际 CPU tiny optimizer 成功更新 **151**（包含重跑与两次 child各6）；不能写“optimizer 全0”。记录充分的运行累计35.74秒，丢失第二次记录的耗时不精确，整体必要 CPU 测试未接近1小时。每次原始结果和边界见 CPU_RUN*.json。

## 6. 来源补证与下一步

CMU 作者实现将 UNIV/UNIV2 映射至 students003/001，但借用视频仅为尺寸；NVlabs 列出 uni_examples 文件名；Google 说明 ETH/UCY benchmark。它们均未绑定本地 source SHA，也未证明 uni_examples 与 students 文件的 recording 独立性。官方 UCY 页面访问失败如实登记；没有下载 annotation/视频或比对保护坐标。semantic raster 的使用已定位，上游生成模型/训练标签/许可仍未知。

旧五份 checkpoint 违规结论保留，未重读重审或“洗白”。下一步先补 recording/session/ID remap/release 到本地 SHA 的正面证据与 semantic 许可，再另行授权 per-window obs-only 产物及 fresh goal→joint 的有界 timing/训练。合格 base 后才能生成新 cache，再 seed3101 L1；**不能直接 A1**。L2 JADE/JFDE/min_delta/early-stop adapter 未实现；CPU通过不等于 L2 ready。

SDD、Skills、Apps 未干预；历史 artifacts 与旧报告保留。无继承旧96h/204h训练授权。
