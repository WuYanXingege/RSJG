# RNG 与 checkpoint 合同 v1

Owner 是 trainer 管理的普通 Python ObjectiveRNG，传给 model，仅用于新训练目标。不属于 nn.Module、网络 state_dict 或 architecture_config。

## Seed、请求和进度

CPU torch.Generator；显式 neighbor_seed 优先，否则 SHA256("jdv2.expected-conditional.cpu.v1:<training seed>") 前 8 bytes big-endian mod 2**63。没有 Python hash() 或 global RNG 抽 seed。fresh 初始化才 manual_seed；resume 用完整 uint8 state。global Torch/Python/NumPy 和独立 evaluation generator 不因该抽样改变。

每个合法请求抽一次，含 E0；post/prior 共享。draw_calls +=1，sampled_agent_draws +=4N；成功 optimizer 更新单独计数。Stage-A accumulation=1。未完成请求置 pending_backward=True（也覆盖 draw 后尚未 backward 的阶段）；成功更新或显式 skip 才清理。skip 消费 RNG 但不增更新。draw 后 cost/gradient/optimizer 异常保留状态并停止，不回滚、不自动重试、不允许续跑保存。

## 元数据与语义身份

新模式 payload 字段 jdv2_goal_objective_state：

format_version=1；objective；neighbor_draws=4；generator_device=cpu；generator_algorithm=torch.cpu.MT19937；sampling_semantics_version=multinomial-replacement-Nx4-transpose-v1；torch_version；namespace；initialization_seed；cloned generator_state；draw_calls；sampled_agent_draws；successful_optimizer_updates；stage_total_optimizer_steps；stage_progress；checkpoint_role；pending_backward；objective_semantic_fingerprint；source_commit；resume_scope。

fingerprint 覆盖 namespace/目标版本/S/target sigma、dataset/test_set、cache/base/clean-split 身份、成功步数进度定义、optimizer/scheduler/clip/LR/group scale。固定总步数另精确匹配。source_commit 为运行时实现仓库 HEAD，记录但不参与 fingerprint；文档-only commit 不构成目标不兼容。未提交补丁的审计身份另由 SOURCE_SCOPE/文件哈希固定。未来改变算法须升级语义版本，不能靠沿用字符串绕过。

snapshot generator state clone，不随 owner 继续抽样改变。先在临时 generator/optimizer/scheduler 上校验；同 stage 还核验 model 架构/provenance、保存边界、顶层 progress、optimizer tensors/step 和 scheduler epoch/LR。旧 loader 的晚期检查由权重/内部 pending 状态回滚保护。非法恢复 fixture 中权重、optimizer、scheduler、RNG/progress 均不污染。

## 兼容矩阵

| 操作 | 合同 |
|---|---|
| old 无字段 → mean training | 原兼容；无 MC RNG/新字段 |
| MC → same compatible training | 只从 resume_last；恢复完整 RNG、optimizer、scheduler、progress |
| MC → mean training | 拒绝 |
| old mean 同 joint_goal → MC | 拒绝；baseline_initialization=True 也不能吞掉 mismatch |
| 真正 legacy/base cross-stage 初始化 → MC | 只初始化允许的权重；新 seed、清零 optimizer/scheduler/progress；不叫 resume |
| MC → weights-only evaluation | 架构/权重检查照旧，不恢复/抽 MC RNG；train_test 也显式 weights_only |
| 版本/S/device/sampling/torch/fingerprint/总计划不匹配 | 拒绝，无 reseed/fallback |
| 缺/损坏 RNG、负计数、progress 不一致 | 拒绝 |
| pending 请求/梯度 | 保存和恢复均拒绝 |
| best/numbered → training resume | 拒绝；只作选择/评价权重快照 |

跨 stage 不是泛化 allow-mismatch：本实现只开放 genuine legacy/base → Stage-A 初始化，不能借此从别的 JDV2 目标快照自动 warm start。

## 保存角色、顺序和原子性

生产保存顺序静态核验：numbered/best 在 epoch scheduler 之前；last 在之后。MC：
weights_numbered / weights_best 不可续跑；resume_last 必须已调用 completed-epoch marker、无 pending、成功计数/progress 一致并有 optimizer。epoch-local marker 保存在同 payload 的 mc_scheduler_epoch。训练默认预算/选择指标/历史 epoch 均未改。

仅 MC resume_last 使用：
同目录唯一临时文件 → torch.save(包含全部权重/训练状态/RNG) → flush → 文件 fsync → os.replace → 目录 fsync。
序列化/flush/文件 fsync/replace 前失败，旧目标字节不变，临时文件删除。rename 后目录打开/fsync 的 OSError 抛 CheckpointDurabilityError(committed=True)：新文件已提交，目录持久性未知；不宣称旧文件还在。没有 RNG sidecar。旧模式/MC weights-only 快照保存策略不顺带重写。

## 有界认证与 OPEN

生产方法通过 constructor-free raw fixture 调用；子进程实际加载 last，再运行相同固定输入剩余 6 步。连续9 vs3+6 exact，比较字段见 RESULTS.resume。也检查先 load 后创建 optimizer 的生产延迟恢复顺序。

仅限 objective RNG + optimizer/scheduler/progress、相同输入序列、相同计划和指定 epoch boundary。**不包含** DataLoader/worker/order、augmentation、全部 global RNG、CUDA/AMP、不确定性、真实完整训练。CPU step 成功计数通过，不表示 GradScaler skip 已认证。真实模型/真实 checkpoint 未读取或运行。
