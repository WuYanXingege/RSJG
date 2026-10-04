# 有界 restart trace 执行合同

授权文件：`RSJG_Restart_Trace_Codex_Prompt_ea846fc.md`；SHA256 `6a569ac26ea26c344bc4b22d6417a0d69e76cf48d7ba521e2f6c2844fcb606a7`。基准 HEAD `ea846fcb3dafeff45dbe5ae841b477c36b6676be`；GPU 运行前提交的 probe 源码 `80ee0e2c1762b019a43e7297e12e2fd880a8ed03`。

仅新增独立 tools 与本次审查记录。未改 src、冻结配置、checkpoint、缓存、生产 sampler、精度或指标。无适用 AGENTS.md 被发现。未恢复或改动 skills，未卸载 Apps。

## 运行命令与预算

在 `RSJG_JDV2_clean` 仓库根目录分别执行，前一个进程结束后才启动下一个：

```bash
../.conda/rsjg/bin/python -B tools/jdv2_restart_probe.py --process P0 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_restart_probe.py --process P1 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_restart_probe.py --process P2 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_restart_probe.py --process P3 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
```

以上是预定命令，实际开始/完成数以 RESULTS 的 budget 和持久 STARTED/INITIALIZING/INITIALIZED/budget/COMPLETE/FAILED 记录为准，不以列出命令代表已执行。

P0/P1 无内部 trace；P2/P3 只对 window13 开启 trace。每进程新建唯一目录；已存在就拒绝，不能自动重试。seed2036 从原 loader 顺序执行 windows0–13，分别保持 encode 与 ts_sample 的原 autocast 上下文。完成13后直接退出，不取 window14。最多4进程、56次完整 A forward；最多4次不含 diffusion 的额外 region 诊断单列；CPU exact payload 重算不冒充新的 encode/sampler 重放。

持久目录：`outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/restart_trace_ea846fc/`。每次 forward 前写 attempt，成功后写 complete；异常捕获记 failed，进程突然退出留下未知 in-flight，不只统计已保存输出。资源检查在初始化前及每个窗口开始前进行。

## 资源、初始化与观测边界

沿用已授权的桌面图形 allowlist：Xorg、gnome-shell、gnome-control-center、sunloginclient、code、chrome，只接受 G 类型；任何其他 compute owner 阻断任务，不终止对方。SDD 不恢复。

复用 exporter 的来源/输入检查和原 `_active_evaluator`，仍由原 loader 校验全量 source cache。最终 Stage-A checkpoint strict load、epoch13、strict-no-z、CUDA/BF16、K21/P20、M4/rank8、两轮 exact、corrector 输出层严格零/eval 均受现有配置及附加门槛约束。resolved args 与历史参考仅允许 model_dir/save_dir 不同。中间 legacy baseline 初始化的 missing-key 日志不等于放松最终 Stage-A strict load。

全四臂在13前做 state_dict 原生字节指纹，13后再次核验；保存 Python/NumPy/Torch CPU/CUDA RNG 的13前、encode后、diffusion后状态。此处 CPU 拷贝及同步代价是明确的共同观测成本，不能宣称复原历史 allocator/调度或 seed2035 的先前执行历史。

trace 使用限定 Python 函数 call/return 边界；复用 `_emit_diagnostic` 时的实际参数和 sampler frame 中已经完成两次 index_add 的 accumulated 引用，不重写计算，也不反算 accumulated。保存每个引用的 dtype/device/shape/stride/storage alias/version；任何后续原地变更均标无效，不能作为当时快照。保存仅在原采样成功后发生。

生产 `randn`/`randn_like` 临时 passthrough 包装记录实际 draw，返回原对象；scope 结束恢复原函数。不生成替代 noise，不跳过任何 RNG draw。x_T 的真实 CPU randn 结果记录为 CPU（原代码随后 to(device)），branch noise 按真实调用序号记录。trunk denoising 不额外抽取逐步噪声。Round0 保存实际 Gumbel、原函数中的 clamped uniform、perturbed score；clamp 前的原始 uniform 不宣称捕获。Round1/2 generator 的创建与前后 state 分开记录，不能声称 exact rounds 消费了 Gumbel。

本轮不是 operator dispatch observer。持有引用会改变 allocator 生命周期，Python profiler/包装器增加调度开销，generator.get_state 也有观测成本；CPU 验收通过不等于 GPU neutrality 通过。若无 trace 自身已变异，A/T 差异不足以证明 hook 有或无影响，应保留 uncertified。不启动确定性算法、CUDA_LAUNCH_BLOCKING、CUBLAS_WORKSPACE_CONFIG，不改 AMP、TF32、累加算法或设备。

native dtype `.pt` 和 raw-byte SHA256 保留 BF16/FP32/整数位模式；序列化重载比较不得先转 FP32。exact priorities、tuple/objective 大整数以十进制字符串保存。全量 trace 留本地，Git 仅提交摘要/元数据 hash 清单。

## 历史只读复核

重新完整解析旧 BANK_MANIFEST 的695条记录；旧/新 resolved args 仅输出目录两项不同；本次核验前一归档13项及最新证据归档4项 SHA256 均一致。EVIDENCE 的29条最终 ID 变化重新按轴核对：0/1不变；2/3/5/6/7/8仅 slot 重排；4移除10并新增16。这里的轴不是实体 ID，候选 multiset 保持不等于完整轨迹 multiset 保持。

最终 relation embedding 在最终 IDs 之后计算，不代替逐轮 relation/energy 首差证据。历史 Round0/逐轮 score/actual random payload 缺失，新运行不能自动确证旧分歧的同一根因。既有固定 bank 的 ΔJADE=0.03391811925732022m、ΔJFDE=0.0774227560742364m 保留，不被本轮替换。
