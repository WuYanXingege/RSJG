# Stage-A seed2036/window13：有界重启分歧诊断

日期：2026-10-04，Asia/Shanghai。Probe 源码（先提交、后运行）：`80ee0e2c1762b019a43e7297e12e2fd880a8ed03`；接续基线：`ea846fcb3dafeff45dbe5ae841b477c36b6676be`。

任务 **COMPLETE_BOUNDED_DIAGNOSTIC**；结论 **REPRODUCED_OUTPUT_DIVERGENCE_TRACE_INCONCLUSIVE**。Trace neutrality：**TRACE_NEUTRALITY_UNCERTIFIED**。历史根因：**OPEN**。

## 1. 结论先行

两个新建、无内部 trace 的进程 P0/P1 再次产生 window13 的29处最终候选 ID 差异，Y有696个元素不同，最大绝对差 `2.2655719965696335 m`。两者 window0–8 的全部22个保存数组一致；window9–12 先出现最终 relation embedding 微差。原路径的跨进程分歧已真实复现，不能写成“未复现”。

有 trace 的 P2/P3 最早捕获差异出现在 `SocialMotionEncoder.forward` 返回的 FP32 `h[9,128]`：690个元素不同，最大差 `4.76837158203125e-7`；其5个捕获输入 tensor 的 dtype、shape、raw bytes 均相同，模型参数/buffers 与执行设置也对齐。但 P2/P3 的 Round0、Round1、Round2 IDs 以及最终 Y 全部逐位相同。

因此，**SocialMotionEncoder 输出是本次 traced pair 的最早捕获边界线索，不是已认证的 canonical 首个 operator，更不是历史根因确认**。P0/P1 未保存 Round1 内部 payload，只能确认 Round0 IDs 一致、最终 IDs 不同，首次离散分歧属于 Round1 或 Round2，无法进一步区分。不能把 P2/P3 的逐轮一致拿来补齐 P0/P1 的缺失 trace。

## 2. 实际预算与保护边界

| 项目 | 实际数量 |
|---|---:|
| 新进程启动 / 初始化尝试 / 初始化成功 | 4 / 4 / 4 |
| 完整 A forward 尝试 / 完成 | 56 / 56 |
| forward 失败 / 未知 in-flight | 0 / 0 |
| 额外 GPU region/encode-sampler 诊断 | 0 / 4允许上限 |
| 每进程前缀 | seed2036，windows0–13，共14窗 |
| 实际 exact 问题 | P2、P3 各18，共36 |
| 正式 CPU actual-payload 求解复核 | 每问题重复2次，共72次；另有P2预检36次 |

8项小型 CPU observer 验收在 GPU 前通过，不重复旧700组置换审计。四个进程顺序运行；每个窗口开始检查资源，共60份初始化/窗口资源门槛记录，没有其他 compute owner。运行后 `2026-10-04T06:29:53Z` 再查 compute owners 为空。保留已授权桌面 G 进程，未终止其他进程，SDD 未恢复。

保持原 epoch13 strict-no-z A checkpoint（SHA256 `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`）、原 CUDA/BF16、K21/P20、M4/rank8、两轮 exact 和严格全零 corrector 的既有 eval bypass。原 loader 校验缓存、最终 strict load、resolved args 对照均通过。四臂 state_dict 原生字节指纹前后不变，全部模块 eval。未更改生产源码、配置、权重、缓存、sampler、精度、指标或任何旧 bank。

没有重训、energy-I、geometry6、Stage-B 或全139窗五seed重评估。未恢复或修改 skills，未卸载 Apps。源代码与完整命令/观测代价见 [PROTOCOL.md](PROTOCOL.md)；实际启动 PID、初始化、资源、设置及预算见 [EXECUTION.json](EXECUTION.json)。

## 3. 新运行回答了什么

| 问题 / 边界 | P0/P1 无内部trace | P2/P3 有trace |
|---|---|---|
| 最早不同的保存窗口 | 9，仅最终embedding | 同样存在前缀embedding微差；详细逐窗见RESULTS |
| window13 原始/缓存输入及模型状态 | 对齐 | 对齐 |
| 三个 global RNG 边界 | 13前、encode后、diffusion后全相同 | 同样全相同 |
| Social encoder h | 保存的aux中已有微差：554项、最大3.5762786865234375e-7 | 最早捕获函数返回首差：690项、最大4.76837158203125e-7 |
| Round0 IDs | 一致 | 一致；实际Gumbel、perturbed logits也一致 |
| Round1 / Round2 IDs | 未捕获 / 最终不同29项 | 两轮都一致 |
| 实际 diffusion noise | 无内部hook，不宣称捕获 | 121次draw逐位相同 |
| 最终 Y | 不同，最大2.2655719965696335m | 逐位相同 |

P0、P2、P3 的 window13 IDs/goals/Y 一致，P1 不同；四者最终 relation embedding 仍可有微差。所有6个进程配对的输入状态与三个 global RNG 边界均对齐。Global RNG 相同不等于每个无trace运行的实际payload都已直接核验；P0/P1 没有保存内部noise及局部generator states，不能补造。

最小证据与五个 social encoder 输入的原始哈希/布局在 [FIRST_DIVERGENCE.json](FIRST_DIVERGENCE.json)。全6组配对、逐窗输出、window13目标字段、P2/P3内部张量差异及CPU solver结果在 [RESULTS.json](RESULTS.json)。其中 `first_captured_tensor_difference` 按实际捕获插入顺序选取，不按JSON字母排序推断因果。

## 4. Trace 内容、位模式及无干扰边界

P2/P3 各记录384个 tensor 项（别名重复项不是独立样本），包含原 candidates/prior/learned unary/h/edge/base logits、Round0 actual Gumbel及clamped uniform、两轮source/destination relation与energy、两次index_add完成后的实际accumulated、conditional scores、每agent exact输入/previous/geometry、实际priorities与整数objective、最终IDs/goals/relation及真实diffusion draws。参数、buffers、原encode contexts、velocity、Y以及全局/显式generator states另有持久记录。

全部捕获引用在延迟转存前版本未变，没有无效快照。原生dtype序列化回读通过，包括真实 trace 中的BF16；没有先统一FP32再声称bitwise一致。大整数objective/priorities以十进制字符串保存，未转float。完整 tensor metadata（dtype/device/shape/stride/alias/version/raw hash）见 [TRACE_CONTRACTS.json](TRACE_CONTRACTS.json)。

实际diffusion draw共121次：1次CPU `randn` 产生x_T（原代码随后转GPU），20分支×6次`randn_like`；trunk不额外抽取逐步noise。Round0实际消耗initial generator；round_1/round_2虽创建，前后state未变。没有伪造exact rounds的Gumbel。`_unit_gumbel` 中只保存实际clamp后的uniform，不宣称捕获clamp前原始uniform。

CPU测试通过仅证明已测试的观察语义。实际GPU热路径没有追加tensor clone/CPU复制/GPU计算，但持有引用改变allocator生命周期，Python profile/wrapper和generator state观测有调度成本。四臂共同的window13前state指纹/RNG快照也有同步成本。没有认证这些观测与历史无观测运行的全状态等价。

**P0/P1 自身变异，而 P2/P3 IDs/Y 一致，不能证明 hook 修复了问题，也不能证明 hook 没有副作用。** 故不授予GPU trace neutrality PASS；本轮不进入额外region或operator跟踪。56次完整forward预算已用尽，工作转入归档，不继续抽样。

## 5. Exact 求解复核与归因限制

P2/P3 Round1 conditional score分别有1079项不同、最大差3.814697265625e-6；Round2有1124项不同、最大差5.7220458984375e-6。各自完整score/mask/previous/geometry和**实际priorities**交给原CPU exact solver，每个问题重复两次；36个问题的assignment、四级objective、encoded objective均与各自原执行匹配。没有用epsilon tie或普通float Hungarian替代。

不同实际score保持各自完整问题，不交换previous或tie payload。P2/P3 actual priorities一致，objectives可不同，但assignment一致。这只能说明捕获的两个真实问题被原exact solver稳定重现；**本轮没有在已捕获逐轮trace中证明这些score微差造成了P0/P1的29项ID变化**。不同问题的整数lift单位还可能不同，不能直接把encoded objective整数差解释为连续能量差。

没有捕获 SocialMotionEncoder 内部的第一处 operator 差异，没有 operator dispatch/原地快照认证，无法区分内部GRU、message计算、累加、normalization或更底层kernel/调度。源码中存在 `index_add_` 不是归因证据；后续relation/energy/accumulation已经接收到不同上游输入，不能仅因其输出不同就指定该region为第一原因。

历史内部noise/逐轮score/tie payload仍缺失。新P0/P1输出分歧的规模与历史记录匹配，亦不自动证明历史与新运行共享同一个实际内部根因。最早window13边界也不是整个run最早内部差异。

## 6. 历史结论及唯一下一步

旧bank和固定bank配对结果完整保留；正式ΔJADE `0.03391811925732022m`、ΔJFDE `0.0774227560742364m` 不被替换。既有正向配对结果不因此升级为energy交互因果贡献；也不因本次重启分歧自动作废。

唯一建议：**另行授权仅针对已捕获相同输入的 SocialMotionEncoder 最小诊断，先建立观测无干扰合同，再考虑细化operator首差。** 本轮不执行这一后续，不修生产代码、不重训、不做energy-I，不恢复SDD。

## 7. 归档与复核入口

- [RESULTS.json](RESULTS.json)：完整比较与实际预算，状态明确区分复现/neutrality/历史根因。
- [FIRST_DIVERGENCE.json](FIRST_DIVERGENCE.json)：首差线索、随机流及决策边界。
- [TRACE_CONTRACTS.json](TRACE_CONTRACTS.json)：所有捕获tensor的原生dtype/raw哈希与别名/版本合同。
- [TRACE_MANIFEST.json](TRACE_MANIFEST.json)：持久运行文件路径、大小与SHA256；大型 `.pt`/NPZ不提交Git。
- [CPU_CHECKS.json](CPU_CHECKS.json)、[EXECUTION.json](EXECUTION.json)、[PROTOCOL.md](PROTOCOL.md)：验收、执行及可复核命令。

大trace位于仓库相对目录 `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/restart_trace_ea846fc/`。网页端能读取本次提交的摘要、合同和hash，但不能把本地tensor路径当作公开下载地址。上传状态以最终固定commit推送及远端字节回读为准。
