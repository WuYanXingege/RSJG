# h→sampler 预选真实扰动的有界受控传播诊断

日期：2026-10-04（Asia/Shanghai）。基线：4bf2f4ab79e891a96315d67f2d137bac66d2f850。GPU 前提交并实际运行的源码：[066a8c007c887f913805ae6d3ce719fac1ae0acb](https://github.com/WuYanXingege/RSJG/commit/066a8c007c887f913805ae6d3ce719fac1ae0acb)。

## 1. 结论与分层状态

任务 COMPLETE_BOUNDED_DIAGNOSTIC；结果 **NO_ID_CHANGE_OBSERVED_FOR_PREDECLARED_H_PAIR**。

h_A、h_B 各4次，组内 Round0/1/2 和最终 IDs 均稳定；16个跨组配对的每轮 IDs 也全部相同，没有首个离散 A/B 分歧。所有9个 agent 每轮不同 slot 数均为0，候选 multiset 相同；最终 P20 goals 也逐位一致。相应槽位的 ID 本身一致，不是仅在重排后才一致。

**IDs 稳定不代表全片段逐位确定。** 固定同一个 h，两轮 accumulated 和 conditional scores 仍有微差，Round2 最大 score 差为7.62939453125e-6。两组 relation/energy 连续输出有稳定的组间差，但未观察到对应的离散选择变化。

| 字段 | 状态 |
|---|---|
| same_h_ID_stability | A=true，B=true；每组4次，逐轮均稳定 |
| actual_random_payload_equal | true；8次实际 Round0 Gumbel/显式 generator 状态匹配历史 P2 |
| observer_semantic_snapshot_contract | PASS_CHECKED_PROPERTIES_ONLY |
| canonical_GPU_trace_neutrality | TRACE_NEUTRALITY_UNCERTIFIED |
| historical_root_cause | OPEN |
| causal_link_to_previous29_ID_changes | NOT_ESTABLISHED |
| pair_energy_causal_contribution | NOT_ESTABLISHED |

本结果只适用于预选 h 对、冻结 seed2036/window13 其他入口及本次共同观察环境。不能证明所有同量级扰动均无影响，不能消除历史29项 IDs 分歧，也不能确认历史因果链。

## 2. 实际预算

| 类别 | attempted | completed | failed | unknown-in-flight |
|---|---:|---:|---:|---:|
| 新进程/初始化 | 2 | 2 | 0 | 0 |
| 原 h→sampler 片段 | 8 | 8 | 0 | 0 |
| 片段内原 sampler.forward | 8 | 8 | 0 | 0 |
| 额外 CPU exact 求解复核 | 0 | 0 | 0 | 0 |

P0 实际顺序 ABAB，P1 为 BABA；包含冷启动，无额外预热、重试或第五次片段。原 sampler 每次两轮×9 agent，共捕获144个实际 exact 问题；它们属于原决策过程，不是额外复核。

完整 A、net.encode、原 SocialMotionEncoder/message/GRU、GDTS history encoder、heatmap、_jdv2_encode、_jdv2_contexts、diffusion 和额外 GPU 计算重放均为0。没有追加 seed/window/h 对、放大或插值 δ、精度/确定性切换、energy-I、结构修改、Stage-B、训练或全量评估。

SDD 保持停止；初始化、各调用前及结束时检查 GPU owner，仅接受既有桌面 G allowlist，没有终止他人进程。结束后 nvidia-smi compute 查询为空。详见 [EXECUTION.json](EXECUTION.json)。

## 3. 预选 h 与冻结输入认证

唯一 h 对来自 message_layer_local_a13eae9/U0/OUTPUTS.pt：

- h_A=call1/h，raw SHA256：e61dbbf5d790db495fae8392ffde6caa7a95a25905032f88d5c8723565086a5a。
- h_B=call2/h，raw SHA256：6f63c893cb36dbbcc9753b2b30d4d373858281347caa9d6a3892afa16d6d2645。

均为真实无内部 observer 的 [9,128] FP32 返回，stride=[128,1]、offset=0；769项不同，最大差4.76837158203125e-7。按预注册原样使用，没有按下游结果筛选。

先认证 message/restart/social 历史归档及 prompt 指定的 REVIEW/RESULTS SHA256，再核验18个必要本地来源文件。非 h 输入来自 restart_trace_ea846fc/P2 真实 TARGET/TRACE，cache_id=valid-000013，N9/E36/K21/P20。

同源链条：P2 prepared inputs 与 P2 encoder 输入一致；它们与 social T0 所用原输入一致；social T0 message 输入与 message U0 使用的输入一致。原配置只有一个 message layer，message 输出与 final h 位模式及原 alias 分组一致；实际模型 output_projection 为 Identity。

保留22个已保存 tensor 输入字段及 cache_id，包括 candidates map/world、candidate_log_prior、cache graph、obs/scene/frame、world_coord。全部原生位模式、dtype/shape/stride/offset 经核验；22个 storage 彼此独立，非连续 world_coord 的 stride=[1,40,20] 保留，GPU 转移后再次校验。未运行新 encoder/候选生成。

历史观察器未保存的非 tensor scene 字段没有伪造；有显式 cache 时本方法不读取该字段或 goal_prob_map，因此传 goal_prob_map=None。缓存 edge_index 原为 FP32，由原方法 .long()；candidate_mask 由固定候选入口按原方法生成，逐轮逐调用相同。详见 [INPUT_MANIFEST.json](INPUT_MANIFEST.json)。

## 4. 原方法、权重与精度

复用原 _active_evaluator 初始化和最终 epoch13 strict-no-z A strict load，没有运行其完整模型 forward。中间 legacy baseline 允许 missing JDV2 键属于既有初始化流程，不代表放宽最终加载。198个 state 键 dtype/shape/raw bytes 逐项匹配历史 P2；全部模块 eval。

Checkpoint SHA256：699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb。

Config SHA256：bf5d4a692b0a9e841de8523f2780013147ec3e9bcb78877cc3c352571bc63e62。

Resolved args 仅允许 model_dir/save_dir 两个输出路径变化。corrector 最后层 weight/bias 严格零已核验；本片段不执行 correction arithmetic，也未切换其既有 eval bypass。

每次调用原 net._jdv2_goal_outputs(inputs, None, sample=True, include_teacher=False)。仅限定 scope 替换 social_encoder.forward：一次返回当前 h 工作 buffer 的同一对象，禁止落回原 encoder；退出/异常恢复原方法。

原方法按 h **重新计算 base relation 和 learned unary**，继而执行原 dynamic relation/energy/sampler。冻结的是 candidate_log_prior，不是 learned unary；没有给 h_B 配用 h_A 的 unary/base logits。

保持原 CUDA/BF16 autocast 和源码 FP32 islands，实际 dtype：

| 边界 | dtype |
|---|---|
| 注入 h、base relation prob/logits | FP32 |
| unary residual | BF16 |
| learned unary score、Round0 Gumbel/perturbed logits | FP32 |
| dynamic relation、effective energy、accumulated、conditional score | FP32 |
| IDs / 最终 goals | int64 / FP32 |

PyTorch2.8.0+cu128、CUDA12.8、cuDNN91002、driver570.133.07、RTX5070Ti 及 TF32/matmul/deterministic/benchmark 全部匹配原 P2。所有参数及 named buffers 逐调用不变；跨进程模型、非 h 输入和设置相同。

原方法使用 world_coord 的事后诊断日志保留，位置在 sampler 决策之后，teacher=False；标签未用于选择 h 或配置。本轮不据此生成新 Y/JADE/JFDE 结论。

## 5. 随机、共享 buffer 与观察合同

每次片段前重新设置原 sampler context=(2036,13)，原方法消费后为 None。原局部 generator 自然生成 Gumbel；未包装/替换 torch random 函数，没有强制噪声、跳过 draw 或重设 seed 隐藏异常。

八次实际 Gumbel、clamped uniform、全部显式 generator 前后状态逐位匹配 P2。initial generator 消耗；round_1/round_2 状态不变，没有伪造 exact-round Gumbel。每次实际 priorities 匹配 P2，真实 tie seed、整数 priorities/objective 保留。全局 Python/NumPy/Torch CPU/CUDA RNG 在所有调用中一致且不变。

每进程仅一个 FP32 h 工作 buffer；两份原 h 先转移，工作 buffer 预分配，每次片段外 copy_ 共4次，A/B 共用 storage/shape/stride。两个进程的 version 序列均为[1,2,3,4]，每次片段内不变。在下一次 copy_ 前完成该次有效性检查和原 dtype 转存，再解除 scope/引用，没有把早期 h 错存为最后一次值。

所有臂使用同一原 diagnostic_callback 与限定函数边界 profile；只持有真实引用，从原 sampler frame 捕获两次 index_add_ 完成后的 accumulated，不重算。不做全局 ATen dispatch，热路径无新增 clone、CPU tensor 拷贝、item、synchronize、GPU 哈希或运算；原代码自己的同步照常保留。

共同成本为 Python callback/profile、sampler 边界 CPU 账本、generator.get_state、引用生命周期、逐调用事后 native 转存，以及片段外 h copy_、输入/全状态指纹同步。历史地址/allocator/stream 和无观察执行未恢复。

7类 CPU stub 验收通过：指定对象注入/原 encoder 零调用、重复注入拒绝、输入/参数/RNG/随机函数保持、异常/scope 恢复、早期原地引用失效、每次转存保留对应 h、native FP32/BF16 与实际随机消费保持。全部 GPU trace 的 version/native roundtrip 合同通过。**这些不授予 canonical GPU neutrality PASS。**

## 6. 全部重复对照与连续传播

h_A 组6对、h_B 组6对、跨组16对，共28对。每对所有 round 及最终 IDs 均相同；每agent的不同slot数为0、multiset相同、removed/added为空。完整列表见 RESULTS.ID_pairs。

下表“同 h 最大差”取两组组内比较的最大值：

| 边界 | 同 h 不同配对数 | A/B 不同配对数 | 同 h 最大差 | A/B 最大差 |
|---|---:|---:|---:|---:|
| base relation prob | 0/12 | 16/16 | 0 | 1.1920928955078125e-7 |
| base relation logits | 0/12 | 16/16 | 0 | 4.76837158203125e-7 |
| unary residual / learned score | 0/12 | 0/16 | 0 | 0 |
| Round0 perturbed logits | 0/12 | 0/16 | 0 | 0 |
| 各轮source/destination effective energy | 各0/12 | 各16/16 | 0 | 至多9.5367431640625e-7 |
| Round1 accumulated | 12/12 | 16/16 | 3.814697265625e-6 | 3.814697265625e-6 |
| Round1 conditional score | 12/12 | 16/16 | 3.814697265625e-6 | 3.814697265625e-6 |
| Round2 accumulated | 12/12 | 16/16 | 7.62939453125e-6 | 7.62939453125e-6 |
| Round2 conditional score | 12/12 | 16/16 | 7.62939453125e-6 | 7.62939453125e-6 |

第一处已捕获的 h 依赖计算输出差异是 relation_inference 的 prob/logits；同 h 时这些边界稳定。unary 虽重新计算，却在此次原 BF16/FP32 上下文中逐位相同，不能将它写成人为冻结。

同 h 的 source/destination energy 稳定，而 accumulated 变化，显示存在下游背景数值变异；只报告该捕获区域，不追查 kernel。A/B score 差不能全部专属于 h，更不能直接推导离散翻转或历史解释。没有做 unary/relation/energy 分支消融，没有证明 pair energy 因果贡献。

## 7. CPU exact 门槛与未执行项

没有 exact Round1/2 的首个离散 A/B 分歧，不满足额外 CPU solver 复核门槛：实际额外求解和交叉 assignment 精确目标评估均为0，不能写成复核 PASS。

原片段内144个完整 exact 问题仍已捕获 score/mask/previous/geometry、tie seed/priorities、assignment、四级及 encoded objective。大整数以十进制字符串保留，没有把不同 lift 单位的 encoded objective 差解释为 margin。

没有 Y、trajectory multiset 或新 JADE/JFDE 结论；候选 multiset 一致不等于轨迹 multiset 一致。旧 bank 与 ΔJADE=0.03391811925732022m、ΔJFDE=0.0774227560742364m 完整保留。

## 8. 归档

- [RESULTS.json](RESULTS.json)：逐调用/各round IDs、28对、连续比较及分层结论。
- [POST_CHECKS.json](POST_CHECKS.json)：跨进程合同和各round score；完整内容也在 RESULTS.post_run_checks。
- [EXECUTION.json](EXECUTION.json)、[INPUT_MANIFEST.json](INPUT_MANIFEST.json)、[CPU_CHECKS.json](CPU_CHECKS.json)。
- [TRACE_CONTRACTS.json](TRACE_CONTRACTS.json)：全部 tensor 的原生bits/布局/version/alias/事件；精确大整数完整 payload 留在本地 TRACE_METADATA.json，由清单认证。
- [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)、[COMMANDS.md](COMMANDS.md)、[ARTIFACT_HASHES.json](ARTIFACT_HASHES.json)。

本地原生证据：outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/h_to_sampler_4bf2f4a/。

四个 GPU 前工具保持066a8c0原样。GPU完成后仅新增只读 postcheck，补充原分析器未单列的 conditional score 摘要；未重跑 GPU/solver 或改变原始结果。所有 tensor/checkpoint/data/cache 不提交 Git。src 树仍为 a2db08541129ba93e16d288867e7ec4fbd8e8c95，src/configs 工作区无差异，skills/Apps 未改。

## 9. 唯一下一步建议（未执行）

另行预注册**只用本轮已保存 exact payload 的 CPU 决策裕量审计**，以明确预算和原精确目标语义衡量固定问题到 assignment 翻转的余量；不追加 GPU、换 h 或放大 δ。该审计也不能直接认证历史根因。

本轮结束，不自动修生产、扩大实验、做 energy 干预、训练或恢复 SDD。
