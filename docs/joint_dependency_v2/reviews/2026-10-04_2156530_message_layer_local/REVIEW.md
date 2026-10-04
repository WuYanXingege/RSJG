# Message layer 固定输入有界诊断

## 1. 结论与执行边界

本轮有界任务 COMPLETE。原 message 层在无内部 observer 时已复现输出变异；有限 trace 的首个有效不同边界为**两次 feature index_add_ 均完成后的 raw aggregated**。该区域的真实入口逐位相同。经此证据门槛，T1 内进行两次固定入口的独立 source index_add_，复现原语输出变异，随后提前结束，不耗尽剩余预算。

结论分层：

- LOCAL_MESSAGE_VARIATION_REPRODUCED。
- FIRST_VALID_LOCAL_REGION_DIFFERENCE：feature_accumulation/raw_after_both_adds。
- CONTROLLED_PRIMITIVE_VARIATION_REPRODUCED：独立阶段 A 的 source index_add_。
- canonical_GPU_trace_neutrality = TRACE_NEUTRALITY_UNCERTIFIED。
- historical_root_cause = OPEN；causal_link_to_previous29_ID_changes = NOT_ESTABLISHED。
- 未识别具体底层 kernel、调度或归约顺序原因；不从源码预设归因。

没有生产修复、结构修改、训练、完整评估、精度/确定性切换。SDD 保持停止；skills/Apps 未改。旧 bank 与 ΔJADE=0.03391811925732022m、ΔJFDE=0.0774227560742364m 未重算或替换。

## 2. 版本与来源

- 基线：a13eae9e1bd142b95b1cb0ed26667fec0f56cae4。
- 本轮 GPU 实际源码：[2156530d2d7bc15e4b16c53d7e1eb4120ad2dd38](https://github.com/WuYanXingege/RSJG/commit/2156530d2d7bc15e4b16c53d7e1eb4120ad2dd38)。四个独立工具先完成 CPU 准备并提交，再启动 GPU；此后工具未改。
- 生产 src 树：a2db08541129ba93e16d288867e7ec4fbd8e8c95；工作区 src/configs 相对基线无差异。
- checkpoint SHA256：699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb。
- config SHA256：bf5d4a692b0a9e841de8523f2780013147ec3e9bcb78877cc3c352571bc63e62。
- 严格提取 social_encoder.message_layers.0.，10 个 state 键逐项匹配上轮 encoder_state；原 _SocialMessageLayer(hidden_dim=128, edge_dim=14, dropout=0.0)。
- 先核验上轮归档文件及 T0/T1 的 TRACE、合同、来源、初始化、RNG、完成记录 SHA256，再读取实际输入。所有本轮调用仅使用旧 T0 输入，旧 T1 只认证同源一致性。
- 上轮报告及 JSON 的摘要与 prompt 指定值一致，完整来源见 [INPUT_MANIFEST](INPUT_MANIFEST.json)。

## 3. 预算与全部调用

| 类别 | attempted | completed | failed | unknown-in-flight |
|---|---:|---:|---:|---:|
| 新诊断进程/初始化 | 4 | 4 | 0 | 0 |
| 原 message forward | 8 | 8 | 0 | 0 |
| 额外独立 index_add_ | 2 | 2 | 0 | 0 |
| 完整 A / net.encode / encoder / GRU / sampler / diffusion（各项） | 0 | 0 | 0 | 0 |

U0、U1 各两次无内部 observer；T0、T1 各一次无内部 observer、一次有限 observer。共六次无内部 observer、两次有限 observer，包含各进程首次冷调用，没有额外预热或重试。

额外计算只在 T1 已完成原两次调用并通过门槛后执行；阶段 A 两次出现变异即停止，阶段 B 为 0，normalizer/submodule 独立调用均为 0，没有第五个诊断进程。

资源门禁在初始化、批次前后、每次额外调用前与结束时检查。仅接受既有桌面 G allowlist；未终止外部进程。四个进程均完成。结束后的只读 nvidia-smi compute 查询为空。详见 [EXECUTION](EXECUTION.json)。

## 4. 输入、模型、RNG 与布局合同

四个实际输入为 agent_feat [9,128] FP32、edge_index [2,36] int64、edge_feat [36,14] FP32、edge_weight [36] FP32。agent_feat 来自上轮真实 GRU 返回，raw SHA256：

058bf0d81ce4260309e76e22ca91c1e1eee10545de0dd573936ef0bb9394656d

四字段 contiguous、offset 0、彼此不同 storage，无需布局重建。原 forward 从同一个 edge_index 得到 source/target，二者仍共享其 storage，offset 分别为 0/36，stride 均为 [1]。T0/T1 全部对应边界 dtype、shape、stride、offset、alias 分组一致。

模型 eval/no_grad，保留原 CUDA FP32 island（autocast disabled）。PyTorch 2.8.0+cu128、CUDA 12.8、cuDNN 91002、driver 570.133.07、RTX 5070 Ti；cudnn deterministic=True、benchmark=False、allow_tf32=True，matmul allow_tf32=False、precision=highest，deterministic_algorithms=False，均匹配上轮设置。未新增环境精度/确定性干预。

复用经认证的上轮 T0 批次开始 RNG 状态；这不是历史 message 入口的完整状态恢复。四进程输入、参数/buffers 及所有全局 RNG 在批次前后不变；T1 额外诊断后 RNG、模型、输入及捕获引用 version 也不变。原 CUDA 地址、allocator、历史 stream/workspace 不可恢复。

## 5. CPU 与观察合同层级

[CPU_CHECKS](CPU_CHECKS.json) 的 8 类验收通过：原返回对象和 CPU 数值、调用次数/RNG/输入/参数不变、scope 恢复、异常传播、原地修改导致早期引用失效而终态引用有效、条件门槛、native bits（含 signed zero/dtype 区分）及真实输入/权重认证。CPU 真实 message forward 为 0，执行的是小型 CPU fixture。

有限观察使用父 Sequential/update/norm hooks 和只接受原 forward code frame 的 Python line observer；原 forward 未复制或改写。热路径只持有原对象并记录布局/version/alias，未插入 clone、CPU 转存、item、synchronize、GPU 哈希或额外 GPU 运算。退出时恢复 trace/hooks；最终返回是原 norm 返回对象。

两次 feature add 后才持有 raw aggregated，两次 normalizer add 后才持有 normalizer，version_at_capture=version_at_dump=2；除法产生不同输出。没有伪造第一次 add 的快照，也没有将被 inplace ReLU 修改的中间 Linear 输出当作有效快照。全部输出/trace 均通过 native dtype 序列化回读及 version 检查，无 INVALID_SNAPSHOT。

这些检查证明已检查的语义/快照合同；**不能认证 canonical GPU neutrality**。Python hook/frame 调度与引用保留会影响运行上下文，公共账本和保留两次返回也有成本。无 observer 自身已变异，因此跨组差异不能单独证明 observer 有或无影响。

## 6. 原 message 返回的局部重复性

全部 8 个返回的 raw hash 均不同：28/28 对不同；最大绝对差 4.76837158203125e-7。

- 六次无内部 observer：15/15 对不同，最大差同上。
- 两次有限 observer：1/1 对不同，最大差同上。
- 同进程两次返回：4/4 对不同。
- 跨进程相同调用位置：12/12 对不同。

完整逐对比较及每次原生输出合同见 [RESULTS](RESULTS.json)，不挑选匹配旧 hash 的输出，也不把小样本结果推广为所有输入的性质。

## 7. 按真实执行顺序的首个有效边界差异

顺序由 TRACE.pt 的插入顺序与 observer events 核验，不按 JSON 字母排序推断。

| 边界（按执行顺序） | T0/T1 不同元素 | 最大绝对差 |
|---|---:|---:|
| 四个原入口 | 0 | 0 |
| to_source MLP 实际输入及最终输出 | 0 | 0 |
| to_target MLP 实际输入及最终输出 | 0 | 0 |
| source/target 索引、两份加权 message | 0 | 0 |
| raw aggregated：两次 feature add 后、除法前 | 249 | 4.76837158203125e-7 |
| normalizer：两次 add 后 | 0 | 0 |
| normalized aggregated | 249 | 5.960464477539063e-8 |
| update MLP 实际输入 | 249 | 5.960464477539063e-8 |
| update MLP 最终输出 | 616 | 1.1920928955078125e-7 |
| norm 实际输入 | 506 | 1.1920928955078125e-7 |
| norm 最终输出 / 原 message 返回（同一对象） | 570 | 4.76837158203125e-7 |

因此原有限 trace 只定位到**两个 feature index_add_ 组成的区域**；没有原运行第一次 add 后的独立快照，不能从该 trace 宣称已区分第一/第二次原语。MLP/归一化/LayerNorm 的下游差异不能独立归因为它们自身。

## 8. 有条件的独立原语证据

T0/T1 第一个有效差异区域入口（原 agent feature、实际 source/target 索引与加权消息）逐位一致；快照、顺序、布局、模型、设置、CPU 语义合同均通过，EXTRA_GATE.external_contract=true。由此仅选择 feature accumulation 区域。

阶段 A 使用 T1 捕获的真实 source 索引及加权 source message，按原 zeros_like 定义每次分配独立 [9,128] FP32 全零 accumulator，调用原 accumulator.index_add_(0, source, message_to_source)。两次调用完整入口记录相等，dtype/shape/stride/offset/raw bits 相同；每次返回独立保留，未在同一 accumulator 上重复叠加。

两次结果有 **173 项不同，最大绝对差 4.76837158203125e-7**，首次数值差坐标 [0,2]，分别为 -1.1350774765014648 / -1.1350773572921753。原生输出 SHA256：

- A/1：f75dfdff2a31808de4161c6f890e23cedc7b9bb333f6d3e41956b84cbf18986d
- A/2：3758352429b32630d54a60b4cc96b2c4105d73d1336e0ca90376d176fe0e0f40

准备成本明确为 2 次零分配、0 次 pre-state clone；另有输入 CPU 哈希和每次返回 CPU 复制/同步、独立返回保留。这些改变执行上下文，全部在原 message 批次外进行，observer 已关闭。

**全零入口是构造的诊断状态，不是捕获的原运行中间快照。** 此证据仅证明这个原语在此次固定输入、局部环境中出现输出变异。不认证历史第一次 add 的首差、具体底层实现原因、canonical neutrality 或到历史 29 项 IDs / Y 差异的因果链。

## 9. 归档与复核

- [RESULTS.json](RESULTS.json)：全部逐对结果、真实首差、条件门槛、额外调用入口与输出。
- [EXECUTION.json](EXECUTION.json)：全部预算计数、初始化、资源、设置及状态验证。
- [INPUT_MANIFEST.json](INPUT_MANIFEST.json)、[TRACE_CONTRACTS.json](TRACE_CONTRACTS.json)、[CPU_CHECKS.json](CPU_CHECKS.json)。
- [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)：本轮 88 个本地原生/JSON artifact 的大小与 SHA256。
- [COMMANDS.md](COMMANDS.md)：独立脚本、执行命令与归档复核；[ARTIFACT_HASHES.json](ARTIFACT_HASHES.json)：公开归档文件摘要。

原生证据根目录：

`outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/message_layer_local_a13eae9/`

全部原始返回、trace、RNG、独立计算返回留本地，不上传 checkpoint、数据、缓存或 tensor 文件。本轮只新增独立工具和审查归档，保留全部历史结果。

## 10. 唯一下一步建议（未执行）

先预注册一个有界的 h→sampler 受控传播诊断：固定其余实际输入、候选与随机 payload，只替换两份已保存的局部 h，逐轮核对 score/IDs，以检验该量级扰动是否足以改变选择。须另行授权并设计可认证输入/观察合同；即便改变选择，也只证明受控环境的敏感性，不能直接等同历史根因。

本轮到此结束，不自动修生产、扩大采样、启用确定性模式或恢复 SDD。
