# SocialMotionEncoder 固定输入：局部重复性与有限模块观察

日期：2026-10-04（Asia/Shanghai）。基线 `634a467868c3d6a488837be4edc18ccff52db864`；GPU前提交的工具源码 `0ffdf7c35624e2ecc9e446bc0ee0cd9f3ac1e25a`。生产src树仍为 `a2db08541129ba93e16d288867e7ec4fbd8e8c95`，工作区生产源码/冻结配置无改动。

任务状态：**COMPLETE_BOUNDED_DIAGNOSTIC**。局部结果：**LOCAL_UNOBSERVED_OUTPUT_VARIATION_REPRODUCED**。观察语义/快照：**PASS_CHECKED_PROPERTIES_ONLY**。Canonical GPU trace neutrality：**TRACE_NEUTRALITY_UNCERTIFIED**。历史根因：**OPEN**。

## 1. 实际回答

用唯一冻结的P2真实输入，原SocialMotionEncoder在独立进程中仍产生输出变异。14次无内部observer调用的h原生hash全部不同，91组两两比较全部不逐位相同，最大绝对差 `4.76837158203125e-7`。其中U0自己的四次调用就已全部不同，不依赖有hook的比较。

T0/T1仅第4次调用使用有限模块hook。两份trace的GRU输入temporal_input及实际final_hidden逐位相同；message layer的agent_feat、edge_index、edge_feat、edge_weight也全部逐位相同，而其输出有786个元素不同，最大差 `4.76837158203125e-7`。全部快照有效、布局和别名关系对齐。**本次局部观测中的首个不同模块边界为 `message_layer_0/output`，不是 temporal GRU 返回边界。**

这个结论只定位模块边界。未观察message_mlp、累加、normalizer、LayerNorm或GRU内部kernel；不能进一步声称 `index_add_` 已证实、首个canonical operator已找到、或历史原因已恢复。原输入的CUDA地址、allocator、stream时序、GRU workspace及历史运行未复原。

## 2. 来源、布局、权重与精度

先核验上一归档的报告/JSON/合同及manifest，按manifest验证P2/P3的TRACE.pt、TRACE_METADATA、STATE、PROVENANCE、RNG和RESOLVED_ARGS实际文件SHA256。没有重新生成任何输入。

| 唯一使用的P2字段 | 原始形状 / dtype | 原始stride / offset |
|---|---|---|
| obs_traj | [9,8,2] FP32 | [16,2,1] / 0 |
| edge_index | [2,36] int64 | [36,1] / 0 |
| edge_feat | [36,14] FP32 | [14,1] / 0 |
| edge_weight | [36] FP32 | [1] / 0 |
| scene_index | [9] int64 | [1] / 0 |

五者在原trace中属于不同storage group；CPU clone的逻辑字节、stride、offset与原metadata相符，GPU转移后再次核对相同。无需额外布局重建，也没有假称恢复原始内存地址。P3仅认证同源一致性，所有16次调用均使用P2这同一组输入。

原epoch13 checkpoint SHA256 `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`。严格提取真实 `social_encoder.` 前缀的16项state，`strict=True`加载到原类；全部参数字节/shape/dtype与P2/P3历史state指纹一致。这是**encoder子模块严格加载**，不冒称本轮进行了完整GDTS模型strict load或完整模型初始化。

构造依据实际JDV2源码：hidden/output=128、edge_dim=14、message_layers=1；dropout=0、dt=0.4来自已读取的原类默认及常量，均显式记录。所有子模块eval、no_grad。

Canonical A整体CUDA/BF16，但JDV2调用social_encoder原本就在 `torch.autocast(device_type='cuda', enabled=False)` 内，以FP32输入/权重计算。本轮保持此既有FP32区段，**不是关闭AMP的精度消融**。PyTorch2.8.0+cu128、CUDA12.8、cuDNN91002、driver570.133.07、GPU UUID、TF32、matmul precision等均与历史设置匹配。cuDNN deterministic=True/benchmark=False是恢复原isolated_random_seed上下文；未开启新的deterministic algorithms（仍False），未设置CUDA_LAUNCH_BLOCKING或CUBLAS_WORKSPACE_CONFIG。

每进程从认证的P2/RNG.pt恢复before13的Python、NumPy、Torch CPU/CUDA状态。该快照不是历史encoder入口状态，不声称等价恢复历史入口。在四次调用批次前后实际比较，四进程的全局RNG均未变；输入及模型state也未变。认证与哈希在批次之外完成。

完整来源及设置见 [INPUT_MANIFEST.json](INPUT_MANIFEST.json)、[EXECUTION.json](EXECUTION.json)。

## 3. 新预算及执行合同

| 进程 | 调用1–3 | 调用4 | 初始化 / 尝试 / 完成 / 失败 / 未知 |
|---|---|---|---|
| U0 | 无内部observer | 无内部observer | 1 / 4 / 4 / 0 / 0 |
| U1 | 无内部observer | 无内部observer | 1 / 4 / 4 / 0 / 0 |
| T0 | 无内部observer | 有限模块hook | 1 / 4 / 4 / 0 / 0 |
| T1 | 无内部observer | 有限模块hook | 1 / 4 / 4 / 0 / 0 |

总计4个新进程、16次原encoder调用（14次无observer、2次有限observer），无免费预热。完整A forward、net.encode、sampler、diffusion、独立子模块/算子重放均为0。CPU准备只实例化并认证真实encoder，没有CPU真实encoder forward；少量observer测试使用合成stub。没有重跑旧700组置换验收。

使用新的持久根目录 `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/social_encoder_local_634a467/`，每个进程唯一目录，存在则拒绝、初始化失败也占名额。attempt/complete逐调用持久记录，突然退出可以识别未知in-flight。本次全部成功，无重试。

四进程依次运行；初始化前及批次前后资源检查只允许自己的compute PID和已授权桌面G allowlist，均通过。最后2026-10-04 15:10:01（Asia/Shanghai）只读核验compute owners为空。SDD保持停止。未改skills或Apps。

## 4. 观察认证的三个层级

**第一层：语义及快照合同。** GPU前7项CPU验收通过：原返回对象保留、不保留未使用整段GRU输出、不增加调用或随机draw、不变输入/参数/buffers、异常传播、scope退出和异常路径移除hook、后续原地变更使别名快照失效、原dtype/raw-bit比较及真实来源认证。见 [CPU_CHECKS.json](CPU_CHECKS.json)。

实际T0/T1只在call4注册temporal_encoder和唯一message layer的pre/post hooks，记录原对象引用、metadata及执行事件；不使用Python profiler、global torch包装、dispatch或内部算子hook。GRU只保留temporal_input和返回的final_hidden，不额外保留未使用的完整序列输出。最后h直接复用正常函数返回对象。

Hook不clone、不转CPU、不.item、不synchronize、不做GPU哈希或额外计算。结束立即移除hook；批次完成后检查引用version/alias，再保存原dtype数据并验证无损序列化。所有实际快照有效。GRU final_hidden与message输入共享storage、message输出与final_h共享storage的关系保留在合同中。T0/T1全部8个trace项的dtype/device/shape/stride/offset/alias关系对齐。

**第二层：本次固定输入样本。** 16个h各有不同hash，全部120组比较不同；其中同进程比较24组、跨进程相同调用位置比较24组。14次无observer的91组比较也全部不同。不是只挑选冷启动或第4次结果。16个新h均未与任一旧P0/P1/P2/P3 h逐位匹配；这只是辅助信息，不构成历史恢复成功或失败的充分标准。

| T0/T1实际执行顺序的边界 | 位模式比较 |
|---|---|
| temporal_encoder输入temporal_input | 相同 |
| temporal_encoder返回final_hidden | 相同 |
| message_layer_0的agent_feat及3个图字段输入 | 全相同 |
| message_layer_0输出 | 不同786项，最大4.76837158203125e-7 |
| 原encoder最终h | 同一message输出对象，差异相同 |

**第三层：canonical neutrality与历史原因。** 仍未认证。无observer自身已有变异，有/无hook hash不同不能单独证明hook有或无影响。所有臂都持有四份最终h直至批次结束，调用间有公共CPU账本写入；trace臂持有额外引用并安装/执行hook，会改变内存生命周期和调度。没有“绝对零干扰”的宣称。两次有限trace不能证明GRU在所有上下文都确定，也不能确定message层内部的第一处算子差异。

原先P0/P1的29处IDs/Y分歧、P2/P3的h微差、本轮局部message输出微差是不同运行证据；不能拼成一条已证实的下游因果链。本轮未运行下游网络，因此与29处IDs变化的因果关系仍 **NOT_ESTABLISHED**。

## 5. 保留结论、归档与唯一下一步

旧固定bank及历史所有文件保留，正式ΔJADE=`0.03391811925732022m`、ΔJFDE=`0.0774227560742364m`不重算、不替换。未修生产代码，未做确定性/精度切换实验、energy-I、geometry6、Stage-B、重训或SDD恢复。

唯一建议：**另行授权仅针对已捕获相同输入的 message_layer_0 最小内部首差诊断，先验收观测合同，再细化边界；不预先指定index_add_为原因。** 本轮不继续该工作。

- [SUMMARY.json](SUMMARY.json)：实际预算与最小结论。
- [RESULTS.json](RESULTS.json)：全部120组新输出比较、64组历史h对照、局部边界和分层状态。
- [TRACE_CONTRACTS.json](TRACE_CONTRACTS.json)：每次h及两次有限trace的dtype/shape/stride/offset/alias/version/raw hash。
- [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)：持久文件路径、大小及SHA256；原生tensor留本地，不把路径当作公开下载地址。
- [COMMANDS.md](COMMANDS.md)：实际CPU与GPU执行命令。本轮仅上传新工具、报告/JSON和审查索引，上传及远端回读状态见交付的固定commit链接。
