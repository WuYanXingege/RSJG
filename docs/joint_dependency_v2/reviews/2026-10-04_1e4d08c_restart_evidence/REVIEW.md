# 网页端复核归档与 seed2036/window13 只读证据盘点

日期：2026-10-04（Asia/Shanghai）。审查源码：`1e4d08c96ff058e7337c782fcdb5440d9ad631cf`。
状态：**COMPLETE_READ_ONLY_INVENTORY**；根因状态：**OPEN_INSUFFICIENT_INTERNAL_TRACES**。

## 1. 结论与范围

支持所附网页端复核保留的边界：当前固定bank的列配对优势不等于pair energy不可分解项的因果贡献、跨sequence泛化、统一避碰改善或跨进程逐位复现。未放行geometry6、Stage-B扩展、重训、energy-I消融或SDD恢复。

本轮实际执行仅为：对已存在文件做哈希与NumPy数组比较、阅读源码、整理证据和归档。不导入Torch/模型，不生成随机payload，不重算轨迹指标，不运行forward、不使用GPU、不重放。本轮72个NPZ比较是两次已保存前36窗的读取，不能算新增实验。

## 2. 网页端报告的归属

[WEB_REVIEW_ORIGINAL.md](WEB_REVIEW_ORIGINAL.md) 是用户提供文件的逐字节副本，原审查日期仍为2026-10-03。其“81,190项断言通过”等是网页端会话报告的执行记录，不冒充本轮重新执行。

原文第7节“尚未推送”描述的是提供该文件的历史会话；本轮将原文及新增本地盘点一起提交发布，未改写该历史声明。源文件SHA256：`824f51f1f7ded6aa9adc48d588791eff162d8c9d03dac478130036ce1b5b4aa9`。

本轮重新核对原1e4d08c归档清单的13个文件哈希、72个NPZ的整文件哈希/所有22个数组合同、window13的raw source/source batch/deployment cache实际文件哈希。没有重新执行上一轮4,890项baseline parity或真实配对置换。结果在[EVIDENCE.json](EVIDENCE.json)，复现只读盘点的脚本为[inventory_cpu.py](inventory_cpu.py)。

## 3. 已保存差异的准确表述

按窗口序号比较前36窗：window0–8全部保存数组一致；window9是最早的已保存差异，仅最终relation embedding不同。window10–12同样只在该字段出现微小差异，最大约8.94e-8。这不是首个内部operator的定位。

window13为N=9、E=36。22个数组中17个完全一致，包括K候选坐标、candidate log prior、GT、masks、frames、graph、edge weights、trunk ID/goal及sample weights/indices。

| 不同的已保存数组 | dtype | 不同元素数 | 最大绝对差 |
|---|---|---:|---:|
| selected_candidate_ids | int64 | 29 | 18（ID数值差，不是几何距离） |
| joint_goals_world | float32 | 58 | 2.251969 m |
| joint_goals_map | float32 | 56 | 5.5（map坐标） |
| Y | float32 | 696 | 2.265572 m |
| edge_relation_embedding_original_world | float32 | 7266 | 0.037995 |

Y首个数组差异坐标为[2,0,0,0]，selected_candidate_ids为[2,0]；这些是数组索引，不是时间戳或执行先后。完整29处ID变化保留在JSON。上述浮点数组存储dtype均相同；**不能据此声称内部operator的dtype一致或已经定位dtype首差**。

## 4. 输入与trace证据可用性

| 证据 | 当前状态 | 可以/不能支持什么 |
|---|---|---|
| checkpoint/config/cache/source指纹、resolved args | 已记录；相关模型/配置/helper指纹相同，args仅model_dir/save_dir不同 | 支持来源合同；不证明所有运行状态逐位相同 |
| window13原始数据、source batch、deployment cache文件 | 本轮只读重哈希，匹配两次sidecar | 支持输入文件相同；不等于encode后全部中间tensor相同 |
| K候选坐标、candidate_log_prior_K | 两次原始数组字节一致 | prior不是学习得到的完整unary score |
| 最终selected candidate IDs、joint goals、Y | 已保存且不同 | 证明输出和最终离散选择分歧，不能反推是哪一轮先变 |
| Round0 selected IDs、uniform/Gumbel实值 | 未在两次bank中保存 | 只有显式seed及生成规则，不能称实际payload已核验一致 |
| Round1/2 conditional scores、source/destination energy、accumulated cost | 未保存 | 无法定位conditional score首差或归因index_add/其他算子 |
| exact tie priorities、objective tuples、中间IDs | 未保存 | 规则可读，实际历史payload没有存档；本轮不重新生成 |
| agent/edge features、base relation logits、unary score与逐operator dtype/device | 未保存 | 无法建立operator级因果链 |
| 轨迹生成noise、forward前RNG state、velocity | 未保存 | 不能认证历史共享噪声或恢复当时随机状态 |
| rng_capture_serialization_unchanged | 已保存布尔值 | 只证明该次forward后捕获/序列化的RNG未变，不是RNG快照，也不是两次forward的RNG一致证明 |
| runtime/ | 两次均只有evaluation_protocol.json | 未发现该次运行的额外逐轮trace |

“未保存”限定于本轮枚举的两次导出目录与其元数据，不声称全机器所有未知归档都不存在相关材料。文件清单数量、排序清单SHA256及非窗口文件列在JSON中。

## 5. 源码证据与重要纠正

以下链接固定到实际被审查commit，现有源码未修改：

- [sampler回调及forward](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/src/models/joint_dependency_v2/joint_sampler.py#L336)：callback默认None；exporter未注册写trace的callback。
- [initial与两轮refinement](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/src/models/joint_dependency_v2/joint_sampler.py#L420)：Round0存在Gumbel初始化；当前exact persistent refinement两轮不是structured-Gumbel assignment分支，不能把其他policy的round noise当作本次已用payload。
- [最终relation计算](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/src/models/joint_dependency_v2/joint_sampler.py#L496)：selected_joint_relation在最终IDs之后计算；存盘的expected_embedding不是逐轮用于energy计算的relation tensor。因此window13该字段的变化可能跟随候选选择变化，不能将它直接判作起因。
- [模型传递到dependency_state](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/src/models/model.py#L1095) 与 [exporter保存字段](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/tools/jdv2_export_stage_a_bank.py#L122)：initial_candidate_index虽在sampler返回值中暂存，但未序列化到bank。
- [post-forward RNG检查](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/tools/jdv2_export_stage_a_bank.py#L352)：snapshot仅用于内存中比较，不写入NPZ/JSON。

第一处已保存差异、最早离散决策差异、首个内部operator分歧是三个不同问题；现有记录只支持第一个和最终决策不同，后两个的时间定位仍OPEN。不能仅因存在GPU index_add操作就归因CUDA非确定性。

## 6. 下一步边界

已有记录不足以定位内部首差。本轮到此停止，**不启动单窗口重放**，不先做energy-I消融，不改变canonical指标或bank。

若用户另行授权最小重放，申请应限定原A单窗口和原checkpoint/config，先建立无干预reference及共享输入/显式随机payload合同，再记录Round0、每轮score/energy/IDs、tie objective、生成noise/RNG与dtype。必须证明trace hook不改变RNG、dtype、算术和输出；若单窗口不复现，也不能排除原运行预热/执行历史的影响。本报告不将此设计视为新GPU任务授权。

模型、配置、权重、缓存、历史报告和运行产物均未改动；SDD未恢复。技能清理是用户同时提出的独立环境操作，不纳入模型机制证据，也不上传个人技能文件到研究仓库。
