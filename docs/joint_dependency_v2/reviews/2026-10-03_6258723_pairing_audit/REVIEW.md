# 固定完整轨迹配对审计：输入缺失与 CPU 实现验收

日期：2026-10-03（Asia/Shanghai）。状态：**BLOCKED_MISSING_TRAJECTORY_BANK**。

**结论：当前不能判定真实 Stage-A joint 列配对是否有收益。** 本轮没有获得可认证的 post-fix A 完整轨迹 bank；真实审计为 **0 scenes、0 个可用 inference seeds**，不能用旧 D 导出或合成数据替代。已完成有边界的输入定位、既有证据指纹复核、预登记输入合同和独立 CPU kernel 验收。没有重跑 diffusion、没有启动 CUDA/训练，也没有修改 src、冻结配置、sampler、权重或历史结果。

唯一下一动作：**补齐可认证的 post-fix A 完整 trajectory bank**。可提供已有归档位置；若确实没有，再另行授权一次独立 A 导出作业。本 prompt 不授权该 GPU 导出，因此本轮止于输入门槛，不推进能量消融或结构实验。

## 1. 源码、输入生成与归档提交分开记录

| 项目 | 本轮事实 |
|---|---|
| 分支 / 实际审查 HEAD | research/joint-dependency-v2-clean / 6258723915f3842ee974aea6f8744cd685d5f047 |
| 相对 6258723 的已有提交变化 | 无；开始时已跟踪工作树干净，已有未跟踪运行目录保留 |
| 相对 1f07e7a 的 src/configs/tools 变化 | 无；6258723 是上一轮审查归档提交 |
| 历史身份与五种子 artifact 生成源码 | d1a1382ae324b9096ac3a6d8de812ffbbb8d9780 |
| 本轮真实 bank 生成提交 | null：没有已接受输入，不伪造 |
| 本轮归档提交 | 由包含本报告的 Git commit 及最终固定链接确定；JSON 不自引用未来 SHA |
| AGENTS.md | 祖先与相关工作区未找到适用文件 |

本轮完整回读上一份 REVIEW、索引、checks.json、identity fix validation，解析三个 post-fix JSON，并查阅其关键源码和 manifest。未重复旧的63组摘要复算、2940个 gauge赋值或60个合成scene测试；本轮新增测试针对新 kernel 的执行正确性。

**CONFIRMED**：上一报告 SHA256 为 046e17ac45676c2a6e9f2510a69416a4e37a92437763792bdfa4b1fb39916201，checks.json 为 fc255a4d5d72aaf79256f617931ffee2d6e5e0a838fdca75979c471029a8bb91，与 prompt 一致。BF16、FP32、paired-five-seed JSON 指纹仍分别为 3df9d0b…、8796e600…、4c1103cf…；完整指纹见 INPUT_MANIFEST / DISCOVERY。

### 已完成／仍开放／证据缺失

| 分类 | 状态与边界 |
|---|---|
| 已完成 | CONFIRMED：严格全零 Stage-A 推理 bypass、历史 A=B=C 身份认证已关闭，不重新列为修复项 |
| 已完成 | CONFIRMED：post-fix 显式2035–2039 C/D存在，C仅为correction-off基底，D才是实际V2-A；不是本轮新增评估 |
| 本轮完成 | CONFIRMED：输入搜索、旧证据hash、100个置换master seeds预登记、新CPU kernel与非法输入验收 |
| 仍开放 | OPEN：真实完整 bank 的 pairing 效应、真实bank CPU与原evaluator一致性、sequence/block不确定性 |
| 证据缺失 | CONFIRMED于搜索范围：2035、2036、2037、2038、2039均无可认证完整输入；实际指标、tie率与效应均为null，不是0 |
| 仍开放 | 已确认的seed/RNG resume/原子保存/cache安全缺口仍未修；源码未变，只沿用旧最小修复计划，不扩展本轮 |
| 不放行 | geometry4→6、扩展B/容量/轮数仍NO-GO，没有新的真实失败或收益证据 |

A/B/C/D 语义沿用[身份修复认证](../2026-10-03_1f07e7a/REVIEW.md)：A是canonical Stage-A，B同A checkpoint correction off；C使用V2-A checkpoint但关闭correction；D开启correction。匹配 checkpoint 不等于证明某个 bank 的 A=C：使用C时仍须提供适用于该bank、同输入的身份认证。

## 2. 输入定位与拒绝理由

搜索快照：2026-10-03 18:17:25 +08:00。详细命令、排除目录、候选路径及manifest摘要见 [DISCOVERY.json](DISCOVERY.json)。

**CONFIRMED**：在 GDTS / RSJG_JDV2_clean 两项目中枚举66,847个文件名，发现30个manifest路径；该范围未发现 NPZ/NPY/H5/HDF5 或 tar/zip 等匹配的bank/归档容器。检查了项目相关 /tmp 位置及 Downloads/Desktop/Documents 中的相关文件名（外部目录最多5层）。没有提供另一个云端或外部归档位置；未扫描无关账户或全盘，也不主张全世界不存在该bank。主动排除了数据集和活跃 SDD cache/run 目录，没有反序列化大型模型、checkpoint或未知pickle。

| 已定位对象 | 证据 | 处理 |
|---|---|---|
| 旧 stage_b_v2a_epoch005_seed2035_commit05378c7 JMM文本导出 | manifest source05378c7、V2-A epoch5、e4c114…checkpoint、253scenes/seed2035；5,060个sample文本文件 | 拒绝：旧D/correction-on与JMM253口径，不能替代post-fix A139窗 |
| 不带commit后缀的旧同名导出 | 同样253scenes、V2-A checkpoint、5,060个sample文件；manifest未给出source commit | 拒绝：旧D协议且来源更不完整 |
| eth_full_stage_a cache | schema=jdv2-cache-v1、K21、goal checkpoint126acf… | 拒绝：goal候选/训练输入cache不是完整预测Y |
| Stage-A freeze manifest | epoch13、699336…checkpoint、接口/旧指标 | 只作为冻结与checkpoint元数据；不能从汇总反演完整轨迹 |
| old GDTS epoch010 final_results .pt / epoch011 reference checkpoint | 名称与已知来源不同，无合格post-fix A bank证明 | 不反序列化；按未经认证输入排除，不借用其结果 |
| 既有post-fix paired JSON | 逐seed metrics/parity/counts，没有保存Y | 只作为历史认证证据，不能用于轨迹重排 |

拒绝旧导出的D路径解释来自其既有运行配置/审查；manifest本身没有显式correction flag，不把推断伪写成已有字段。

**CONFIRMED 新核实点**：通用 evaluator 的 docstring 虽说保存轨迹，实际循环在计算metrics后删除 all_output（[trainer.py::_evaluate_epoch_unseeded，1657–1721行](https://github.com/WuYanXingege/RSJG/blob/6258723915f3842ee974aea6f8744cd685d5f047/src/trainer.py#L1657)）。专用paired工具也仅累积metrics/parity/counts，最终JSON没有完整Y（[evaluate_seed / main，167–197及250–279行](https://github.com/WuYanXingege/RSJG/blob/6258723915f3842ee974aea6f8744cd685d5f047/tools/jdv2_postfix_paired_five_seed_evaluation.py#L167)）。因此“曾经跑过评估”不等于已有可重排bank。

上述是有界搜索与源码证据，不是所有二进制文件内容已穷尽认证。只要提供另一个bank，其内容和provenance仍须单独验收。

## 3. 输入合同与预登记

[INPUT_SCHEMA.json](INPUT_SCHEMA.json) 是明确的结构/语义合同，不冒充已运行的JSON Schema验证器；[INPUT_MANIFEST.json](INPUT_MANIFEST.json) 在第一次新kernel self-test之前写入，实际inputs与available seeds均为空。

建议未来每个 source/window/scene/inference-seed 单独一个只读numeric NPZ，allow_pickle=False；也可针对已存在的其他格式设计无损adapter，不要求重新生成轨迹。

| 必需内容 | 逻辑形状或语义 |
|---|---|
| 完整future Y / GT | Y[N,P,T,2]，GT[N,T,2]；实际N/P/T、native dtype从artifact读取，不把20/12硬编码进kernel |
| masks | metric_mask[N]为完整观测/未来agent资格；future_mask[N,T]；当前adapter拒绝缺失sample或被评分agent缺少完整future |
| stable IDs / sample indices | source/window/scene/agent/frame/inference-seed身份；每agent原始trajectory ID，不混用K21候选轴 |
| graph | edge_index[2,E]，canonical不重复、同scene，history-only部署graph来源与hash；不得由GT未来重建 |
| sample weights / auxiliaries | 当前支持等权完整bank；非均匀/不完整输入fail-closed，不自行重加权。存在的goals/IDs/sample masks/weights使用同一P轴gather |
| provenance | actual A或有同输入认证的C、生成commit/dirty digest、checkpoint/config/data/cache/坐标变换指纹、split/selection、bank文件hash |
| baseline scores | 与bank同次forward产生的逐agent/scene及总指标，先做CPU parity；旧C均值不能独自认证不相关A bank |

置换协议已冻结：

- 100个master seeds：625872300–625872399，完整列表写入manifest，不读取结果后挑选。
- PCG64局部Generator；用SHA256派生包含source/window/scene/inference-seed及agent稳定ID的typed JSON key，不用Python随机hash、不改全局RNG。
- Original、scene共同列置换、per-agent独立列置换；每seed保留自己的P条轨迹，不拼P100、不跨scene。
- 精确逆置换检查完整Y及所有sample-aligned字段的逻辑顺序bytes/dtype；GT、mask、graph不变。
- 数值不变量预设 atol=rtol=1e−12（float64 CPU adapter）；真实bank与export原evaluator的候选容差预设 atol=rtol=1e−6，超出则停止定位原因，不事后放宽。
- practical ε_joint=null：目前没有合理依据规定“值得优化”的小数阈值，因此未来先报告完整效应与分布，不按临时阈值宣布Go。

**CONFIRMED 定义边界**：minADE/minFDE来自每agent minima；JADE/JFDE在每scene共同slot上先跨agent/time归约再取min。跨scene汇总marginal按有效agent实例加权，joint按scene平均，不能统一直接平均每scene的minADE。定义见 [metrics.py，238–325行](https://github.com/WuYanXingege/RSJG/blob/6258723915f3842ee974aea6f8744cd685d5f047/src/metrics.py#L238) 与 [_metric_append/_metric_finalize，410–426行](https://github.com/WuYanXingege/RSJG/blob/6258723915f3842ee974aea6f8744cd685d5f047/tools/jdv2_stage_b_v1_failure_mechanism_audit.py#L410)。

Collision使用仓库 [collision_rates_per_sample，439–503行](https://github.com/WuYanXingege/RSJG/blob/6258723915f3842ee974aea6f8744cd685d5f047/src/jmm_protocol.py#L439)：future-only连续分段线性、半径0.1m、碰撞agent比例。它在139窗上的名字是JMM公式诊断，不是官方253scene benchmark，也不混称legacy Collision_Rate。官方CR-JADE保留first-index argmin；另报exact tie-set mean/min/max和tie发生率，不伪造独立重组world的稳定原ID。

RME仅在full-mask单scene下按原定义实现；partial-mask时明确省略，不在审计中偷偷修补生产指标（[model.py，1821–1843行](https://github.com/WuYanXingege/RSJG/blob/6258723915f3842ee974aea6f8744cd685d5f047/src/models/model.py#L1821)）。空有效scene明确拒绝并应由未来loader记录排除计数，不把legacy零值静默平均进去。

## 4. 本轮实际CPU验收，而非真实Stage-A效果

新脚本：[audit_cpu.py](audit_cpu.py)。discover读取元数据；score/gather/audit_scene是可复用kernel；self-test验证新实现。**尚未认证真实bank读取器、完整provenance门禁及端到端原指标parity**，因为没有可用输入，不能把kernel测试通过等同于真实审计完成。

| 新检查 | 实际结果 |
|---|---|
| 7类fixture | singleton、多人E0、mixed、all-active、partial-agent-mask、exact CR-JADE tie、可控joint-change |
| 各fixture置换数 | 100组共同+独立对照；沿用预登记100 seeds，共700组，不是700个真实模型scene |
| 完整集合/aux保持 | 逆置换bytes/dtype恢复通过；输入Y/GT/mask/graph bytes未变 |
| minADE/minFDE与common/N1控制 | 全通过；可要求不变的common指标最大绝对差为0 |
| 仓库CPU指标对照 | 直接调用src/metrics.py的4项CPU指标，obs_length=0对应已截取future；最大差4.440892098500626e−16 |
| exact-tie例外 | 100次共同置换中94次改变first-index CR-JADE；tie-set附加指标通过不变性检查 |
| 稳定随机流 | Python/NumPy全局RNG未变；更换agent行次序但保留ID后随机流正确输运；两个PYTHONHASHSEED独立进程一致 |
| fail-closed | 11项非法输入均拒绝：重复agent ID、无有效agent、NaN、不完整有效future、缺sample、非均匀weights、pixel单位、越界edge、重复edge、aux轴错误、非双射 |
| 真实A bank | **NOT_RUN：0 scenes；真实CPU baseline parity、pairing效应、真实tie率、cluster uncertainty全为null/未运行** |

exact-tie的具体fixture：两人GT位于(−1,0)/(1,0)，一个world两人都在(0,0)，其余world位于(−2,0)/(2,0)，每个world JADE均为1但碰撞率不同。该测试只验证对tie的正确处理，不说明真实A数据有94%的tie问题。

数值和descriptive permutation分布见 [RESULTS.json](RESULTS.json) 的 new_CPU_implementation_validation。其中任何delta都属于SYNTHETIC_ONLY，不可抄入真实模型性能表。100次置换是给定bank的扰动，五推理seeds不是五训练seeds；本轮没有新增训练重复。真实输入不足，因此没有生成显著性/置信区间，也没有应用energy因果解释。

运行方式（仓库根目录；仅自己的进程设置）：

~~~bash
nice -n 10 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../.conda/rsjg/bin/python -B docs/joint_dependency_v2/reviews/2026-10-03_6258723_pairing_audit/audit_cpu.py self-test --manifest docs/joint_dependency_v2/reviews/2026-10-03_6258723_pairing_audit/INPUT_MANIFEST.json
~~~

单进程分scene处理，Torch CPU设1线程，未调用CUDA、未导入模型/加载checkpoint；命令只向stdout输出JSON，不自动写/覆盖历史文件。依赖：Python3.10.21、NumPy1.26.4、Torch2.8.0+cu128（只用CPU指标）、SciPy1.15.3。

## 5. 缺bank时的最小导出位置与估计成本（仅设计）

**OPEN，未授权、未实施**：若没有已有归档，另申请一次canonical A单路径导出作业，显式seeds2035–2039；不训练、不重复评估D，也不因导出重建cache。先strict核对原config/checkpoint与已有source cache，不能直接运行可能触发preprocessing的普通main.py test入口。

最小捕获位置是原A forward的prediction产生之后、其被删除之前；沿用生产运算而非重写积分：

1. 如从velocity采集，复用[velocities_to_predictions，396–407行](https://github.com/WuYanXingege/RSJG/blob/6258723915f3842ee974aea6f8744cd685d5f047/tools/jdv2_stage_b_v1_failure_mechanism_audit.py#L396)；如果已取得forward的完整map prediction，不再积分第二次。
2. 按[compute_model_metrics，1742–1777行](https://github.com/WuYanXingege/RSJG/blob/6258723915f3842ee974aea6f8744cd685d5f047/src/models/model.py#L1742)乘down_factor，并逐sample执行scene.make_world_coord_torch；GT采用相同变换。
3. 只保留future段，转置[P,T_total,N,2]→[N,P,T_future,2]，保留实际dtype，记录必要的无损存储提升。不能把trainer的缩放像素输出直接写成world metres。
4. 同次forward保存GT/masks/IDs、history graph和可用aux，以及原evaluator逐agent/scene/总指标；新manifest逐文件hash。不要为导出多做一次采样或消耗额外模型RNG。
5. exporter独立新脚本/输出目录；记录其新source commit/script digest。bank完成后先在CPU认证original指标，再进行100次重排。C仅在同bank A=C证明齐备时可接受，不能默认取代A。

**LIKELY（工程量级估算，不是本轮实测）**：

- 旧canonical 139窗、5seed共695 forwards的记录为433.487秒，约0.624秒/窗（[历史adoption计时，251–269行](https://github.com/WuYanXingege/RSJG/blob/6258723915f3842ee974aea6f8744cd685d5f047/docs/joint_dependency_v2/EXACT_PERSISTENT_REFINEMENT_PRODUCTION_ADOPTION_VALIDATION.md#L251)）。据此单A五seed导出约7–8分钟纯forward量级，另加加载、变换、I/O；新运行与SDD竞争资源时不可沿用此时长或并发启动承诺。
- 历史每seed176,640个未来轨迹值，若存FP32，Y约706,560 bytes/seed；五seed约3.53MB（3.37MiB），再加GT、graph、IDs、aux及metadata。若实际dtype/shape不同重新计量，不固定压成这个预算。
- bank就绪后置换只在CPU做，复杂度主要由collision的scene内agent pairs决定；尚无真实bank，不编造CPU完成秒数。

## 6. SDD保护、限制和决策

**CONFIRMED只读快照，18:18:24 +08:00**：pipeline PID962146、主进程962176，仍是independent GDTS goal_pretrain；活跃stdout为resume_after_loss_mask_fix.log，看到Epoch5、1076/2440 batches。只读取了进程关系与日志尾部，没有改变进程、worker、线程、配置、checkpoint或cache；该进度只是时间快照。

**OPEN**：真实pairing收益仍未知。不从合成反例推出Stage-A有效/无效，不从缺bank推出需要更大网络。仅在真实bank得到稳定效应后才考虑另一次单边项/I分解审计；本轮不同时推进该方向。geometry6仍仅是可推导几何的显式归纳偏置，缺少失败case与graph coverage证据，维持NO-GO；不新增B结构或loss。

**唯一下一步仍是补齐可认证完整bank**。本轮的blocked报告与可重复CPU验收已完整保留并按授权归档上传；是否上传成功以最终远端回读结果为准，不把本页写入本地等同于已发布。

## 7. 归档清单与不可变指纹

| 文件 | SHA256 |
|---|---|
| [RESULTS.json](RESULTS.json) | 647d66130ea1517eea0a5fda004e702dcc3601af91d30338377525e53f4a9af7 |
| [INPUT_MANIFEST.json](INPUT_MANIFEST.json) | 79092dcd11c4b5e54ad3e7bd28dcd27fe66eee07f18bf335fedf5d88e866b41c |
| [INPUT_SCHEMA.json](INPUT_SCHEMA.json) | 14316435b0fa5ef771c75258dee91860c6eaa1c2ad11053dbd2110fde4ad39a9 |
| [DISCOVERY.json](DISCOVERY.json) | 567e7ad046c311a6fdc1e0c8b70745c3e51905932b9a6f5ee9d7f7213e9b36c4 |
| [audit_cpu.py](audit_cpu.py) | 80aeae22f7d0a488779e5920a9093519ce05329d82e9361b72ae6f080e4d3225 |

没有报告自己的SHA或提前猜测归档commit。提交只包含本目录与索引新增条目，历史review/JSON保持原字节；未提交大bank、数据cache或运行日志。最终发布核验须确认远端分支可达及报告/JSON与本地字节一致。
