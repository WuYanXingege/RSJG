# 独立 UNIV 分组：全量准备与 fresh goal preflight

结论：**EXTERNAL_EVIDENCE_BLOCKED，尚非 GOAL_TRAINING_READY；本轮未启动正式长训练。**

协议、全量 observation/targets、原生 packing、真实 CPU/CUDA FP32/BF16 小步更新、独立进程 checkpoint/replay 与运行接口已完成验收。剩余来源阻断是当前 semantic raster 的资产级生成/训练范围证据；依赖该资格的生产 full-loader 和真实 inner selection 仍被 gate 拒绝。不能用工程 smoke 或作者 ZIP 匹配替代这一资格。

源码基点：`7a0f83bb9341617b4f214cefc3563927961c08d6`，分支 `research/joint-dependency-v2-clean`。实际执行基于本轮修改，模型记录与 supervisor 留存执行时源码 SHA；最终 smoke 的源码 SHA 与提交前文件再次核对。发布 commit 是本报告所在的新提交，不把基点冒充修复后的源码。SDD、历史实验、技能和 Apps 均未改动。

## 1. 实际完成情况

| 条件 | 实证与边界 |
|---|---|
| 新协议 | `p2_grouped_univ_hotel_v1`；train2537 / inner1267 / outer445；精确迁移947窗，原地址/成员/帧/agent/cohort不变 |
| 发行绑定 | 作者 ZIP 实际下载并核对60个本地成员引用，全部匹配；H/RGB/pred_mask及raw别名均有SHA链 |
| observation | 4249份全部验收；947份重封装，3302份通过明确链复用；不机械复制旧8GiB |
| targets | train2537 + inner1267 =3804份全部生成并回读；outer0；原Gaussian/down8/peak normalization，无clamp/丢窗 |
| 全量纯输入 | 3804窗obs-only adapter通过；坐标roundtrip最大误差 `1.7053025658242404e-13`；不解释为仅靠可逆性证明camera身份 |
| 原生packing | train175 packs/11122 agent-window exposures；inner391/24955；不混source、不丢尾包，实际原函数输入准备通过 |
| CPU回归 | 最终152项通过：102项新协议/loader回归 + 50项原专用guard测试；包含合成epoch、selection/权限/状态反例 |
| 真实更新 | 两轮共16 attempts/16 successful/0 failed/0 skipped，全部train，1024 agent-window更新曝露；重复验收计入同一24次上限 |
| CPU/CUDA/AMP | 每轮CPU goal/joint各1步；CUDA FP32各2步、BF16各1步；相关参数组实际改变、loss/梯度有限 |
| 保存/replay | 两轮均独立进程回读4个CUDA case；weights/optimizer/scheduler/RNG身份通过；固定输入/noise eval loss逐位相等，最大差0 |
| fresh初态 | smoke后重新seed3101、未经更新、独立目录；前后两次fresh初始化参数hash相同，非clean parent |
| 生产入口 | 实际调用main的launch-check及独立生产loader/inner-selection接口，因真实semantic缺口拒绝，不伪造成功 |
| 后续依赖 | joint等待qualified goal；cache/shared/L1等待qualified complete base及对应缓存/初态，不自动串行开训 |

权威机器结果：[RESULTS.json](RESULTS.json)、[READINESS_MATRIX.json](READINESS_MATRIX.json)、[最终一致性审计](ARCHIVE_CONSISTENCY_AUDIT.json)。逐项状态见[矩阵说明](READINESS_MATRIX.md)。

## 2. 新协议身份与数据用途

UNIV三源保守归一inner组，ZARA三源保守归同一train组，ETH为train、HOTEL为outer。作者loader的五场景映射提供跨逻辑角色组的正面依据；这不是八份原始recording/session/ID-remap全部认证，也不把重叠验证窗口视为独立recording。

| 身份 | SHA256 |
|---|---|
| 原source rows | `a3df43f81bd1c198d369b1607bf3e02ed7104c9197a094e9d93fcf6a0254e4c1` |
| 移动947成员 | `bdcdd21cf6dd8995bea27d8881519704e37f8027a6d3b07b569cf884ce449ea9` |
| 新source rows | `5b17b449313e33402a1b19eb50abca70ece02e275cb4dc4080b65f4773f74ff4` |
| 新seed3101窗口order | `30fe498bbc50ad6b3792568b761d0a90b13474347a8c4f68b323c6ebd88a7637` |
| 最终data_binding | `73206ec37a642b6702949c898661e2d45f23c27076a935d320e21ce6c88f26a7` |

见[迁移收据](MIGRATION_RECEIPT.json)、[用途与资格合同](PROTOCOL_AND_PREPARATION_CONTRACT.md)、[准备manifest](PREPARATION_MANIFEST.json)、[实际输入审查manifest](QUALIFIED_INPUT_REVIEW_MANIFEST.json)。train窗口order用于Stage-A；基座175/391原生packing按固定source/window顺序，二者不混为同一个hash。

未来标签按 `(window_id,t,frame,agent)` 授权slot读取，不使用全source帧并集或full-pickle。完整导出扫描中有432924个window-local target slots（重叠窗口按自身身份计数）、50300次去重XY解码；outer仅做原字节hash及已有obs验证，没有outer future数值解析。原始canary是七个非outer source各第一个冻结窗口，未按loss选样。

[导出结果](ARTIFACT_EXPORT_RESULT.json) 与[最终回读](ARTIFACT_VERIFY_RESULT.json)各自索引全部catalog shards，包含8053个 observation/target条目、sidecar/hash/producer/reuse链；大型tensors与weights仅留本地独立outputs。原导出runtime版本另存，旧历史归档不覆盖。

## 3. 作者发行绑定已解决，semantic资格尚未解决

固定作者GDTS版本 `297d508558c10831983ea4b19c2b3e657459a449` 的 [download_data.sh](https://github.com/Winderting/GDTS/blob/297d508558c10831983ea4b19c2b3e657459a449/download_data.sh) 指向本次实际下载的dataset.zip。归档144912564 bytes，SHA256：

`47e9c37fe0c7496db0b6000e8ee81bf09c7b1be2d8b88c8beea057feeb0f760e`

完整链见[AUTHOR_RELEASE_BINDING](AUTHOR_RELEASE_BINDING.json)：作者固定代码→URL→archive→member→本地raw/H/RGB/raster→source/scene映射。只在内存选择性解压所需ETH/UCY成员用于hash；没有下载模型权重或原视频，没有覆盖本地数据。60是本地成员引用数，不声称有60份独立recording。

Goal-SAR说明语义图来自学习式预处理器，其实现部分描述在相应轨迹训练场景图像上训练分割；这些段落没有提供当前released raster逐资产的权重/fold/标签清单。因此不能推断它们满足新UNIV-inner分组。见 [Goal-SAR §3.2、§4.1及实现细节](https://arxiv.org/pdf/2204.11561)。GDTS也描述预训练语义预测器，但未补齐这些raster的具体producer身份，见 [GDTS §III-B](https://arxiv.org/html/2311.14922v2)。

fresh goal实际受阻资产是 ETH、UNIV、ZARA1、ZARA2 的 `pred_mask.png`；完整路径/SHA在[SEMANTIC_SOURCE_EVIDENCE](SEMANTIC_SOURCE_EVIDENCE.json)。HOTEL semantic也保留UNKNOWN，但不会单独阻断不消费它的goal/joint。后续cache消费outer时才额外要求该资格。

所需补件：每个raster SHA→分割checkpoint SHA→训练scene/label范围→生成版本/命令，以及许可和未使用禁用的holdout标签、future轨迹/统计的证据。不能把“作者分发提供了这个文件”改写为“非学习静态资产”。原recording/session/remap/converter细链仍UNKNOWN，不重新要求完整历史转换日志来否定已经通过的分发版本绑定。

若补件不可得，[具体替代方案](SEMANTIC_REMEDY_NOT_EXECUTED.md)是另行批准仅用train场景合法RGB/标注训练并冻结预处理器，生成独立版本输入；本轮未实施，不置零通道、不换RGB、不使用oracle，也不计为原主线READY。

## 4. 训练、checkpoint和网络边界

网络/loss定义没有结构性变化。原模型唯一差异是可选geometry-only dataset注入，避免旧构造器隐式读取GT semantic/未使用scene/旧cache；旧默认调用保持。Goal613772参数，GDTS7826588注册参数；额外注册的Transformer模板不当成第三个有效层。Stage-A/MC目标、Social128/M4/rank8/K21/P20/strict-no-z保持原合同，复用未受影响的既有验收，不新跑Stage-A训练或bank。

实际层/shape/source差异见[NETWORK_AND_LOSS_AUDIT](NETWORK_AND_LOSS_AUDIT.json)与[实现说明](IMPLEMENTATION_AND_NETWORK_SCOPE.md)。BF16是autocast路径，参数和报告的loss保持FP32；没有逐层dtype trace或跨precision逐位等价声明。

goal验证按有效agent-window加权masked BCE；joint按原有效agent ADE、native pixels/P20。验证随机流与训练流隔离；strict improvement、tie保留早epoch。只在完整train+inner epoch之后推进scheduler并保存原子完整状态；partial训练/selection不生成可作为父产物的epoch。正式FRESH_START_ONLY，目录必须全新。父权重validator检查实际SHA、协议/资产/order、初态、完整更新/曝光、最终run receipt、best指标/tie、optimizer/scheduler/RNG和goal-only初始化策略；旧五份违规权重、SMOKE_ONLY、未完成run都不能晋升。

初态state SHA：`ff931446488399425f54ea689aaec0409c64319bd2a0ece74cf7366e65d7c8f0`，实际文件及配置绑定见[FRESH_INITIAL](FRESH_INITIAL.json)。它是未经更新的初始化审计，不是best/last或qualified goal。

16次更新、固定短replay与通过的合成epoch测试，不证明收敛、性能增益、pair energy因果贡献或完整CUDA exact resume。真实inner模型评估没有执行，因此不报告inner指标或完整GPU epoch速度。

## 5. 资源、失败记录与交付使用

累计GPU子进程wall保守计费73.7793秒，峰值reserved1,826,619,392 bytes，远低于30分钟/10GiB；所有真实更新含两轮共16/24。全量导出约4033秒、最终verify约476秒、全packing纯准备约136秒。完整资源、各namespace/PID/start ticks、HWM与采样峰值分开见[RESOURCE_SUMMARY](RESOURCE_SUMMARY.json)及append-only[PROCESS_LEDGER](PROCESS_LEDGER.jsonl)。

首次测试143通过/6失败源于旧guard suite运行方式，修正后原50项全部重测，没有删测试；后续新增合同最终152通过。两轮真实smoke均保留不可变attempt文件。用途护栏、data-binding改进和证据覆盖限制见[FAILURE_AND_COVERAGE_HISTORY](FAILURE_AND_COVERAGE_HISTORY.md)。最早bootstrap无完整PID/RSS仪表、import/native覆盖UNKNOWN如实保留，不把历史缺口补写为零。

实际命令、FP32配置、150/250epoch上限、24/48小时pack-boundary失败预算及估计假设见[COMMANDS_AND_BUDGET](COMMANDS_AND_BUDGET.md)。没有把150/250epoch上限当成本轮计划；没有自动开训。补齐真实semantic证据后，应通过既有接口补跑production loader/inner selection和联合check，而不是再申请已经完成的targets或重复数据迁移。

GitHub发布只包含代码、测试、报告与必要索引。发布后以完整commit逐文件回读远端bytes/SHA256/Git blob；为避免自引用，远端回读receipt留在独立本地publication目录，最终聊天给固定commit链接。未完成远端回读前，不以本报告文字宣称上传验证完成。
