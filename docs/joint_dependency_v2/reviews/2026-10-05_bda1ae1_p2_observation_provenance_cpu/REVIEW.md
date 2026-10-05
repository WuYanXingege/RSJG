# P2 独立 observation 与来源审查

## 1. 结论边界

本轮是授权的 CPU 数据准备，不是训练或性能评估。观察产物完整性、输入数值合同、recording/map 资格、生产 launch readiness 分别记录在 [RESULTS.json](RESULTS.json)。有了观察文件不等于 clean base 合格，也不验证新目标收益。

实际结果：4249/4249生成并通过独立全量回读，train3484/inner320/outer445；失败/缺失/hash mismatch均0，最终CPU合同50/50通过。产物及sidecar共8,721,506,830 bytes，最大单pt为12,071,433 bytes。数据进程峰值RSS916,168,704 bytes，全量verify结束时累计wall2812.805s（约46.88min；后续归档时间另计）。所有生产launch-check仍阻断，合格clean goal/base均0。

## 2. 固定输入与隔离

固定提交/实际开始 HEAD：bda1ae147d5912e8847642db36b5822584420563；分支 research/joint-dependency-v2-clean。开始时无 tracked 改动，已有 data symlink 与 untracked outputs 保留；未重置工作树。旧归档51文件逐一SHA核验，全部4249 records读取并核对 source rows a3df43f81bd1c198d369b1607bf3e02ed7104c9197a094e9d93fcf6a0254e4c1。详见 INPUT_EVIDENCE_READS.json 与 SOURCE_SCOPE.json。

仅新增独立默认关闭工具、pure CPU支持/护栏、50项合同测试及本目录。生产网络、loss、参数/buffer、训练推理默认路径、Registry gate、旧 source/cache 均不改。没有任务相关AGENTS额外指令，没有启动子代理，没有技能/Apps/SDD操作。

## 3. 权限与数值合同

PREPARATION_GRANT.json独立授权全部角色的artifact_preparation/validation，不替代生产provenance。按window_id+前8frame+注册agents路由；只在命中本窗obs槽位时解码x/y。source完整bytes hash、全行ID解析、window-local坐标查询是三种不同权限，计数分开。

沿用float64原生pixel、world2pixel/H_inv、ETH/HOTEL swap；原NEAREST/down8、semantic6 onehot、RGB/255；原Gaussian以pixel.float()/8生成8时间通道、std=min(h,w)/64、peak normalization。不做future裁剪/统计，不clamp或丢窗。输入字段与原函数对应见 [OBSERVATION_ONLY_PIPELINE.md](OBSERVATION_ONLY_PIPELINE.md)；授权、原子保存与续做见 [OBSERVATION_PREPARATION_CONTRACT.md](OBSERVATION_PREPARATION_CONTRACT.md)。

canary按每source字典序首window固定8项，产生前保存摘要。其中1个观察槽位在raster几何范围外；原函数仍finite/peak=1，没有越权时窗查询或额外边界处理。不能写成“栅格外坐标为零”。全量对应精确计数在RESULTS与每窗catalog。

此计数严格按下采样采样中心范围[0,w−1]×[0,h−1]判断，全量1129槽位；不是原图连续像素区域外的独立证明，也不证明H错配。

## 4. 实际观察产物

完整验收要求window集合恰等于注册4249且无重复，train3484/inner320/outer445；每份原子保存、文件SHA、sidecar、自身source身份及tensor content SHA校验、FileProviders weights_only CPU回读。export先复用8个通过canary，再生成其余；独立verify再次逐一回读全部，未读取full20 pickle或真实target。

[OBSERVATION_ARTIFACT_CATALOG.json](OBSERVATION_ARTIFACT_CATALOG.json)索引每窗实际本地路径/文件与sidecar SHA/N/shape/内容SHA，17份shards全部hash绑定。[ENVIRONMENT_AND_ARTIFACT_BINDING.json](ENVIRONMENT_AND_ARTIFACT_BINDING.json)把catalog与环境、数值代码、CLI/guard代码、实际RUN记录绑定；没有事后伪称sidecar含环境字段。

本地新目录：RSJG_JDV2_clean/outputs/joint_dependency_v2/p2_observation_bda1ae1_20261005。大型tensor、原始source/static assets不上传。新候选manifest只加observation引用及preparation状态；原source/address/cohort/frame-agent顺序、20帧身份、8/12边界、train_order完全保留。UNKNOWN及parents/cache/shared_initial_state也保留。

## 5. 来源证据：有推进，未认证

新正面链：GDTS固定README归属Goal-SAR，双方download脚本Git blob一致（11494d64f6a78e1804fd5ce075413664448cab56），指向同一Dropbox dataset.zip。固定本地脚本133bytes，SHA256见AUTHOR_DISTRIBUTION_BYTE_EVIDENCE.json。作者链接在PRIMARY_SOURCE_SEARCH.json。

这只证明分发链PARTIAL，不证明本地文件是哪份archive成员、converter参数/执行、recording/camera/session/clip、frame offset、ID remap或跨角色recording独立。8个byte-distinct源逐项矩阵在RECORDING_RELEASE_BINDING.json。没有用文件名/SHA差异或既有ID碰撞数量证明独立/重叠；没有读未来坐标做排疑。没有确认split冲突，故也不虚报BLOCKED_REGISTERED_SPLIT_CONFLICT。

静态H/RGB/semantic均本地SHA绑定并CPU转换；MAP_ASSET_PROVENANCE.json与MAP_SOURCE_USAGE_AND_GAPS.json列实际代码绑定与缺口。UNIV raster/H被students001/003/uni_examples复用；原始资料是否支持该几何复用UNKNOWN。pred_mask命名既不证明learned，也不认证PROVIDED_STATIC_ASSET；producer/weights/training-label roles/protocol UNKNOWN。静态地图本身不自动等于泄漏，本轮没有新分割模型执行。

搜索8个直接相关入口、少于30分钟，annotation archive/video/weights下载0；Dropbox HEAD失败并仅一次重试、UCY timeout、CVF403均保留。旧CMU/Google/NVlabs证据不升格、不重复冒充正面本地绑定。

## 6. 测试与失败历史

最终CPU_TESTS_6为单次50/50通过；这是最终完整运行，不是将分次结果拼成一次。之前插件setup失败（0测试）、33通过/5序列化失败、后续6+6+6定向通过分别保存。修复仅是torch.save使用binary file handle，发生在真实canary前；保留RESOURCE_PLAN旧producer指纹，不回写历史。详见 [COMMANDS_AND_FAILURE_HISTORY.md](COMMANDS_AND_FAILURE_HISTORY.md)。

测试覆盖权限先于payload、source/manifest/address/hash错误、非法future token不被解析、future反事实obs逐位不变、重叠窗本地路由、ID/成员/顺序、N1/N多、geometry/swap、独立Gaussian数学参考、dtype/channels/scales、合成prepare_inputs↔observation_inputs、wrapper重hash篡改、半文件、预算中止与复用。未构造真实网络。原prepare_inputs本就分段求导，不误报跨界。真实full20 parity NOT_RUN。

## 7. CPU执行账本与预算覆盖

RUN_plan/canary/export/verify分别记录字节hash/scan、raw-row解码（每source scan内唯一，跨canary/export可重复）、window-local槽位、静态资产转换、保存/复用/回读。真实future/target/teacher、full-source pickle、模型构造/forward/backward、optimizer更新、GPU计算与训练均0。合成full20/target仅为测试夹具，不能并入真实标签计数。

实际原始字节SHA85次/31,252,026 bytes，文本扫描5,758,268 bytes/155,688 ID行；每scan唯一obs原始行解码累计47,758，路由到298,192个window-local观察槽位（37,274 agent-window exposures×8）。静态转换16 source-phase次，生成4249独立payload；真实consumer回读8506次（canary8+export4249+verify4249），合成测试回读另列。PER_SOURCE_READ_ACCOUNTING.json从固定metadata路由重建各source分项并与以上实际全局计数逐项对账，不冒充独立逐source计时仪表。

故意CUDA查询10次（Python8/native2），全部被护栏拦截；已覆盖路径转发0。Torch import之前和未列出的native入口覆盖UNKNOWN；首次pytest setup失败快照未保留，首次metadata脚本快照被本轮刷新覆盖，均不补写完整零计数。pid=2/ppid=1可来自不同sandbox PID namespace，不能误认一个进程。历史上一轮CPU子进程CUDA availability覆盖缺口不改写。

累计预算起点07:46:57 UTC固定，最大90分钟，数据最多32GiB；单writer/1计算线程，测试或metadata最多再1进程/1线程。实测数据进程峰值及wall在RESULTS；此前测试/metadata未保留RSS，故全进程合并实测峰值UNKNOWN，保守并行估计不是实测。未扩大资源、降精度或重置预算。进程/测试完整边界在TEST_AND_PROCESS_LEDGER.json。

计时/RSS仅在phase/process范围实测，没有逐source/role单独仪表；该细分值在RESULTS明确NOT_INSTRUMENTED，不按窗口比例分摊伪造。窗口/exposure/bytes/shape/raster外计数则来自逐窗catalog精确聚合。

## 8. 训练仍阻断

实际goal/joint/cache/L1 A0/A1的既有parser launch-check均来源阻断，在设备/网络/权重/provider之前退出。独立缺口包括recording/map资格，train3484+inner320用途分离target，合格fresh goal/base，新cache4249与train teacher3484，共享初态、pair预算，L2 adapter与后续授权。见 [DATA_READINESS_MATRIX.md](DATA_READINESS_MATRIX.md)。

真实target exporter CLI尚未实现；只保留合成split接口与后续逐用途streaming设计。outer targets不纳入方案。未执行的依赖及仅launch-check命令见 [NOT_EXECUTED_FRESH_GOAL_TO_JOINT.md](NOT_EXECUTED_FRESH_GOAL_TO_JOINT.md)，没有可直接开训的放行命令。

## 9. 未批准提案与历史结论

UNIV三源保守归inner的提案从固定metadata导出947个移动windowIDs：2537/1267/445，train175 agent64 packs、inner391。只是降低该跨角色疑点的提案，不证明同recording、不解决map来源；未改现行P2。准确成员/hash/预算/可比较性代价见 [SPLIT_GROUPING_PROPOSAL_NOT_APPROVED.md](SPLIT_GROUPING_PROPOSAL_NOT_APPROVED.md)。

旧五份违规checkpoint未重新加载或升格，qualified clean goal/base仍0；历史标签暴露/forward/backward、配对收益、seed2036非逐位resume、首差及因果边界全部保留。本轮不解释旧完整A/B差距，也不认证新目标收益。

唯一建议下一步：明确审批该独立UNIV分组协议提案，批准后再做metadata/schema/权限的CPU迁移验收；仍不自动开训，map及其他来源资格仍需证据。

## 10. 归档与远端验证

只提交本轮代码、测试、审查文本与metadata索引，正常非force push，旧归档/历史结果保留。EVIDENCE_MANIFEST.json绑定本轮提交前各文件SHA256（自身除外）。提交完成后按完整Git commit逐一回读每个变更文件，检查bytes/SHA256及Git blob SHA1；远端收据在本地仓库父目录，最终回复给出其结果和固定commit链接。收据不嵌入自身提交，避免循环hash。
