# JDV2 审查结果索引

用途：将每次审查的结论、依据和机器可读检查记录保留在 GitHub，便于网页端 GPT 按固定版本分析。此目录是审查归档，不是训练/评估运行入口。

## 最新审查

| 日期 | 源码基点 | 审查 | 机器证据 | 结论 |
|---|---|---|---|---|
| 2026-10-06 | `48ad36a129975e70dd73f306da1b4841cca4c9d4` | [标准 benchmark 协议更正与当前运行重新分类](2026-10-06_48ad36a_standard_benchmark_protocol_correction/REVIEW.md) | [RESULTS](2026-10-06_48ad36a_standard_benchmark_protocol_correction/RESULTS.json) | PROTOCOL_CORRECTION_ACCEPTED；当前 grouped-UNIV 训练降级为辅助实验并继续运行，不与论文公开数字直接排名；主线改为先认证并复现标准 HOTEL，再扩展 ETH/UCY 五折；本轮无停止、无新训练、无 cache/A0/A1 |
| 2026-10-06 | `9c593013eba22f5523f4e335544937e5a308495d` | [官方语义先验 fresh joint 独立配置与验收](2026-10-06_6d04a87_official_prior_fresh_joint/REVIEW.md) · [启动合同](2026-10-06_6d04a87_official_prior_fresh_joint/COMMANDS_AND_BUDGET.md) | [RESULTS](2026-10-06_6d04a87_official_prior_fresh_joint/RESULTS.json) · [READINESS](2026-10-06_6d04a87_official_prior_fresh_joint/JOINT_READINESS.json) · [启动快照](2026-10-06_6d04a87_official_prior_fresh_joint/LAUNCH_SNAPSHOT.json) | FRESH_JOINT_RUNNING；epoch96 goal-only 父权重精确加载，history/diffusion seed3101 fresh；CUDA两步三模块梯度/更新、exact reload、完整inner 391/24955通过；初态identity与175次正式更新已确认，只运行fresh joint |
| 2026-10-05 | `2fe1320bd775a10ba1e90a1a3452a9cf6aaf1b11` | [官方语义先验独立协议与 fresh goal 验收](2026-10-05_2c25c1f_official_prior_fresh_goal/REVIEW.md) · [启动合同](2026-10-05_2c25c1f_official_prior_fresh_goal/COMMANDS_AND_BUDGET.md) | [RESULTS](2026-10-05_2c25c1f_official_prior_fresh_goal/RESULTS.json) · [READINESS](2026-10-05_2c25c1f_official_prior_fresh_goal/READINESS_MATRIX.json) · [启动快照](2026-10-05_2c25c1f_official_prior_fresh_goal/LAUNCH_SNAPSHOT.json) | FRESH_GOAL_RUNNING；官方语义栅格作为冻结环境先验，分割器训练范围仍UNKNOWN；生产175/391 packs、完整inner24955、CPU/CUDA/reload/fresh初态通过；首个正式epoch完整保存，只运行fresh goal |

新增状态（2026-10-03）：用户随后明确授权停止 SDD、保留桌面并完成 canonical A 导出与 CPU 审计，结果见新增 COMPLETE 条目；SDD 尚未恢复。以下旧 BLOCKED 报告和旧轮次的资源/授权说明保留为当时事实，不代表最新执行状态。

| 日期 | 审查源码 | 报告 | 检查记录 | 核心决策 |
|---|---|---|---|---|
| 2026-10-05 | `7a0f83bb9341617b4f214cefc3563927961c08d6`（基点；本轮修改SHA另录） | [独立UNIV分组完整preflight](2026-10-05_7a0f83b_grouped_univ_complete_preflight/REVIEW.md) · [用途/资格合同](2026-10-05_7a0f83b_grouped_univ_complete_preflight/PROTOCOL_AND_PREPARATION_CONTRACT.md) | [RESULTS](2026-10-05_7a0f83b_grouped_univ_complete_preflight/RESULTS.json) · [readiness](2026-10-05_7a0f83b_grouped_univ_complete_preflight/READINESS_MATRIX.json) · [作者发行绑定](2026-10-05_7a0f83b_grouped_univ_complete_preflight/AUTHOR_RELEASE_BINDING.json) | EXTERNAL_EVIDENCE_BLOCKED；新train2537/inner1267/outer445，obs4249/targets3804全部回读、175/391pack通过；152项CPU、累计16次真实CPU/CUDA FP32/BF16更新与独立reload通过；作者ZIP成员绑定通过，但raster具体分割权重/训练范围未证；无正式长训练、无outer targets/metrics，不把smoke或初态当父权重 |
| 2026-10-05 | `bda1ae147d5912e8847642db36b5822584420563` | [P2 observation准备/来源审查](2026-10-05_bda1ae1_p2_observation_provenance_cpu/REVIEW.md) · [授权合同](2026-10-05_bda1ae1_p2_observation_provenance_cpu/OBSERVATION_PREPARATION_CONTRACT.md) | [RESULTS](2026-10-05_bda1ae1_p2_observation_provenance_cpu/RESULTS.json) · [全窗catalog](2026-10-05_bda1ae1_p2_observation_provenance_cpu/OBSERVATION_ARTIFACT_CATALOG.json) · [CPU账本](2026-10-05_bda1ae1_p2_observation_provenance_cpu/TEST_AND_PROCESS_LEDGER.json) | obs4249/4249生成并全量回读，50项最终CPU通过；recording/map仍UNKNOWN，作者分发链PARTIAL；train3484/inner320/outer445不变，base0/targets0/模型/GPU/训练0；CUDA覆盖与RSS细分缺口披露；UNIV分组仅未批准提案，不开训 |
| 2026-10-05 | `28728bf96a290ccd83c5c637769deb987f477b07` | [P2 role loader / step-cap CPU接线](2026-10-05_28728bf_p2_role_loader_step_cap_cpu/REVIEW.md) · [权限合同](2026-10-05_28728bf_p2_role_loader_step_cap_cpu/P2_SCHEMA_AND_PERMISSION_CONTRACT.md) · [bounded合同](2026-10-05_28728bf_p2_role_loader_step_cap_cpu/BOUNDED_UPDATE_AND_STOP_CONTRACT.md) | [RESULTS](2026-10-05_28728bf_p2_role_loader_step_cap_cpu/RESULTS.json) · [CPU账本](2026-10-05_28728bf_p2_role_loader_step_cap_cpu/CPU_RUN.json) | 工程CPU通过（P2 63/相关唯一157）；readiness仍BLOCKED、base0；556 base packs/epoch，L1 pre-fetch cap；真实模型/数据更新/GPU计算0；历史CPU子进程CUDA availability护栏缺口明确披露并定向修复，不宣称整轮枚举0；不开训 |
| 2026-10-05 | `4f11f44c3a490c491eb310b4fa7529806c71c1ed` | [P2基座/来源/两臂准备审计](2026-10-05_4f11f44_clean_base_split_objective_preflight/REVIEW.md) · [资源依赖](2026-10-05_4f11f44_clean_base_split_objective_preflight/RESOURCE_AND_DEPENDENCY_PLAN.md) | [RESULTS](2026-10-05_4f11f44_clean_base_split_objective_preflight/RESULTS.json) · [L1/L2预登记](2026-10-05_4f11f44_clean_base_split_objective_preflight/PILOT_PREREGISTRATION.md) | AUDIT_COMPLETE / BLOCKED_BASE_PROTOCOL_VIOLATION；合格base0、GDTS违规2、派生A违规3；P2 train3484/inner320/outer445，跨recording身份待核；30项CPU检查通过，checkpoint5次，模型/GPU/训练0；两臂TEMPLATE_BLOCKED，先clean base/新cache/loader与step-cap，不开训 |
| 2026-10-05 | `c9dbfecf64eb95504664c2901d3c2bb26f41aea5` | [CUDA/AMP与真实loss smoke](2026-10-05_c9dbfec_cuda_amp_loss_smoke/REVIEW.md) · [设备/RNG合同](2026-10-05_c9dbfec_cuda_amp_loss_smoke/DEVICE_AMP_CONTRACT.md) | [RESULTS](2026-10-05_c9dbfec_cuda_amp_loss_smoke/RESULTS.json) · [原生证据](2026-10-05_c9dbfec_cuda_amp_loss_smoke/gpu_run/RESULTS.json) | CUDA_FP32_BF16_LOSS_SMOKE_CERTIFIED_NO_TRAINING；CPU172通过，GPU16有效/8非法，真实4次loss/backward；toy更新2、真实更新0、训练0；3.334s/144MiB reserved，默认旧目标与历史保留；完整resume/性能/observer neutrality仍OPEN |
| 2026-10-05 | `dcf5fde1aa77f62a25a9209fd91e4f854a21510a` | [MC CPU接线与RNG/checkpoint验收](2026-10-05_dcf5fde_mc_wiring_rng_checkpoint/REVIEW.md) · [输入/模式](2026-10-05_dcf5fde_mc_wiring_rng_checkpoint/INPUT_AND_MODE_CONTRACT.md) · [恢复合同](2026-10-05_dcf5fde_mc_wiring_rng_checkpoint/RNG_CHECKPOINT_CONTRACT.md) | [RESULTS](2026-10-05_dcf5fde_mc_wiring_rng_checkpoint/RESULTS.json) · [最终接线测试](2026-10-05_dcf5fde_mc_wiring_rng_checkpoint/CPU_RUN_3.json) · [追加错误测试](2026-10-05_dcf5fde_mc_wiring_rng_checkpoint/CPU_RUN_4.json) | CPU_MC_WIRING_RNG_CHECKPOINT_CERTIFIED_GPU_PENDING；166项有效覆盖通过，9 vs 3+6独立进程exact；默认旧目标、Stage-A accumulation1、参数0；合成updates279（含重跑），真实模型/GPU/训练0；首轮输出截断与派生账本显式保留 |
| 2026-10-05 | `cb5160fa8783cd7c90f0f0453809bf3e84665ed6` | [Expected conditional纯loss与CPU验收](2026-10-05_cb5160f_expected_conditional_cpu/REVIEW.md) · [输入合同](2026-10-05_cb5160f_expected_conditional_cpu/INPUT_CONTRACT.md) | [RESULTS](2026-10-05_cb5160f_expected_conditional_cpu/RESULTS.json) · [最终测试](2026-10-05_cb5160f_expected_conditional_cpu/CPU_RUN_2.json) · [源码隔离](2026-10-05_cb5160f_expected_conditional_cpu/SOURCE_SCOPE.json) | CPU_LOSS_IMPLEMENTATION_CERTIFIED_NOT_INTEGRATED；92/92通过，固定S4均值/梯度期望与精确参考一致于预登记容差；新增参数0、旧23定义不变、生产调用者0；无接线/模型调用/GPU/训练/SDD干预 |
| 2026-10-04 | `79ea3fa5e94afe6818f8c8299b89aba1d677589e` | [创新、理论与网络架构设计审查](2026-10-04_79ea3fa_innovation_theory_architecture/REVIEW.md) · [逐层架构](2026-10-04_79ea3fa_innovation_theory_architecture/CURRENT_AND_PROPOSED_ARCHITECTURE.md) | [RESULTS](2026-10-04_79ea3fa_innovation_theory_architecture/RESULTS.json) · [伪代码/配置](2026-10-04_79ea3fa_innovation_theory_architecture/IMPLEMENTATION_PSEUDOCODE.md) · [预注册](2026-10-04_79ea3fa_innovation_theory_architecture/EXPERIMENT_PREREGISTRATION.md) | DESIGN_COMPLETE_EVIDENCE_PENDING；主线A为0新增参数的MC conditional目标，备选C复用V4；B geometry6暂NO_GO；新颖性/收益未验证；仅源码、checkpoint元数据与CPU标量检查，GPU/模型调用0，SDD未恢复 |
| 2026-10-04 | `066a8c007c887f913805ae6d3ce719fac1ae0acb` | [h→sampler 预选扰动传播诊断](2026-10-04_066a8c0_h_to_sampler/REVIEW.md) | [RESULTS](2026-10-04_066a8c0_h_to_sampler/RESULTS.json) · [逐轮score复核](2026-10-04_066a8c0_h_to_sampler/POST_CHECKS.json) · [执行账本](2026-10-04_066a8c0_h_to_sampler/EXECUTION.json) | COMPLETE；2进程8片段，A/B各4次；逐轮/最终IDs全部一致，实际Gumbel匹配P2；固定h的下游score仍微变；额外CPU solver 0；只支持本预选h对未观察到ID改变，neutrality未认证、历史因果OPEN |
| 2026-10-04 | `2156530d2d7bc15e4b16c53d7e1eb4120ad2dd38` | [Message layer 固定输入诊断](2026-10-04_2156530_message_layer_local/REVIEW.md) | [RESULTS](2026-10-04_2156530_message_layer_local/RESULTS.json) · [执行账本](2026-10-04_2156530_message_layer_local/EXECUTION.json) · [输入合同](2026-10-04_2156530_message_layer_local/INPUT_MANIFEST.json) | COMPLETE；4进程8次原message；6次无observer输出均不同；有限trace首差为两次feature累加后的raw aggregated；门槛通过后2次独立source index_add_复现173项差异；neutrality未认证、历史因果OPEN；完整A/encoder/GRU均0 |
| 2026-10-04 | `0ffdf7c35624e2ecc9e446bc0ee0cd9f3ac1e25a` | [Social encoder固定输入局部诊断](2026-10-04_0ffdf7c_social_encoder_local/REVIEW.md) | [RESULTS](2026-10-04_0ffdf7c_social_encoder_local/RESULTS.json) · [SUMMARY](2026-10-04_0ffdf7c_social_encoder_local/SUMMARY.json) · [输入合同](2026-10-04_0ffdf7c_social_encoder_local/INPUT_MANIFEST.json) | COMPLETE；4进程16次encoder，完整A为0；14次无observer输出均不同；有限trace中GRU及message输入相同、首差为message_layer_0输出；仅局部模块边界，canonical neutrality未认证、历史根因OPEN |
| 2026-10-04 | `80ee0e2c1762b019a43e7297e12e2fd880a8ed03` | [有界重启trace诊断](2026-10-04_80ee0e2_restart_trace/REVIEW.md) | [RESULTS](2026-10-04_80ee0e2_restart_trace/RESULTS.json) · [首差摘要](2026-10-04_80ee0e2_restart_trace/FIRST_DIVERGENCE.json) · [TRACE_MANIFEST](2026-10-04_80ee0e2_restart_trace/TRACE_MANIFEST.json) | COMPLETE；4进程56窗；无trace复现29处ID差/Y最大2.265572m；traced pair首差线索为SocialMotionEncoder输出但IDs/Y一致；neutrality未认证，operator/历史根因OPEN；无额外region、无生产修复、SDD未恢复 |
| 2026-10-04 | `1e4d08c96ff058e7337c782fcdb5440d9ad631cf` | [网页复核与重启证据盘点](2026-10-04_1e4d08c_restart_evidence/REVIEW.md) · [网页原文](2026-10-04_1e4d08c_restart_evidence/WEB_REVIEW_ORIGINAL.md) | [EVIDENCE](2026-10-04_1e4d08c_restart_evidence/EVIDENCE.json) · [只读脚本](2026-10-04_1e4d08c_restart_evidence/inventory_cpu.py) | 只读盘点完成；72个既有NPZ核验，window9为最早保存差异；Round0/逐轮score/实际随机payload缺失，内部首差与根因仍OPEN；无重放、无GPU、SDD未恢复 |
| 2026-10-03 | `aa7022ef8acd8fa500b4959588765a21e9daa004`（2035复用`e4c3603`） | [真实 Stage-A bank 配对审计](2026-10-03_aa7022e_stage_a_bank_export_pairing/REVIEW.md) | [SUMMARY](2026-10-03_aa7022e_stage_a_bank_export_pairing/SUMMARY.json) · [RESULTS](2026-10-03_aa7022e_stage_a_bank_export_pairing/RESULTS.json) · [BANK_MANIFEST](2026-10-03_aa7022e_stage_a_bank_export_pairing/BANK_MANIFEST.json) | COMPLETE；139窗×5 inference seeds；parity/负控通过；独立重配ΔJADE=+0.033918m、ΔJFDE=+0.077423m，优势集中all-active；重启非逐位一致另列 |
| 2026-10-03 | `02d3fa2876e7a2dd663b9a43df934400cf0c1c99` | [Stage-A bank 导出准备与配对审计](2026-10-03_02d3fa2_stage_a_bank_export_pairing/REVIEW.md) | [RESULTS](2026-10-03_02d3fa2_stage_a_bank_export_pairing/RESULTS.json) · [BANK_MANIFEST](2026-10-03_02d3fa2_stage_a_bank_export_pairing/BANK_MANIFEST.json) | BLOCKED_RESOURCE_UNAVAILABLE；独立工具与7类CPU集成验收通过，28项保护测试通过；SDD占用唯一GPU，真实bank未导出 |
| 2026-10-03 | `6258723915f3842ee974aea6f8744cd685d5f047` | [固定完整轨迹配对审计](2026-10-03_6258723_pairing_audit/REVIEW.md) | [RESULTS](2026-10-03_6258723_pairing_audit/RESULTS.json) · [INPUT_MANIFEST](2026-10-03_6258723_pairing_audit/INPUT_MANIFEST.json) | BLOCKED_MISSING_TRAJECTORY_BANK；7类新CPU合成验收通过，真实A审计0 scenes；唯一下一步补齐认证bank |
| 2026-10-03 | `1f07e7a5b81374033377c5c057d5c6ab3673b02a` | [优化必要性与科学贡献审查](2026-10-03_1f07e7a/REVIEW.md) | [CPU 与 artifact 复核 JSON](2026-10-03_1f07e7a/checks.json) | 不改结构、不重训；优先固定完整 trajectory bank 的 marginal-preserving 配对破坏审计 |

审查时未定位到合格的 post-fix Stage-A 完整 trajectory bank。已有旧 correction-on V2-A 导出不能替代；如无外部归档，导出需要另行授权，不能把本次上传视为训练或 GPU 评估授权。

## 给网页端 GPT 的阅读顺序

1. 先读最新报告第1、2、10节，确认源码版本、A/B/C/D 路径与执行边界，再读第3–9节。
2. 结合检查 JSON 区分“历史 GPU artifact”“本轮 CPU 合成检查”和“仅拟定实验”。检查记录不等于本轮重新执行真实模型。
3. A=B=C 中的 C 使用 V2-A checkpoint，但 **correction arithmetic 关闭**；D 才是实际 correction-on V2-A。
4. 追问建议必须符合冻结范围：本轮只审查，不改模型、不重训；所有未来实验另行授权。保持历史结果，不用新口径覆盖旧结果。

## 支撑证据与既有审查

以下链接固定到被审查的源码提交，不随分支移动：

- [Stage-A identity 修复与 post-fix 配对评估](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/docs/joint_dependency_v2/JDV2_CANONICAL_STAGE_A_IDENTITY_FIX_VALIDATION.md)
- [A/B/C canonical route 认证](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/docs/joint_dependency_v2/JDV2_CANONICAL_STAGE_A_ROUTE_CERTIFICATION.md)
- [原始独立审查 cfa3dd2e](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/docs/joint_dependency_v2/RSJG_JDV2_Audit_cfa3dd2e.md)
- [BF16 认证 JSON](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/canonical_stage_a_route_certification/postfix_bf16_results.json)
- [FP32 认证 JSON](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/canonical_stage_a_route_certification/postfix_fp32_results.json)
- [显式 2035–2039 paired C/D JSON](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/canonical_stage_a_route_certification/postfix_paired_five_seed_results.json)

## 后续每次审查的归档与上传约定

依据用户 2026-10-03 的指示，后续审查完成后，默认按以下方式归档并上传；若该次任务另行明确要求“仅本地、不提交、不推送”等，以该次要求为准。

- 每次新建 `日期_被审查源码短SHA/`，同日同源码多次审查增加序号，不覆盖历史报告。
- 至少保存完整报告与已有的关键检查 JSON；记录实际审查 commit、实验 source commit、checkpoint/config/protocol 指纹和执行范围。没有进行的实验不补造结果。
- 使用 GitHub 固定源码链接；本地路径只作 provenance 注明，不当作网页端可访问证据。
- 上传前检查敏感信息及链接；只暂存本次审查文档与明确的支撑记录，不提交模型、配置、checkpoint、数据/cache、运行日志或无关用户改动。
- 核对远端分支头，仅进行普通非 force push；出现远端分歧或需上传无关提交时，先处理边界，不覆盖他人工作。
- 上传后核对远端 commit，返回本次报告的固定 commit URL 及此索引入口。上传不授权启动训练/GPU 评估，也不授权干预活跃 SDD。

## 2026-10-03 归档指纹

发布版报告只在本地原稿基础上增加归档说明并修正网页链接；审查结论与数值未改。JSON 原样保留。

| 对象 | SHA256 |
|---|---|
| 本地原稿 `RSJG_Optimization_Review_1f07e7a_20261003.md` | `14e6590b289a9c71632f28a9697b53abff808ab71c649bf9181b2fbc4b678f96` |
| GitHub 发布版 `2026-10-03_1f07e7a/REVIEW.md` | `046e17ac45676c2a6e9f2510a69416a4e37a92437763792bdfa4b1fb39916201` |
| `2026-10-03_1f07e7a/checks.json`（与本地原记录相同） | `fc255a4d5d72aaf79256f617931ffee2d6e5e0a838fdca75979c471029a8bb91` |

检查 JSON 的 `script_path` 是当时的本地临时执行位置，脚本未在本次上传中归档；公式、范围、结果及脚本 SHA 已保留在报告中。
