# JDV2 审查结果索引

用途：将每次审查的结论、依据和机器可读检查记录保留在 GitHub，便于网页端 GPT 按固定版本分析。此目录是审查归档，不是训练/评估运行入口。

## 最新审查

新增状态（2026-10-03）：用户随后明确授权停止 SDD、保留桌面并完成 canonical A 导出与 CPU 审计，结果见新增 COMPLETE 条目；SDD 尚未恢复。以下旧 BLOCKED 报告和旧轮次的资源/授权说明保留为当时事实，不代表最新执行状态。

| 日期 | 审查源码 | 报告 | 检查记录 | 核心决策 |
|---|---|---|---|---|
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
