# Stage-A 完整轨迹配对审计：网页端只读复核

审查日期：2026-10-03（Asia/Shanghai）。固定对象：`1e4d08c96ff058e7337c782fcdb5440d9ad631cf`。本轮仅回读公开归档、源码与机器记录，复算已保存数值；没有运行模型、重新生成轨迹、使用 GPU、修改用户源码或恢复 SDD。

**判断：支持当前固定 bank 的原始列配对相对指定随机置换 null 改善 JADE/JFDE。该结论不能升级为 pair energy 因果贡献、跨序列泛化或跨进程 bitwise replay。Stage-A 机制研究可保留；geometry6、Stage-B 扩展和重训仍不放行。**

## 1. 本轮实际复核范围

完整解析固定 commit 的 REVIEW、SUMMARY、RESULTS、BANK_MANIFEST、EXECUTION、RESTART_PREFIX_COMPARISON、DETAIL_VALIDATION 及其他列入 ARTIFACT_HASHES 的 JSON。13 个列明 artifact 的 SHA256 全部匹配；没有把 hash 索引本身和 README 算作额外通过的自哈希对象。

对原始 RESULTS 的37个统计组、740组 metric/arm 的100次置换分布，重新计算 delta、均值、5/50/95% 分位。核对五种子加权归约、活动分层加权归约、SUMMARY 与原始结果的对应数值。81,190 条数值一致性断言通过，最大差异1.734723475976807e-15；这些是归档自洽性检查，不是独立模型试验或独立样本数。

核对 bank manifest 的695个唯一 seed/window键、每seed139窗、同窗口跨seed的IDs、来源指纹、GT/mask/graph数组合同。与 RESULTS 的695条明细元数据逐项对应，1840个有效agent实例以及310/150/235活动分层计数一致。

本会话无法访问用户本地轨迹及CPU明细NPZ，故未独立重算原始坐标上的ADE、碰撞或inverse-gather。真实NPZ验收、4,890项CPU parity、strict load及CUDA RNG均依据已哈希的原始执行记录；不能把它们说成本轮重新执行。

## 2. 核心结果与分层

差值定义为 independent shuffled − original，误差指标的正值表示打散后更差。

| 指标 | Original | 随机配对均值 | 差值 |
|---|---:|---:|---:|
| JADE（m） | 0.413185 | 0.447103 | +0.033918 |
| JFDE（m） | 0.701908 | 0.779330 | +0.077423 |
| RME legacy full-mask（m） | 0.297907 | 0.365417 | +0.067510 |
| minADE（m） | 0.278201 | 0.278201 | 浮点归约级别零 |
| minFDE（m） | 0.386215 | 0.386215 | 浮点归约级别零 |

JADE/JFDE分别较原始值增加约8.21%/11.03%。这回答的是保存bank内的列对应关系对joint oracle指标是否有用，不能推成真实生成分布的概率marginal保证。

| 活动分层 | 唯一窗口 | scene-seed记录 | ΔJADE（m） | ΔJFDE（m） |
|---|---:|---:|---:|---:|
| all-active | 62 | 310 | +0.075801 | +0.174382 |
| mixed | 30 | 150 | +0.000498 | −0.001663 |
| singleton | 47 | 235 | 约0 | 约0 |

按scene记录加权，all-active贡献约99.68%的JADE总差值；其JFDE贡献约100.46%，因为mixed存在小幅反向项。百分比是描述性加权分解，不是模块贡献或因果份额。N、图活动、场景难度相关，不能把该分层解释为激活edge的因果效应。

全部47个E=0窗口都是N=1。实际数据没有多人E=0负控，不能将singleton不变外推为多人E=0必然不变。

## 3. 负控与碰撞边界

原始记录报告：逐scene/replicate marginal差值恰为0，全部common/N1最大误差5.551115123125783e-17；总体marginal约1e-16的变化为归约舍入。原始bank与独立置换均无exact JADE ties，官方first-index CR-JADE与tie-set附加分数在本批一致。合成tie验收保留其覆盖价值，但不能作为真实tie率。

CRmean均值打散后增加0.605643个百分点；CR-JADE增加0.431403个百分点。然而CR-JADE的置换5%—95%区间跨零，2035/2037均值方向反向，不能宣称统一避碰改善。该区间是给定bank的置换分布分位，不是总体泛化置信区间；ECDF不解释为显著性p值。

仅一个原始sequence，窗口重叠，五个seed均为推理采样。现有资料不支持五次独立训练、跨sequence置信区间或新held-out证据。CI=null和epsilon_joint=null的处理正确。

## 4. Seed 2036 重启：影响与不能得出的解释

RESTART_PREFIX_COMPARISON显示：36条旧/新前缀中31条全部数组一致，另4条仅edge relation embedding存在约1e-7差异；window13发生候选IDs、joint goals和Y变化，Y最大差2.265572m，JADE/JFDE旧新差约−0.099445/−0.099960m。

这证明跨进程同seed重放未逐位一致；尚不能定位第一处内部operator或归因CUDA非确定性。缓存/配置来源一致不等于所有运行状态及算术逐位一致。导出停止原因及未知in-flight尾部也不能编造。

新2036整seed采用，旧36条前缀整体排除，2035只复用完整139条。这一策略避免了逐窗挑选更好结果。最终695记录来源为139复用+556新forward；两次尝试至少731个已捕获forward。SUMMARY给出的按seed来源映射与EXECUTION一致。

本轮只读敏感性复算如下，不删改canonical结果、不做新的轨迹采样：

| 指标 | 原五seed平均delta（m） | 排除2036后的四seed平均delta（m） |
|---|---:|---:|
| JADE | +0.033918 | +0.032790 |
| JFDE | +0.077423 | +0.074173 |
| RME | +0.067510 | +0.065695 |

五seed每个平均方向均为正；逐一留出任意seed后，JADE/JFDE/RME平均方向也均保持正。这说明总体正向描述不依赖2036这一个seed。它不修复bitwise replay、不证明首个分歧根因，也不是新的独立训练不确定性证据。

固定bank已经生成后，Original与shuffled使用完全相同轨迹集合；跨进程生成差异并不自动否定这个条件化于bank的对比。但后续需要重新采样的模块因果实验，必须另外认证共享输入、随机payload和无干预reference路径。

## 5. Go / no-go

| 决策对象 | 判断 | 依据与边界 |
|---|---|---|
| 保存bank内的列配对优势 | GO，作为受限实证结论保留 | 负控记录、五seed方向、重算及留出敏感性一致 |
| Stage-A机制研究主线 | GO，继续解释收益来源 | 当前结果支持研究价值，不等于方法创新已经成立 |
| pair energy不可分解项I有效 | OPEN | 尚未干预I；unary单边项、generator/shared noise等来源未隔离 |
| 跨序列的普遍增益或edge因果 | NO-GO | 单sequence；分层有N/难度混杂；mixed弱效应 |
| 统一避碰改善 | NO-GO | CR-JADE均值和置换分布存在混合证据 |
| bitwise跨进程重放合同 | 未通过 | window13候选决策和轨迹已有直接差异，根因OPEN |
| geometry6、扩容、Stage-B扩展、重训 | NO-GO | 本轮没有为这些改变提供新的归因或失败机制证据 |
| 自动恢复SDD或启动下一实验 | 不执行 | 用户明确本轮止于网页端只读复核 |

科学问题从“列配对是否有用”推进到“这项优势由什么机制产生”。后续energy分解是有价值的候选设计，但本轮不执行，也不能由此直接授权新GPU任务。

## 6. 唯一建议的下一步

先只读整理2036/window13的重启分歧证据：原/新source/cache输入、candidate IDs、Round0/tie payload、噪声与RNG记录、conditional scores及dtype有哪些已保存，哪些缺失。将“第一个已保存且不同的tensor”与“第一个内部不一致operator”分开；缺乏中间trace时明确无法定位。

只有若现有记录不足，才另行拟定单窗口共享输入的最小重放申请。本轮不重放，不先做energy-I消融，不改变数据或指标，不恢复SDD。

## 7. 固定来源与归档状态

- [审查报告](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/docs/joint_dependency_v2/reviews/2026-10-03_aa7022e_stage_a_bank_export_pairing/REVIEW.md)
- [原始机器结果](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/docs/joint_dependency_v2/reviews/2026-10-03_aa7022e_stage_a_bank_export_pairing/RESULTS.json)
- [bank manifest](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/docs/joint_dependency_v2/reviews/2026-10-03_aa7022e_stage_a_bank_export_pairing/BANK_MANIFEST.json)
- [重启前缀差异](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/docs/joint_dependency_v2/reviews/2026-10-03_aa7022e_stage_a_bank_export_pairing/RESTART_PREFIX_COMPARISON.json)
- [CPU审计归约源码](https://github.com/WuYanXingege/RSJG/blob/1e4d08c96ff058e7337c782fcdb5440d9ad631cf/tools/jdv2_audit_stage_a_bank.py#L55-L74)

本次复核新文件尚未推送GitHub：公开读取成功，但此会话的Git写入通道缺少GitHub用户名/凭据，普通push的dry-run失败。未采用force push、未修改认证或原远端内容。用户已有的归档/推送授权继续有效；本文件可由具有写权限的Codex作为新审查记录归档，不能标记成本会话已发布。

## 8. 本轮机器可读复算摘要

```json
{
  "status": "PASS",
  "scope": "Read-only checks of archived scalars, distributions, input contracts; no NPZ or model replay",
  "checks": {
    "metric_groups": 37,
    "metric_arm_distributions": 740,
    "numeric_equalities": 81190
  },
  "max_numeric_difference": 1.734723475976807e-15,
  "coverage": {
    "unique_windows": 139,
    "scene_seed_records": 695,
    "agent_instances": 1840
  },
  "activity": {
    "all-active": 310,
    "singleton": 235,
    "mixed": 150
  },
  "same_input_contracts_across_seeds": "PASS",
  "sensitivity": {
    "JADE": {
      "all_five": 0.033918119257319995,
      "excluding_2036": 0.032790141778204246,
      "minimum_leave_one_seed_out": 0.032790141778204246,
      "all_seed_means_positive": true
    },
    "JFDE": {
      "all_five": 0.07742275607423711,
      "excluding_2036": 0.07417253732621963,
      "minimum_leave_one_seed_out": 0.07417253732621963,
      "all_seed_means_positive": true
    },
    "RME_legacy_full_mask": {
      "all_five": 0.06750996044569389,
      "excluding_2036": 0.06569477911825501,
      "minimum_leave_one_seed_out": 0.06569477911825501,
      "all_seed_means_positive": true
    }
  },
  "weighted_contribution_to_total_delta": {
    "JADE": {
      "all-active": 0.9968315147718397,
      "mixed": 0.003168485228153709,
      "singleton": 5.533902828800938e-16
    },
    "JFDE": {
      "all-active": 1.004636769773564,
      "mixed": -0.004636769773555009,
      "singleton": 9.697385400525767e-16
    }
  },
  "limitations": [
    "Local trajectory and detail NPZ bytes inaccessible here; no independent trajectory-level verification.",
    "One raw sequence; no generalization confidence interval.",
    "Restart cause remains unlocalized; exclusion is sensitivity only, canonical results unchanged."
  ]
}
```

## 9. 原归档artifact指纹独立回读

| 文件 | 本轮核对SHA256 |
|---|---|
| BANK_MANIFEST.json | `ac6cd236d1f3aa1828dfe2548abe3a094ee7989dca1a7720d76771a872853ed1` |
| CPU_CHECKS.json | `d92975ae79e1218cb50892db14e71331232f1c2e87eea300aa89eac48652e440` |
| DETAIL_VALIDATION.json | `55019b86f730ce10912e9de82c3bd520cc72776d568e6bfaf20fd850020cb0af` |
| EXECUTION.json | `b09636cef7c88b922aecc2518379b86fce4af57e2facddfa92dabbb66a654398` |
| RECOVERY_PRECHECK.json | `6763d1af9652fe2b7e5909c5481ddd3fe2c3f4ed1dc4374957a13a5d63b07ec6` |
| RECOVERY_SOURCE_INCOMPLETE.json | `b117ac3a9ae375e6b3827c847ce0d7f1a7dee243b32882b1d5ddbb633b920aee` |
| RECOVERY_SOURCE_RESOLVED_ARGS.json | `1958fe68faabc06e8400080c811fa84bb79b48a3dd69875c5e7390c5f445c27b` |
| RESOLVED_ARGS.json | `df5c3d9dd4cf855e741e461884ab10c10b1f22f17a9876e2932eb6a1d03f32bb` |
| RESTART_PREFIX_COMPARISON.json | `6184685aeeafe5f7ceda8ff4485948e4ece1e895368786ee29a833d97e9bcd0c` |
| RESULTS.json | `0061d510cd0e047d7e12e5d97e729c0f63a7671c73514d6df3b3746f3e098367` |
| REVIEW.md | `2cde197f8ddd7b0f0647a499836c4bd47d24017cd5363018daa47a5cb5e0c11b` |
| SDD_STOP_RECORD.json | `74819254e60b35c4d0e2885ecaef48010349f8e7cc80c02f8d5fe8c14a9e385a` |
| SUMMARY.json | `f66774e37642e5adba3cca9ffa99bfb2a68e28b8ddeb77f70d07373e5202a4eb` |

原始轨迹bank NPZ及CPU明细NPZ未由本会话读取；以上13项均指GitHub归档文件，不能把manifest内宣称的NPZ hash说成本轮直接验证。
