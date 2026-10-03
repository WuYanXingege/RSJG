# RSJG / JDV2 优化必要性与科学贡献审查

> GitHub 归档说明（2026-10-03）：用户在审查完成后另行授权将结果上传。本文件保留审查阶段的结论与执行记录；正文中的“未 commit/push”“仓库外层”描述的是当时的审查阶段，不包括后续这次文档归档。发布副本只新增此说明、修正网页链接与检查记录链接；源码、配置、训练和历史结果未改。`/tmp/.../checks.py` 是本地执行来源，不是可供网页端访问的归档地址。后续审查入口见[审查索引](../README.md)。


审查日期：2026-10-03（Asia/Shanghai）。审查对象：`RSJG_JDV2_clean`，分支 `research/joint-dependency-v2-clean`，实际 HEAD **`1f07e7a5b81374033377c5c057d5c6ab3673b02a`**。身份修复源码为 `d1a1382ae324b9096ac3a6d8de812ffbbb8d9780`。本报告是新增更正与决策记录，不替换任何历史报告或结果。

范围：只读源码、配置、既有 JSON/日志与 primary papers；在独立 `/tmp` 目录进行小规模 CPU 数学检查。未实施补丁、未执行模型/训练/GPU 评估/预处理、未改变 SDD 进程、配置、checkpoint、cache 或线程设置；未 commit/push。工作树无已跟踪修改；已有未跟踪数据和运行目录保留。未发现适用的 `AGENTS.md`。

标记约定：**CONFIRMED**＝源码、artifact 或本轮明确执行的检查直接支持；**LIKELY**＝有支持但未直接建立因果；**OPEN**＝需要实际实验/额外证据；**FALSE ALARM**＝该具体指控不成立或已经关闭。建议与 go/no-go 是审查决策，不等于实施授权。除外链论文外，下文所有源码行号均指上述固定 HEAD，路径相对于 `RSJG_JDV2_clean`；历史结果与本轮合成检查严格分开。

## 1. 一页决策

**现在不改结构、不重训、不扩展 Stage-B。唯一最高科学优先级：固定完整 trajectory bank，只破坏同一 scene 内每个 agent 的 sample 配对，测量原 joint correspondence 的真实贡献。**

| 类别与优先级 | 问题 / 判断 | 范围与预期价值 | 决策 |
|---|---|---|---|
| 必要工程 P0 | 普通 test 重复 seed、resume 状态不全、部分保存边界不一致、checkpoint 非原子、普通 cache 消费者可能删除共享缓存：CONFIRMED | 可信度、恢复与数据安全；不构成理论创新，不需重训 | 必须规划修复；本轮仅列最小补丁与验收，不触碰运行中 SDD |
| 科学 P1-1 | 完整 trajectory pairing 的增益：OPEN | 只改变多人对应关系，严格保持每人有限轨迹集合；已有 bank 时只需 CPU | 唯一下一动作；先认证 bank 的来源与完整性 |
| 科学 P1-2 | effective pair cost 的不可分解项是否有效：OPEN | 排除 coverage、单人 unary 及 pair 单边 ranking 的解释 | 在 P1-1 后做 frozen-checkpoint counterfactual；不先训练 |
| 科学 P1-3 | candidate-conditioned relation 的必要性：OPEN | 实际 inference flag 有效，但其收益未被以上两个问题隔离 | 条件性保留第三项；不与前两项同时开展 |
| 条件结构 P2 | geometry4→6：+64 weights，信息/收益充分性 OPEN | 更直接的直线途中接近归纳偏置，不是自动新增信息 | 当前 **NO-GO**；没有系统失败案例和可判别证据，不立项重训 |
| 暂不做 | 更大 M/rank/hidden/K/P、更多 sampler rounds、新 loss/attention、扩展 B | 未见 capacity 不足证据；会增加归因困难与计算量 | NO-GO |

**CONFIRMED：identity 已关闭。** A=B=C 不等于实际 correction-on V2-A（D）与 Stage-A 相同。post-fix C/D 是 mixed trade-off，不支持把 B 升为主要收益来源。P02 原先的数值后果已有直接定位；小幅 metric 变化既不削弱身份合同修复必要性，也不能解释全部旧 A/B 差距。

**OPEN：执行输入缺口。** 本轮在两个项目的 output/outputs 与相关 manifest 中未定位到 post-fix A 的完整 bank。现有旧 correction-on D/JMM 导出不是替代品。下一动作先定位可认证 A bank；若外部归档也没有，再申请一次按显式 2035–2039 导出的评估作业。此导出没有在本轮执行。

## 2. post-fix 证据、路径与配置→执行矩阵

### 2.1 本轮实际核验与历史认证的区别

按 prompt 顺序读取了指定的 15 份文档（14 个条目，其中 SDD 条目包含两份），包括 identity、route、原审查、architecture、A/B freeze、B V1 failure、exact adoption、A failure/no-z、SDD 与 V4 三份报告。三个 post-fix JSON 均完整解析，SHA256、source commit、invariant、parity 栏位核对通过；五种子文件的 **63 组均值/总体标准差及 paired delta** 从逐 seed 数值重新计算，一致至 `1e-14`。

以下 JSON 位于 `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/canonical_stage_a_route_certification/`：

| 文件 | 本轮 SHA256 | JSON 中 source_commit / status |
|---|---|---|
| `postfix_bf16_results.json` | `3df9d0b8592c8fc448594b05cf7dee2ee534de66e8b4bf9b27ecebeda4625bb9` | `d1a1382…` / `CANONICAL_STAGE_A_LITERAL_IDENTITY_CONFIRMED` |
| `postfix_fp32_results.json` | `8796e60061e5c913763e0f3aedbbab94cbb992240eadc99f59337d2d0c1f0aec` | 同上 |
| `postfix_paired_five_seed_results.json` | `4c1103cfd2bd923c93368d881e16de83c308cd9693087a25b1f5cc85014c39f6` | `d1a1382…` / `COMPLETE` |

**CONFIRMED（已存 artifact，不是本轮 CUDA 复跑）**：CUDA/BF16 与 CUDA/FP32 的 A/B/C 各比较均覆盖 139 windows、176,640 个轨迹值，different-elements/max-abs-diff 均为 0；分层为 47 E=0、30 mixed、62 all-active。candidate IDs、goals、contexts、shared relation、RNG、20 个 unique IDs 等 invariant 均通过。checkpoint reload 与零初始化训练梯度的历史验收见 identity fix validation；本轮未重跑 pytest/CUDA，历史“504 tests”不能写成本轮执行数。

checkpoint 指纹从这些认证记录核对，本轮没有重读大型权重重新哈希：

- A，strict-no-z epoch13：`699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`。
- V2-A epoch5：`e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73`。
- 记录中 A→V2-A 改变的 22 个 tensor 全属于 corrector，非 corrector 改变数为 0；A 输出层 weight/bias 严格全零。

证据：[identity fix validation](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/docs/joint_dependency_v2/JDV2_CANONICAL_STAGE_A_IDENTITY_FIX_VALIDATION.md)、[route certification](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/docs/joint_dependency_v2/JDV2_CANONICAL_STAGE_A_ROUTE_CERTIFICATION.md) 与上述三份 JSON。

### 2.2 四条路径及实际控制项

| 路径 / 配置 | 实际执行语义 | 能证明什么 |
|---|---|---|
| A：frozen A，canonical inference | strict-no-z、K21/P20、M4/rank8、两轮 exact；corrector 最后一层严格全零时走 literal base arithmetic | 修复后的 Stage-A 身份路径 |
| B：同一 A checkpoint，correction off | 明确禁用 correction arithmetic | A 的直接 base 对照 |
| C：V2-A checkpoint，correction off | 使用 B 阶段 checkpoint 中未改变的基底，不执行 correction | **仅 Stage-B base reference**；不是实际 V2-A |
| D：V2-A checkpoint，correction on | 非零 corrector，六个 active timestep，component zero-mean projection | 实际 V2-A 效果 |
| Stage-B 零初始化训练 | 即便输出层为零，仍保留 differentiable corrector 路径 | 不得把 A inference bypass 外推到训练 |
| `energy_weight=0` / `pair_energy_normalization=mean` | JDV2 sampler 未读取这两个 legacy 参数 | **FALSE ALARM**：不能据此声称完成 energy-off/degree normalization |
| `use_joint_energy=False` | 跳过整个两轮 refinement，返回 weighted-Gumbel Round0 | 有效，但不是“保留 solver 的纯 pair-cost 置零” |
| `use_dynamic_relation=False`，frozen inference | selected/full relation 回到 history-only base logits | 可做依赖性消融；训练中还影响 teacher posterior 路由，不能混称同一干预 |
| 普通 `test()` 配置 `num_test_runs:5, seed:2035` | active structured/exact JDV2 每次 fallback 到同一个 args.seed | 不等于显式 2035–2039；本次 post-fix 专用审计不受此问题影响 |

**CONFIRMED 源码**：`model.py` 构造 367–375 / 408–448，传参 910–911，Stage-A bypass 3781–3805，Stage-B loss 3230–3322；`joint_sampler.py` 446–487；`dynamic_relation.py` 186–191；`trainer.py::test` 899–912、`_evaluate_epoch` 1581–1592。配置：`configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml` 15–58、73–75；`jdv2_stage_b_v2a_eth.yaml` 32、85–87。

Stage-A bypass 的必要条件是正确 stage 的推理状态与最后一层 weight/bias **严格全零**；不依赖 projection，不用近零阈值，不改变非零 corrector。旧首个 dtype 分歧是 BF16 epsilon 与 FP32 零残差相加导致 promotion；首个数值分歧发生在后续 DDIM 算术，历史 trace max-abs 为约 `0.00044845`。这是已确认且已修复的合同问题，不列为待修复项。

### 2.3 post-fix 显式 2035–2039 C/D

| 指标（越小越好） | C reference off | D actual correction on | 相对 C |
|---|---:|---:|---:|
| minADE | 0.2781666942 | 0.2799707898 | +0.649% |
| minFDE | 0.3862256102 | 0.3872171843 | +0.257% |
| JADE | 0.4133276605 | 0.4149582141 | +0.394% |
| JFDE | 0.7020514868 | 0.6981691292 | −0.553% |
| Relative Motion | 0.2978071507 | 0.2974412465 | −0.123% |
| Goal Endpoint | 0.6899332012 | 相同 | 0% |
| Compatibility | 0.4875126556 | 相同 | 0% |

**CONFIRMED**：这些是相同 batch/goals/contexts/relation/noise 下开启 correction 的配对结果；每 seed 的 active-agent 139,680 个轨迹值全部发生变化，E=0 / inactive 保持 exact。五个 seed 是推理重复，不是五次独立训练；ETH valid/test 镜像也不支持严格 held-out generalization。**OPEN**：小幅改善是否跨 training seed、独立 sequence 与新 selection/test 稳定；当前不能宣称显著收益。

## 3. “joint”贡献分解与可证伪审计

### 3.1 不应混为一体的九个来源

| 来源 | 当前可确认边界 | 尚待回答 |
|---|---|---|
| Frozen candidate quality/headroom | 候选生成器冻结不等于候选足够好 | 每 agent 可达 oracle、crowded/degree strata 的上限与失败类型 |
| K21/P20 without replacement | 保证候选 ID 覆盖约束；不保证概率 marginal 或完整 trajectory multiset 保持 | 旧增益有多少仅来自覆盖保护 |
| Learned unary | history/social + candidate 单人 ranking；E=0 仍存在 | 与 frozen heatmap prior 的独立贡献 |
| Pair cost 的单边项 | 可吸收入 unary，仍可能产生很大 ranking 效应 | 不能把全部 pair-energy 收益命名为 interaction |
| 不可分解交互 I | 具有不能由两个单边函数相加表达的候选依赖 | 是否改善实际 joint worlds 且不伤害 marginal |
| Dynamic candidate conditioning | effective mixture 随候选改变 | frozen inference 依赖与从零训练必要性不同 |
| 多人 sample correspondence | 配对可改变 joint oracle，保持每人完整集合 | 最高优先级的固定-bank审计 |
| GDTS diffusion/tree | goal ID 的选择与 branch noise/context 共同决定完整轨迹 | 改 goal 再 diffuse 不保持 trajectory marginal |
| Corrector | D/C 已隔离，mixed trade-off | 有益/有害分支、oracle exposure 与投影的剩余机制 |

这些来源的存在为 **CONFIRMED**，在当前 frozen A 上的独立 task 贡献除 C/D 外均不能由此表自动确认。E=0 时 Social 仍执行 update+LayerNorm，learned unary 仍工作，因此 E=0 A 不是 independent GDTS 的严格复本（`social_encoder.py` 44–77）。

### 3.2 符号更正、gauge 分解与正确干预

以源码为准，定义低 rank 分量与 effective cost：

\[
E_{ij}^{m}(k,l)=-\langle L_{ij}^{m}(k),R_{ij}^{m}(l)\rangle/\sqrt8,
\qquad \bar E_{ij}(k,l)=-\log\sum_m p_{ij}^{m}(k,l)e^{-E_{ij}^{m}(k,l)}.
\]

\[
s_{i,p}(k)=u_i(k)-\sum_{j\in\mathcal N(i)}\bar E_{ij}(k,k^{old}_{j,p}).
\]

**CONFIRMED**：sampler 对 source/destination incident costs 求和后从 unary 减去；没有额外生效的 legacy energy weight 或 degree mean。出处：`joint_energy.py` 118–152，`joint_sampler.py` 462–477。这是对原审查中混用 energy/compatibility 符号的新增更正，原文件保持字节不变。单个关系 factor rank≤8；经 nonlinear mixture 与 candidate-conditioned relation 后的 effective matrix 不保证 rank≤8。

仅对有限的合法 Cartesian support \(K_i\times K_j\) 作均匀平均：

\[
g_e=\mathbb E_{k,l}\bar E_e(k,l),\quad
a_e(k)=\mathbb E_l\bar E_e(k,l)-g_e,\quad
b_e(l)=\mathbb E_k\bar E_e(k,l)-g_e,
\]

\[
I_e(k,l)=\bar E_e(k,l)-g_e-a_e(k)-b_e(l),\quad
\mathbb E_k I_e=\mathbb E_l I_e=0.
\]

对每条 canonical edge 仅计一次，令

\[
u'_i(k)=u_i(k)-\sum_{e:i=src(e)}a_e(k)-\sum_{e:i=dst(e)}b_e(k).
\]

则

\[
S=\sum_i u_i(k_i)-\sum_e\bar E_e(k_i,k_j)
 =\sum_i u'_i(k_i)-\sum_e I_e(k_i,k_j)-\sum_e g_e.
\]

**CONFIRMED（代数）**：完整 gauge 补偿只改变不依赖 assignment 的常数。固定 neighbors 的每个 slot 也只增加与该 agent 候选无关的常数，数学上保持 injective assignment 目标。**OPEN（有限精度）**：重新计算 full cost、做减法与 reduction 会改变舍入/near-tie，不能直接许诺 FP32/BF16 tensor-exact 或相同 `J→stay→geom→R` 决策。

只移除 I 的正确对照是 `original u + cost(g+a+b)`，或 `compensated u' + cost(0)`；保留两轮 exact、原 Round0 IDs、tie payload、graph、候选、noise。不能 double-center 后不补 unary，也不能用跳过两轮的 `use_joint_energy=False` 冒充该对照。建议未来独立 audit wrapper 在 `joint_sampler.py` 462–477 的 selected cost / conditional score 接口干预，先 strict load 原 checkpoint，禁止修改原 hash 或放宽加载。

诊断输出应包含 support 上的 \(g,a,b,I\) 范数、I 奇异值、单边/交互消融的 assignment churn 与轨迹指标，按 N、E、degree 分层。范数占比不是 task 贡献比例。若做固定 heatmap prior 加权分解，另立口径且保持 prior/temperature 不变；本轮首选 uniform valid support。该二因素分解是已知 gauge 分析，不是新理论贡献。

### 3.3 固定完整 trajectory bank 的审计合同

输入是每个 scene、每个 inference seed 单独的 `Y[N,P=20,T=12,2]`，已完成 diffusion、积分与统一坐标变换。干预位于坐标变换并冻结 artifact 之后、metric 之前。`trainer.py` 1657–1659 的 forward 输出仍为 `[P,T_total,N,2]` 缩放像素坐标；如从这里截取，必须按 `model.py` 1742–1777 的现有逻辑乘 down_factor、转换世界坐标，截取未来12步并转置为 `[N,P,12,2]`，再冻结 bank 与执行 gather。专用导出工具也须核对同一变换，绝不在 goal allocation/diffusion 之前重排。

```python
# 设计伪代码；本轮没有对真实模型执行
# local RNG: stable source/window/seed/replicate/agent identity，独立于训练 RNG
pi = stack([local_rng(i).permutation(P) for i in scene_agents])
Y_null = Y.gather(1, pi[:, :, None, None].expand_as(Y))
assert each_row_is_bijection(pi)
assert exact_equal(inverse_gather(Y_null, pi), Y)
# 用 Y_null 的新 common columns 重新计算 joint metrics
```

**CONFIRMED（有限集合不变量）**：每人完整 empirical trajectory multiset、逐 agent minADE/minFDE 保持；这不证明真实生成分布/heatmap probability marginal 不变。当前等权 P20 适用；若另有非均匀 world weights，不能直接将此结论外推为加权 marginal 保持。独立随机 π 在单列的期望上给出各经验 marginal 的乘积，但同一 bank 的各列因无放回而相关，不是 IID joint posterior samples。

预注册 100 个 local permutation replicates；2035–2039 各保持 P20，不能拼成 P100。固定 bank bytes、GT、mask、graph、scene/agent IDs 与协议。若报告 goal/auxiliary 指标，相同 P 轴 gather 必须同步作用到 selected goals/IDs；K21 candidate axis 不能当 P20 slot axis。没有 auxiliary 就不报 goal 指标。

必须同时验证：

- inverse gather exact；逐 agent minADE/minFDE exact，而非只检查总体均值。
- scene 内共同 π 的 JADE/JFDE、CRmean、RME 在事先声明的浮点容差内不变；单有效 agent 不变。
- N>1/E=0 **不要求** joint 指标不变；无 edge 不等于无 sample correspondence。ETH 历史 47 个 E=0 窗口恰为 N1，不能替代一般 E0 多人测试。
- 独立 π 后重新选 joint-best index，重新计算 RME/CR，不沿用原 index。
- 按 N1、多人 E0、mixed、all-active、degree/component 汇总；以 `shuffled−original` 表示变化，误差类指标为正才支持原 pairing 优于 null。

**CONFIRMED 新负控边界：官方 CR-JADE 的 exact-tie 例外。** `jmm_protocol.py` 565–571 用 `np.argmin` 选第一个 JADE 最优 slot。如果多个 world 的 JADE 完全相同而碰撞率不同，共同 π 会改变官方 CR-JADE；这不代表 bank 改变。本轮 CPU 反例：两人 GT 恒定在 (−1,0)/(1,0)，一个 world 两人均在 (0,0)，其余 19 个 world 在 (−2,0)/(2,0)，全部 JADE=1；共同反转后官方 first-index CR 从 1 变 0。

处理：保留官方 first-index 分数；另报 exact argmin tie-set 的 CR 均值/范围，作为审计附加口径。共同 π 可携带 original-world-ID 做稳定 tie break，但独立拼接后的 world 不再有单个原始 world-ID；跨所有 arms 的统一附加口径以 tie-set 为宜。不能把附加分数冒充未修改官方指标。无 tie 时官方 CR-JADE 负控应通过。

统计：先在 scene 内总结 permutation/seed 重复，再按独立 source sequence 或非重叠时间 block 做 paired uncertainty。重叠 windows×5×100 不是独立样本量；ETH 单 source 的 block uncertainty 不是跨数据集泛化。实际效应阈值 `ε_joint` 与未来允许的 marginal 损失 `δ_marginal` 必须在真实结果前规定；本实验的 marginal 约束为严格零。该审计若有收益，只证明现有 bank 的配对优于指定 null，不能单独归因于能量 I；若无收益，也不能否定 bank 生成阶段的所有 interaction。

### 3.4 sampler / teacher / B 的已知边界

**CONFIRMED**：exact 仅对固定 neighbors 的单 agent P×K injective assignment 精确求解 `J→stay→geom→R`，不是全 scene MAP、posterior sampling 或收敛保证。历史 exact-adoption 报告 229–269 行：交互记录 R0→R1 churn 85.19%，R1→R2 78.82%，two-cycle 40.66%；E>0 goal oracle R1=0.350775、R2=0.350778。循环的性能损害因果仍 OPEN，不据此加轮数。q 与 p 均接受 KL 梯度，q 还参与 PLpost，不是 fixed-teacher 单向蒸馏；M4 没有物理标签，不能擅自命名为 yield/follow 等。

**CONFIRMED**：B 训练按每 scene 的 GT endpoint MSE 选择一个共同 oracle branch（`model.py` 105–155、3230–3253），noisy state 来自 GT velocity 加噪（3273–3274），推理使用 reverse diffusion 当前 state（3860–3878），active t 为 30/25/20/15/10/5。**OPEN**：oracle/non-oracle exposure、noise-state、t、degree/component 是可测机制，mixed gain 本身不能证明哪一项致因。

历史 V1 failure audit 的 common-mode 去除回收约 56.85% ADE / 71.59% FDE 退化，属于 V1 因果证据，不自动是 V2-A 的新增收益；其 endpoint/ADE oracle 一致率约 55.11%，不能将全部问题归于 branch mismatch。V2-A 记录中的 relative weighted-gradient 约 1.775%、cosine 约 0.824，也不足以立即新增 loss。当前把 B 保留为可选消融与机制研究，而非主要收益模型。

## 4. 必要工程修复：证据、影响与最小验收

本节都是**待批准计划**。除明确的目标/方法变化外，所列防护不改变理论、不要求重训；不会为“让历史更好看”回写旧结果或旧 provenance。

| 优先级 / 问题 | 状态、源码证据与影响边界 | 最小修复范围 / 风险 | 最小验收与 go 条件 |
|---|---|---|---|
| P0 普通 test 重复 seed | CONFIRMED：`trainer.py::test` 899–912 非 clean 传 None，`_evaluate_epoch` 1581–1592 每次回 args.seed；当前 shuffle/augmentation 关闭。显式审计和 clean 不受影响 | 显式 evaluation-seed list，逐 run 传入并存 manifest；新结果新记录，不覆写旧均值 | CPU stub 捕获恰为指定列表；同 seed 可复现且训练 RNG 不变；之后授权窗口做真实认证 |
| P0 checkpoint 非原子 | CONFIRMED 风险，实际损坏 OPEN：`trainer.py::_save_checkpoint` 560、`goal_pretrain.py::_save_checkpoint` 313 直接写 final | 同目录唯一临时文件→flush/fsync→replace；必要时父目录 fsync。失败保留旧 final；勿套用到活跃 SDD 进程 | save/replace 前后故障注入：final 始终旧完整或新完整、可 load，失败不丢最后可用版本 |
| P0 debug/formal cache 不隔离且 consumer 删除 | CONFIRMED：`data_grouping.py` 18–58 路径无 debug，manifest 83–84 有 debug；`data_pre_process.py::__init__` 57–70 mismatch/缺失/坏 JSON 自动 rmtree；`main.py` 24–26 普通 test 也可能预处理 | debug namespace；formal mismatch fail-closed；显式 builder、realpath/ownership/symlink guard、锁和临时目录发布。风险为误删共享运行数据，故最高安全优先级 | 同 root debug/formal 不同路径；缺/坏/mismatch 保持 formal bytes；两个 consumer 不删除；冲突 writer 被锁拒绝。当前不重建 cache |
| P0 完整 RNG resume | CONFIRMED 缺口：trainer 476–559 / 1024–1057、goal pretrain 225–245 / 426–441 未存全局/loader RNG；`data_loader.py` 385–388 generator 重设 seed | 先支持完成 epoch 边界：保存/恢复 Python、NumPy、Torch CPU/CUDA、各 loader generator、next epoch/step、schema；首次 iterator 前恢复。核对实际 augmentation RNG；旧 ckpt 不伪称 exact | 真正随机 CPU 小循环 workers0/1，比较不中断/续跑 batch IDs、augmentation replay、noise、loss、参数/LR；CUDA 后续授权。stateless augmentation 会改历史流，需新协议 |
| P0 best/numbered 保存边界 | CONFIRMED：trainer numbered1128、best1162 在 scheduler1198–1205 前，last1245 在后；goal482/510/529–542 同型；resume 一律 epoch+1 | 统一 validation/scheduler/stopping 后的 resume snapshot；或明确 best inference-only。numbered 还可能缺当轮 selection 状态。风险是破坏历史恢复语义，需 schema | 用真实 `_train_loop` CPU stub 验三类 snapshot 的下一 LR、best/patience/collapse；覆盖 ExponentialLR/Plateau/非 validation epoch。FALSE ALARM：“last 也在 scheduler 前” |
| P0/P1 source provenance 不完整 | CONFIRMED 缺口，污染 OPEN：普通 manifest `data_grouping.py` 61–90 无 raw/map/transform/code bytes/构造 seed；`data_pre_process.py` 233/329 有随机 shuffle | 版本化原始数据、地图、变换、预处理实现与随机协议；下游引用 immutable manifest。大规模 hash/重建需以后安排；旧 cache 不回填伪造 provenance | 独立改 raw/map/transform/code/seed 都按协议变指纹并拒绝错用；合法 source pin 记录兼容理由 |
| P1 partial-mask finite lift | CONFIRMED API 缺口：unary82–83 非法项 −inf；sampler73–108 允许 valid_K≥P；exact313 对全矩阵 lift，72–78 拒绝非 finite。canonical `model.py:821` 全 True，故不推翻已认证结果 | lift 输入仅把 mask 外值替为有限占位，保留 hard mask；合法 NaN/Inf fail-closed。不能偷偷给非法项概率 | 真实 Unary→Sampler K21/P20/valid20，active/mixed/E0、valid<P、合法非 finite；全 finite 旧路径 exact parity；非当前 canonical 故不抢占 P1 科学审计 |
| P1 RME mask / packed E0 | CONFIRMED：`model.py` 1821–1843 忽略 metric_mask；全 E0 1823 只回 `[0]`，否则按 scene 返回 | 按 mask 两端筛边，输出 scene 对齐 value/count；明确无有效边口径。风险是统计定义变化，必须新标签，保留 legacy | 改 masked agent 不影响指标；packed/unpacked 一致；全 E0/mixed/非连续 scene IDs。现有 full-mask 单窗 ETH 不能据此直接判错 |
| P1 empty target / loss | CONFIRMED API：`metrics.py` 292–294/319–321 空 joint 返回0；`joint_goal_loss.py` 597–611 空 compact `.max()`；正式触发 OPEN | 明确 invalid/excluded 与 valid count，或显式 legacy-zero；loss 明确拒绝空 N/定义 zero+count | 空 scene 不静默稀释分数；empty/masked/packed 一致；历史结果不回写 |
| P1 gradient ownership | CONFIRMED：dynamic `[4,16]` embedding A loss 不使用，B 冻结且 no_grad；见第6节 | 先标注固定随机 mode code，增加 owner/grad 验证，不删除 key、不擅自增加 loss/解冻；收益损害 OPEN | A backward 检查 owner 列表；B 仅 corrector 可训练；区分 energy `[4,64]` 可训练 embedding |

补充边界：

- **FALSE ALARM**：“JDV2 cache 完全不 hash 数据”。`joint_dependency_v2_cache.py::split_hash` 47–55 hash source batch bytes，`build_manifest` 84–116 hash checkpoint、commit/config。缺口是 upstream raw/map/dirty-code 来源链，而非这些已做的 checks。
- **FALSE ALARM**：“所有 cache 都非原子”。JDV2 record/JSON 在 135–162 已 tempfile+replace；`batch_cache_io.py` 压缩 pickle 48–72 也原子发布，未压缩 43–45 直接写 final。目录事务/并发安全与单文件原子性不同。
- **CONFIRMED**：`data_loader.py::_validated_jdv2_manifest` 51–54 memo key 只有 path/checkpoint/pinned commit，没有完整 config/file version；同进程换 config 的验证缓存有漏洞，真实触发 OPEN。纳入新 manifest 单测即可，不声称历史已经污染。
- **CONFIRMED**：PL 是 scene-balanced，relation KL 是全 edge mean，B diffusion 是 active-agent mean，relative loss 是 edge/time/coordinate mean（`joint_goal_loss.py` 597–611、810–830、925–929；`model.py` 2868–2869、3307–3310）。统一 reduction 属于训练目标变化，不归入“无损工程修复”。
- **CONFIRMED**：`metrics.py::Collision_Rate` 371–381、412–436 以 pair 为分母且包含最后观测→首预测段；`jmm_protocol.py` 439–503 以发生碰撞的 agent 为分母、future-only、严格 `<2r`。不可与 JMM CRmean 混名。

现有测试的覆盖限制（本轮静态阅读，未复跑）：`test_jdv2_resume_stopping_state.py` 75–89/108–167 使用确定性假梯度、post-step last，不覆盖完整随机流/真实 best 保存顺序；integration 429–465 的 `isolated_random_seed` 仅 global RNG，477–500 验 selection/last；data_loader 32–85 不测 resume replay；data_grouping 144–161/370–382 不证明 mismatch 不删除；cache 28–113 不覆盖 raw/map、checkpoint 故障或目录并发；exact-production 256–272 不覆盖 Unary 产生 −inf 的端到端 partial mask。

### 4.1 性能优化必须服从已有证据

**CONFIRMED（历史计时）**：exact-adoption 报告 251–269 的 production sampler 28.434 ms/window，full 623.723 ms/window，约 4.56%。D0 sampler 10.863 ms，而 full 630.718 ms；不可用单模块差值宣称端到端加速，也不能把 exact solver 断言为主瓶颈。本轮未做同机 profiler。

| 类别 | 可以评估的范围 | 验收 / 禁止扩大解释 |
|---|---|---|
| 理论/数值都预期不变 | 静态 graph/geometry/tie metadata 重复构造、日志保留/诊断频率、artifact retention | 先确认无 RNG/执行分支副作用，再 exact parity；本轮不改日志线程 |
| 理论不变，数值/随机流可能改变 | 批量 CPU snapshot、selected-energy kernel、vectorized LSTM、BF16 reductions、worker augmentation 序列 | 必须新源码 commit、dtype/first-divergence、五种子配对；不能称自动无损 |
| 方法变化 | trunk steps、rounds、edges、K/P、mixture、训练分布/目标 | 独立研究 variant，不作为纯加速补丁 |

exact 的逐 agent 同步真实存在：`exact_lexicographic_assignment.py` 290–292 CPU score snapshot、134–155 geometry lift、379–380 degree scalar、386–387 回 device。但无损整数 lift 与 strict tie 是合同，不能以浮点 epsilon tie 替换并声称 same solver。Stage-A `[Echunk,21,21,4]` 单个 FP32 cost tensor 为 `7056×Echunk` bytes，尚不含 factors/geometry hidden/autograd；`model.py` 2782–2858 的各 chunk graph 保留到总 backward，不能认为 chunk 自动封顶峰值。2834–2846 的 CPU diagnostic 同步有优化候选，实际收益仍 OPEN。


## 5. 唯一结构候选：geometry4→6，目前 NO-GO

**CONFIRMED**：当前四维含候选 endpoint distance、displacement cosine、relative displacement norm、displacement norm 差（`dynamic_relation.py::geometry` 46–82）；历史 graph 也含 TTC/closest-distance（`interaction_graph.py` 186–205、242–255），energy 输入有历史与候选位移。因此不能声称整个系统完全不知道途中交互，也不能仅凭 endpoint error 判断需要加层。

设 \(r_0=x_j-x_i\)，\(d=(g_j-x_j)-(g_i-x_i)\)。候选定义为

\[
\tau_* = \operatorname{clip}\left(-\frac{r_0^\top d}{\max(\|d\|^2,\epsilon)},0,1\right),
\quad d_{min}^{lin}=\|r_0+\tau_*d\|.
\]

新增 dmin_lin/radius、tau_star，只将 Linear(4,32) 改为 Linear(6,32)；bias 与后续层不变。**CONFIRMED：只增加 2×32=64 个 weights，dynamic 模块 820→884 参数**。M4、rank8、Social128、K21、P20、两轮 sampler、目标/其余模块全部保持。

**CONFIRMED（代数边界）**：

\[
r_0^\top d=\frac{\|r_0+d\|^2-\|r_0\|^2-\|d\|^2}{2}.
\]

给定同一 edge 的历史距离 \(\|r_0\|\)，现有 endpoint norm 与 relative-displacement norm 已可恢复 dot product，进而恢复上述两个新量。在精确实数、同 radius、未舍入 descriptor 的条件下，同一 edge 若旧四维完全相同，新两个量也相同；不能声称它们在数学上区分这种完全相同输入的候选对。FP32 norm/除法可能把不同几何舍入为相同 old4，直接从坐标算出的新量未必逐位相同；此代数结论不是 tensor-exact 认证。历史信息经过 base logits 压缩后，当前加性网络是否容易利用这一关系是 **OPEN**；合理表述是将可推导量显式提供给 candidate-conditioned 模块的归纳偏置/优化便利，而非新增原始观测信息。

仅在后续同时满足以下条件时才申请新 variant：

1. frozen A 的完整 trajectories 显示可重复的途中交互失败，而不是只有末端误差；这些 pair 已被当前 graph 覆盖。
2. 对应候选的直线 surrogate 与真实失败存在可检验联系；旧模型对这种可推导几何利用不足有证据，而不是把过近直线误认实际碰撞。
3. 固定预算、selection/test 与训练 seed 的单变量设计预先冻结，并能控制 marginal 损失和 runtime。

任一条件不成立则 NO-GO。它不新增 missed edges，不读取未来 GT 构造 deployment descriptor，不等于完整 trajectory awareness，不保证避碰；不因 +64 参数便自动有论文创新。首层 shape 改变须独立 Stage-A 重训和新 strict checkpoint schema，禁止 partial-load 伪装原 frozen 方法。当前没有上述失败证据，因此从本轮实验队列删除该训练项。

## 6. 保留的实际网络、shape、参数与梯度路径

以下参数量是本轮按源码层定义静态核算，**不是本轮运行模型后的 numel 输出**。构造入口为 model.py 213–228、408–448，freeze 为 494–506；实际归档配置包括 jdv2_stage_a_no_z_full_seed2035/config.yaml 的 branch14/e_dim256/obs8/pred12，不能只凭默认 parser 推断网络。

| 模块 | 执行层 / shape | 参数量 | 当前 gradient owner |
|---|---|---:|---|
| Goal UNet | 输入6 scene+8 history=14 channels；五个 DoubleConv stage：14→32→32→64→64→64；四个 decoder stage：64→64→64→32→32；1×1 head32→12 maps | 613,772 | 上游 goal pretrain；A/B 冻结 |
| GDTS history | 单层 LSTM，逐时8→hidden256 | 272,384 | 上游 independent；A/B 冻结 |
| GDTS denoiser | [N,12,2]；ConcatSquash2→512，context259；Transformer **2 层**、d512、4 heads、FF1024；ConcatSquash512→256→128→2 | 注册6,940,432；forward 使用4,837,648 | 上游 independent；A/B 冻结 |
| Social | Linear4→128；单层GRU128；message270→128→128；update256→128→128；LN128；h[N,128] | 200,576 | A 的 PL/KL 路径；B 冻结 |
| History relation | 共享 MLP270→128→128→4，双方向 logits 平均；[E,4] | 51,716 | A；B 冻结 |
| Unary | goal4→64→64+LN；agent128→64；context LN64；concat192→64→1，输出层零初始化；[N,21] | 25,409 | A PL；B 冻结 |
| Dynamic relation | geometry4→32→16；query[4,16]、bias4；另有 embedding[4,16]；full logits[E,21,21,4] | 820（其中64无 A/B loss owner） | A 学 geometry/query/bias；另见下文 |
| Energy | candidate4→64→64；pair270→128→64+LN；Embedding4×64；fusion LN64；factor64→8；factors[E,4,21,8] | 48,456 | A PL；B 冻结 |
| strict-no-z future teacher | 训练 descriptor6→64→64→4；无 z | 4,868 | A PLpost 与 KL(q‖p)；不部署 |
| Dependency corrector | state21→64→64+LN；time32→64→128 FiLM；gate64→32→1；value64→64→64；output64→64→2，zero final head | 30,851 | B diffusion/relative；A 冻结 |
| Graph / sampler / projection | sparse canonical edges；K21/P20，两轮同步 exact；逐 component/time/coordinate 去 agent 均值 | 0 | 无参数 |

**CONFIRMED**：GDTS 三模块注册总数 7,826,588。diffusion.py 90 的 self.layer 仍注册，91 的 TransformerEncoder 克隆两层，forward109 只执行后者；不能把额外注册的 2,102,784 参数解读为第三个执行层，也不建议删除它破坏 frozen checkpoint namespace。Goal 层见 model_utils/U_net_CNN.py 6–141，LSTM 见 hist_traj_rnn_encoder.py 39–42，ConcatSquash 见 common.py 51–64；其余层见各模块 __init__。

Stage-A 新模块 requires-grad 数为 331,845，loss-connected 数为 331,781，差值64；Stage-B 仅 corrector 30,851。A/B 梯度主线：

~~~text
frozen goal bank + history
    → Social → unary / history relation / energy factors
    → candidate-conditioned prior p → PLprior + KL(q || p)
GT future descriptor → learned teacher q → PLpost + KL(q || p)

A objective: 0.5 PLpost + 0.5 PLprior + beta KL
              （q、p 均未 detach）

B: frozen A encode/sample → scene endpoint-oracle branch (no_grad)
   + frozen GDTS + noisy GT state → corrector → component projection
   → diffusion loss + 0.05 relative-motion loss
~~~

证据：model.py 1531–1535、2757–2912、3230–3322；joint_goal_loss.py 833–860。**CONFIRMED owner 缺口**：dynamic_relation.py 42/44 的 [4,16] relation_embedding 只在 selected_joint_relation 261–265 被取期望；A loss 调用 _jdv2_encode 时不 sample（model.py 1079–1081），不使用该输出；B 3234–3253 的 frozen encode 又在 no_grad 下。它目前是固定随机 mode code，不是已学 relation representation。它仍可能提供有用模式编码，不能从无梯度直接推断“无效”或立即新增 loss。不要与 joint_energy.py 的可训练 [4,64] embedding 混淆。

**CONFIRMED 对称性与单位边界**：

- Graph canonical edge 同 scene、每无向边一次；edge14 的反向符号处理在 interaction_graph.py 14–48，构图171–205，基础权重259为1。Social 聚合对 agent permutation 等变，但 E0 aggregate=0 后仍 update+LN。
- History relation 共享双向 MLP 后平均（relation_inference.py 64–70、126–145）；dynamic geometry 对边反转不变；energy 的边反转对应左右交换/candidate axes 转置（joint_energy.py 37–65、86–115）。整体坐标 MLP/GRU 不保证全局旋转不变。
- pathwise agent-permutation 检查需同时输运 noise/tie payload；exact 的 seed 含 agent index（46–57），重置同 seed 并不等于同物理 agent 收到同随机量。
- Social/history relation 在 model.py 835–845 强制 FP32；dynamic 113–134/207–231、energy dot/effective129–152/216–237、conditional accumulation472–477、projection141–166 有 FP32 island。不能把 unary、energy factor、corrector 全部 MLP 一概称为 FP32。
- Corrector 输入 noisy velocity 经 world adapter（3197–3209），输出是 epsilon residual，不是 m/s 速度修正；零和由 projection 保证，不由未投影网络自动保证。
- ETH 几何须确保 last/goals 同单位。scene_sdd.py 49 明确 pixel，96–101 函数名 world 不是米制证据；radius6m、collision r0.1m 不可照搬 SDD。

geometry6 若以后获准，唯一拟改伪代码如下；full-pair/selected-neighbor 必须共用 descriptor：

~~~python
# 仅设计，未实施；所有输入来自 history/frozen candidates，FP32 island
with torch.autocast(device_type=xi.device.type, enabled=False):
    r0 = xj.float() - xi.float()
    d = (gj.float() - xj.float()) - (gi.float() - xi.float())
    den = (d * d).sum(-1)
    tau = (-(r0 * d).sum(-1) / den.clamp_min(eps_len_squared)).clamp(0, 1)
    tau = where(den > eps_len_squared, tau, zeros_like(tau))
    dmin_scaled = norm(r0 + tau[..., None] * d, dim=-1) / graph_radius
    geom6 = cat([old_geom4, dmin_scaled[..., None], tau[..., None]], -1)
    h = linear_32_to_16(silu(linear_6_to_32(geom6)))
    logits = base_logits[..., None, None, :] + einsum('...h,mh->...m', h, query) / 4 + bias
    log_p = log_softmax(logits, dim=-1)
~~~

FP32 island 必须显式禁用 autocast；只有输入 .float() 不足以保证其内 Linear/einsum 执行 FP32（参见 dynamic_relation.py 113–117）。eps_len_squared 与 den 同单位（长度平方），small-motion 取 tau0；半径必须正且同单位。empty/masked/nonfinite 输入需明确 fail/skip 合同，不对 NaN 静默清洗。边反转时 r0,d 同时变号，新量不变；平移/旋转性质只针对 descriptor，不等于整网对称性。CPU 公式验证见第10节，未验证任何模型收益。


## 7. V4 去重与 primary related-work 定位

### 7.1 仓库已经有完整轨迹双射耦合

**CONFIRMED**：V4 不是只配 goal。multiway_trajectory_coupler.py 30–56、147–166、211–234 编码完整 bank，之后 hard integer gather；trajectory_pair_relation.py 154–264 使用12维候选路径几何（含最小/平均距离、相对运动、最近离散时刻），267–333 使用7维时序特征→GRU128；trajectory_pair_energy.py 31–87、410–458 使用 rank8 factor 与 mixture。permutation_synchronizer.py 475–507 为 C@Pj / C.T@Pi 消息与 Sinkhorn，528–534 为训练 soft mix，628–655 为推理 Hungarian/identity bypass；hungarian_projection.py 33–52、78–84 保证双射。训练软表示不等于最终 hard-gather，但推理输出不修改单条轨迹坐标。

| 历史 V4 结果，不是本轮复测 | raw→aligned JADE | raw→aligned JFDE | marginal gap | changed fraction |
|---|---:|---:|---:|---:|
| ETH | 0.50811→0.45973 | 0.89124→0.76780 | 0 | 0.12713 |
| UNIV | 0.65580→0.65467 | 1.32055→1.31784 | 0 | 0.00292 |

证据：[ETH V4 报告](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/docs/ETH_MULTIWAY_COUPLING_V4_PACKED_RESULTS.md) 80–99、[UNIV V4 报告](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/docs/UNIV_MULTIWAY_COUPLING_V4_PACKED_RESULTS.md) 80–99。UNIV 接近 identity 不是继续堆 coupling 容量的证据。

**CONFIRMED 协议边界**：V4 的上游是 independent epoch100，内部 source-block 320 windows 做 selection；ETH/UNIV test 为139/947 windows。其 upstream、selection 与 JDV2 frozen A 不同，旧表不能直接与 post-fix A 排名。实际 cached test 为 **4 seeds、每 seed K20**：trainer.py 890–908 让 manifest 覆盖 num_test_runs，train_packed_multiway_v4.sh 81–85 设4，GDTS/output/{eth,univ}/joint/runs/multiway_coupling_v4_packed_cache_v2/evaluation_protocol.json 内嵌 manifest 为4且 diagnostics 为 seed00–03。不能将报告建议的20 seeds或 YAML默认5写成实际执行预算。

**CONFIRMED 公式风险 / OPEN 历史逐窗成因**：ETH 旧 recovery_ratio=−64195.78261；multiway_coupling_loss.py 1096–1117 平均逐 group 的 (raw−aligned)/max(headroom,eps)，近零 headroom 可使比例爆炸。没有逐窗归因前不把它当有效“回收 headroom 百分比”；原始 JADE/JFDE 可分别讨论。

已静态阅读 V4 tests 的 shape/gradient/reversal/bijection/packing/marginal/empty cases（test_trajectory_pair_v4、test_permutation_synchronizer_v4、test_trajectory_bank_packing_v4、test_multiway_coupling_loss_v4），本轮未复跑。V4 提供内部先前实现，不等于当前 A 的配对贡献已验证，也不能把“固定 bank 重排”包装为新架构。

### 7.2 论文检索边界与比较

本轮按 nature-academic-search 的 primary-source/证据分级规则检索；当前未提供其学术 MCP，OpenAlex 请求 DNS 失败，采用可访问的论文原文/官方 proceedings 回退。实际阅读范围为摘要及相关方法/评估段落，不声称对全部论文做了全文复现；检索截至 2026-10-03，**novelty 仍 OPEN，不是穷尽性优先权认证**。

| Primary source / 已读范围 | 目标、joint 变量、信息与预算 | marginal / 推理边界与本项目差异 |
|---|---|---|
| Joint Metrics Matter，ICCV2023；指标方法与评估段 | 以共同 world 衡量多人预测，区分 marginal best 与 joint best | joint metrics 是已有评估背景，不是 RSJG 创新；其官方 ETH scene 口径不能与 GDTS139窗混合。[论文](https://arxiv.org/html/2305.06292v2) |
| Joint Pedestrian Trajectory Prediction through Posterior Sampling，2024；方法相关段 | 以 diffusion posterior sampling 构造完整多人未来，区别于离散 goal allocation | 不提供 RSJG 固定 P20 bank 双射合同；不能把本项目称为首次 joint diffusion/未来交互。[论文](https://arxiv.org/html/2404.00237v2) |
| QCNeXt，2023；decoder 方法段 | 共享 scene mode 与未来交互 decoder；AV2 常用6 modes/60 future steps | learned trajectory decoding，不是冻结完整集合的 permutation；“future interactions”本身不新。[论文](https://arxiv.org/html/2306.10508v1) |
| JFP: Joint Future Prediction with Interactive Multi-Agent Modeling for Autonomous Driving，CoRL2022/PMLR2023；官方摘要 | 交互式 joint future、结构化 pair 建模，WOMD | pairwise joint modeling 有前作；本轮仅摘要级比较，不据此断言其全部 solver 保证或具体 runtime。[官方论文页](https://proceedings.mlr.press/v205/luo23a.html) |
| From Marginal to Joint Predictions，2025；§IV-A 等 | 在 AV2 对 marginal modes 做 product-confidence/beam top-K recombination，并比较 joint alternatives | 重用 marginal proposals 不等于每人 P 条完整轨迹恰好一次；top-K 可重复/丢弃候选。预算/数据均不可与 ETH 直接排名。[论文](https://arxiv.org/html/2507.05254v1) |
| JAM: Keypoint-Guided Joint Prediction after Classification-Aware Marginal Proposal for Multi-Agent Interaction，2025；摘要/方法 | marginal proposals→keypoint-guided joint refinement，WOMD | 会生成/调整 joint trajectories，不是固定 empirical bank 严格双射；“marginal→joint”两阶段本身不新。[论文](https://arxiv.org/html/2507.17152v1) |
| Projected Coupled Diffusion，2025 preprint/ICLR2026；方法与官方条目 | 多个 pretrained marginal diffusion 的 test-time coupling/projection，处理 joint constraints | 采样坐标可改变；hard constraint 与固定经验 marginal 不是同一合同，任务也非本 ETH 预算。[原文](https://arxiv.org/html/2508.10531v2)、[ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d3831fa12e608fc9f7690e05b599e69e-Abstract-Conference.html) |
| From Learned-Mode AV–Traffic Pairing to Planner Decisions，2026-09-14 arXiv v1；§III-B 与 limitations | 固定预测轨迹，5 learned modes 做加权 Cartesian product，再保留1个固定模式，共26 combinations；AV2 planner counterfactual | 直接涉及 marginal-preserving pairing destruction：保留加权 AV/traffic marginals，traffic 内部仍一起；不同于每 agent 独立 P20 双射。论文承认干预改变 conditioned concentration，不能隔离其与 pairing 的全部 outcome 效应。故审计本身也不能宣称首次提出。[论文](https://arxiv.org/html/2609.14997v1) |

上述比较支持的是定位边界，不是各方法优劣排名。尤其最新 preprint 的 existence/方法说明按原文确认，接受状态与完整优先权不作额外推断。其 weighted Cartesian control 与本项目 fixed-budget random bijections 不同，但“保持 marginal 检查 pairing”已不是空白。

### 7.3 最多一个科学主线

建议主线仅保留：**冻结 marginal generator、固定有限候选/采样预算，分离 coverage、单边 ranking 与不可分解 interaction，构造可认证的 joint worlds，并约束 marginal 损失。**

这是研究目标，不是已证明的新理论。成立需同时有：明确区别于 V4/现有 prior art 的机制或推理约束；I 的可归因收益；独立 selection/test 与跨 sequence/training-seed 的证据；准确陈述 sampler 保证。只有 bug 修复、gauge 公式、已有 permutation 或6维 MLP，则当前不足以主张新增方法论文贡献。完整 trajectory permutation 能严格保住 empirical marginals；当前 goal allocation→diffusion 只能把 marginal no-harm 当经验约束，不能偷换为相同数学保证。

## 8. 单变量实验计划：仅保留3项，顺序执行

**本轮均未执行。** 第1项是唯一下一动作；第2项在其之后，第3项只有前两项支持继续分析 candidate conditioning 时才申请。geometry6 重训、lambda_relative .05→0 重训目前条件不成立，删除，不为凑5项安排训练。

所有真实模型实验先按原 config strict load/认证 checkpoint 与来源；独立 wrapper/内存干预不篡改 hash，不更改 frozen 方法，不放宽 strict load。若后来实施补丁/评估，新增结果须记录实际新源码 commit、dirty 状态/patch digest、checkpoint/config/protocol/bank hashes、硬件与精度；历史结果保留。

### E1：完整 bank 的 marginal-preserving 配对破坏（优先级最高）

- **假设（OPEN）**：现有 A 的 common-column 配对相较独立随机双射有实质 joint 优势。
- **唯一变量**：scene 内每 agent 的 P20 slot permutation。模型/坐标/完整每人轨迹集合不变。
- **固定量与位置**：第3.3节全部合同；trajectory 世界坐标输出之后、metric 之前；GT/mask/graph/provenance不变。使用本地 RNG，不扰动训练全局 RNG。
- **预算/协议**：2035–2039 每 seed20条完整轨迹，各100个预定 permutations；原139窗口径作为 frozen-protocol诊断，不做 checkpoint selection、不冒充新的 held-out test。只有控制通过后再考虑外部干净协议。
- **训练/计算**：不训练；认证 bank 已有时纯 CPU gather/metrics，碰撞计算依赖各 scene N²，不承诺未经测量的秒数。若缺 bank，须后续授权一次导出作业（包含五个 inference seed），不能把本轮“只读审计”默认为导出授权。
- **输出**：bank manifest、逐 agent inverse/marginal checks、逐 scene/seed/replicate 指标、分层 Δ、block/sequence uncertainty、CR-JADE tie 诊断、失败 case IDs。仅未来获准后写新目录。
- **验收与负控**：inverse exact、逐 agent minADE/FDE exact、N1/common-π 控制按第3.3节通过；预注册 ε_joint。原配对优势稳定超过阈值才支持继续“joint correspondence 有效”叙事；零/负效应则收缩该叙事，不用改变抽样 null 补救。

### E2：只移除不可分解 I，保留 pair 单边 ranking

- **假设（OPEN）**：在固定无放回约束、K/P 预算与 Round0 下，I 带来超过单边项的 joint 收益。最终选中的20个候选ID集合可以改变，因此本实验不保证完整 trajectory marginal 保持。
- **唯一变量**：selected effective cost 从 full E 变 g+a+b；或等价 compensated unary 加 zero cost，但始终保留两轮 solver。
- **实际干预位置**：joint_sampler.py 462–477，先记录/固定原 unary 生成的 Round0 IDs、tie payload 与 noise，在相同 legal support 分解 full cost。full/selected 计算路径数值差异先认证，禁止 legacy flag 假消融。
- **固定量/预算**：frozen A、K21/P20、M4/rank8、graph、两轮、原始候选、同2035–2039；原协议139窗，先记录不是独立泛化。不训练。新选 goal 后，context 由新 goal 按原代码重建；“固定生成规则和随机 noise”不等于强行复用不匹配旧 goal 的 context。
- **计算与输出**：代数/factor统计可 CPU；真实 trajectory指标需要之后授权的配对推理，约 baseline/no-I 两臂五 seed 的任务级预算，已有可认证 baseline可复用。输出 g/a/b/I 范数/奇异值、Round0/1/2 IDs、score/near-tie、marginal/joint/CR/RME 与 runtime；无 profiler许可就不启 profiler。
- **负控**：原始 wrapper 精确复现 baseline；完整 gauge 加补偿在 toy exact 通过，并对真实 FP32 的 score常数误差和 decision差异独立报告。E0保持原路径。两种 no-I 写法数学相同，舍入差异不可当科学效应。
- **go/no-go**：只有 I 带来的 joint优势达到预注册 ε_joint、marginal损失≤δ_marginal，并跨有效 block/sequence稳定，才支持“不可分解交互有效”。若移除 I 无损甚至更好，撤回该解释；不能把 a/b 的改善算到 I。

### E3：冻结模型下 candidate-conditioned relation→history-only（条件性）

- **假设（OPEN）**：动态候选条件化对前两项发现的交互效应必要。
- **唯一变量/位置**：model.py 910→sampler454/460/500→dynamic_relation.py 186–191，把 p(k,l) 切换为 history-only base logits；保持 energy factors/unary/graph/solver 不变。不改变 teacher 或重训。
- **固定量/预算**：与 E2 同 checkpoint、K/P、原 Round0/tie/noise、两轮与五推理 seeds；139窗仅诊断，同样没有新 selection。模型结构与参数不变，context按各自所选goal合法生成。
- **计算/输出**：待授权的 baseline/history-only 配对推理；报告 prior KL/候选间方差、effective costs、assignment churn、轨迹指标和分层。没有实测不填精确 GPU 时长。
- **有效性/负控**：flag真的使所有候选对 logits退化到相同历史prior；单独构造原 dynamic residual=0 时两臂一致；history-only不等于 energy-off，确认非零 pair costs仍生效。
- **go/no-go**：超过预注册 joint/marginal阈值才支持 frozen dependency；无变化则不主张它是必要贡献。无论结果如何，不自动等于“从零训练删模块也无损”。E1/E2没有提出有价值的新问题时跳过本项。

只有一份 frozen checkpoint 时，不足以衡量 training-seed uncertainty；这些实验不能靠更多 inference seeds 补足。未来 publication-level结论需要独立训练重复与干净split，但当前没有训练授权，也没有在此计划中新增训练项。


## 9. SDD 运行保护、bank 可用性与跨 dataset / clean 门槛

### 9.1 SDD 是活跃 independent baseline，不是 JDV2

**CONFIRMED 只读快照，2026-10-03 17:38:37 +08:00**：

| PID | PPID / PGID | 身份 |
|---|---|---|
| 962146 | 1764 / 962146 | tools/run_sdd_paper_baseline.sh all 的 pipeline |
| 962176 | 962146 / 962146 | goal_pretrain 主进程，13:33:37 启动 |
| 993069 | 962176 / 962146 | 子 worker，同命令行并不代表另一份独立训练 |

当前主进程参数为 SDD、goal_model_type=independent、training_stage=baseline、seed2025、150 epochs、augmentation=true、num_workers=1、FP32。本轮较早的只读 GPU owner 查询只有 962176、8192MiB；不是从两个相同命令行推断两份训练。未修改上述进程或设置。

活跃 stdout 对应：
RSJG_JDV2_clean/output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025/resume_after_loss_mask_fix.log。
17:38 快照为 **Epoch4，1970/2440 batches，约81%**；会继续变化，不承诺完成时间。旧 SDD 状态文档、IDE 中旧 seed2035 baseline resume/log_curve，以及历史失败的 pipeline.log 都不能替代当前活跃日志。尚未看到后续 trajectory baseline 阶段启动的证据，不能称 SDD baseline 已完成。

SDD 完成前可以继续：源码/小 JSON/日志只读检查、有限 CPU toy、方案与 provenance 设计；若有已认证 bank，可另行批准纯 CPU 审计并限制资源。不能启动 main.py 普通 test 来“顺便读指标”，因为其入口可能触发共享 preprocessing/cache 删除；也不启动新 CUDA tests、导出、评估、profiler、训练或 cache 重建。

### 9.2 bank 定位结果不是“已具备审计输入”

**CONFIRMED**：post-fix paired 工具 tools/jdv2_postfix_paired_five_seed_evaluation.py 117–197 累积指标、parity 与计数，但没有保存完整 Y。

已定位的完整旧导出：

~~~text
outputs/joint_dependency_v2/eth/joint_dependency_v2/jmm_official/
  stage_b_v2a_epoch005_seed2035_commit05378c7/
    trajectories/gdts/biwi_eth/frame_*/
      obs.txt, gt.txt, sample_000.txt ... sample_019.txt
~~~

其 manifest 是 V2-A epoch5、checkpoint e4c114…7b73、source05378c7、correction-on、seed2035、P20、253 scenes/364 agent instances、JMM official world-metres 口径；既不是 post-fix A，也不是139窗五种子 bank。不能混入 A 的审计或据此声称 A 贡献已测。没有 selected-goal auxiliary，不报对应 goal metrics。

V4 的 trajectory_bank_cache.py 183–215 定义 raw_trajectory_banks[S,N,K,T,2]、obs/GT/masks/IDs，349–392 packing 保留 seed 轴；本轮未在输出路径找到其独立 raw-bank 文件，但历史 evaluation_protocol 已嵌 manifest，不能说历史没有缓存。外部归档是否有 post-fix A bank为 **OPEN**。

未来合格 bank manifest 至少需要：actual route A/B/C/D、source commit/dirty digest、checkpoint SHA256、resolved config、dataset/split/selection、source window与agent稳定ID、seed/branch budget、dtype、单位/坐标变换、GT/mask/graph fingerprints、artifact SHA256、auxiliary schema。若用已认证 C base-route 代替 A，必须显式标 C 并提供适用于该 bank 的 A=C 认证，不能默默改名；D不允许替代。

### 9.3 clean 与跨数据集的放行条件

**CONFIRMED 现有限制 / OPEN 泛化**：

1. ETH 原 valid/test 镜像：既有数字是固定原协议诊断，不是严格 held-out generalization。新 clean 协议须先固定训练/selection/test source boundaries，检查重叠轨迹、时间窗与缓存来源；不能用看过的 test 选择 geometry/阈值。
2. 上游 marginal generator、goal candidate数量、完整 trajectory数、diffusion budget、checkpoint selection 和 inference-seed预算一致后，才能比较 A、V4、独立 baseline；不能从现有异协议旧表直接排榜。
3. SDD 像素/变换、帧采样与预测时间、窗口/agent mask、图半径和 collision阈值必须重新认证；不要因变量名 world 自动声称米制。此门槛满足前不复用 ETH 的米制几何解释。
4. canonical masks/partial masks、N1/E0/mixed/all-active、packed/unpacked 与 empty target 均要有口径；strict checkpoint/schema、post-fix身份、显式2035–2039是之后每个新实验的 provenance，不是覆盖旧结果的理由。
5. inference repeats 估计随机采样不确定性；跨 training-seed/sequence证据另需设计。效应与容忍阈值在真实结果前冻结，不从小数差事后编造“显著”。

## 10. 最终结论、实际检查记录与唯一下一动作

### 10.1 决策

- **现在改结构吗？不改。** geometry6 只有 +64 weights，但价值 OPEN，当前没有失败证据满足 go 条件；没有 underfitting 证据支持扩容量。
- **B 是否作为主要收益模型？目前不支持。** post-fix D/C 为 CONFIRMED mixed trade-off，保留 B 作为可选消融/机制分支；不新增 loss、gate、attention 或训练。
- **哪些修复必要？** 第4节 P0 可信度/恢复/保存/cache安全缺口应修；partial-mask、metrics、owner与manifest细项需明确合同与单测。identity已修复，不再重复立项。修复只是待批准计划，本轮未实施。
- **SDD 完成前能做什么？** 只读证据、受控小型 CPU 推导、整理计划；不动现有运行。完成也不自动授权后续训练/评估。
- **唯一下一动作：定位并认证 post-fix A 完整 trajectory bank，然后执行 E1 的 marginal-preserving 配对破坏审计。** 缺 bank 时停在导出申请门槛，不能用旧 D 代替或擅自启动 GPU。当前可交付的是完整审查与可执行合同，不是尚未产生的科学结果。

### 10.2 本轮实际执行的 CPU 检查

采用独立临时脚本 /tmp/rsjg-optimization-review-tD0ICc/checks.py，Python -B、无模型执行、无 CUDA；仅 NumPy/stdlib 与只读导入 jmm_protocol 的 collision helper。脚本 SHA256：

~~~text
4b9dd726d69c735a6a9694357bc05f3fccd6affc7be93f369f3c62265be0fa11
~~~

| 检查 | 本轮输出 | 能说明 / 不能说明 |
|---|---|---|
| 三份 post-fix JSON | SHA/status/source_commit/invariant/parity 通过；63组summary/delta复算一致 | 认证记录自洽；不是本轮重新执行 CUDA |
| exact rational gauge | 100全支持问题×27 assignments=2700；20 partial-support问题×12=240；共2940个全局赋值等式通过，I行列均值为0 | 分解/补偿与no-I代数正确；不证明 GPU舍入决策一致 |
| 完整轨迹配对 | 60个随机scene，N1–6各10个；逐agent marginal、inverse gather exact；50个多人scene joint值改变；共同π最大误差2.22e−16 | 验证不变量，不是实际 A 的 pairing增益 |
| 构造反例 | marginal minADE仍0，JADE由0→5 | 单人指标不变时joint可变化；不是模型性能差值 |
| CR-JADE exact tie | common reverse后官方 first-index CR 1→0，JADE不变 | common permutation负控必须处理tie；不证明真实数据存在大量tie |
| geometry6公式 | 100随机案例反转/平移/旋转 +1 stationary；最大误差5.27e−16，容差1e−12 | descriptor性质成立；不证明网络收益或避碰 |

以上是本轮独立执行，不把用户 prompt 自带的 toy 结果当作本轮执行记录。机器可读结果另存为 [CPU与artifact检查记录](checks.json)。临时脚本可能被系统清理；关键公式、检查范围与结果已在新报告中保留。

两份重点历史报告复核 SHA256 未变：

~~~text
RSJG_JDV2_Audit_cfa3dd2e.md
00a22338153c9c5e51166021f3dfaa2f0cb3cc6349f29e2b8db98f57fa878549

JDV2_CANONICAL_STAGE_A_IDENTITY_FIX_VALIDATION.md
428f7171fac284b2fac3cce8e8fcdc5865b645bf3d1a41dc9891fc1e28cfafe1
~~~

新报告与检查记录在仓库外层工作目录，不覆盖 docs、outputs 或历史 JSON。报告写入遇到本地沙箱 mountinfo 读取故障，使用 apply_patch 对本轮新建文件完整发布；没有借此改动受审文件。

### 10.3 固定源码导航补充

以下均为 1f07e7a5b81374033377c5c057d5c6ab3673b02a；前文数字行号不可无核对套用于后续 commit。

| 文件（src/下） | 类 / 函数及关键行 |
|---|---|
| models/joint_dependency_v2/unary_goal.py | UnaryGoalResidual.__init__14；forward36；mask82–83 |
| models/joint_dependency_v2/dynamic_relation.py | DynamicHypothesisRelation.__init__21；geometry46–82；history-only186–191；selected_joint_relation261–265 |
| models/joint_dependency_v2/joint_energy.py | RelationSpecificJointEnergy.__init__17；factor37–115；effective cost118–152；selected216–237 |
| models/joint_dependency_v2/joint_sampler.py | ParallelConditionalSampler.__init__282；forward347；two-round branch446–487 |
| models/joint_dependency_v2/future_teacher.py | future_pair_descriptor12；SceneFutureTeacher.__init__48 |
| models/joint_dependency_v2/dependency_corrector.py | DependencyCorrector.__init__29；forward52 |
| models/joint_dependency_v2/component_residual_projection.py | build_component_metadata21；component_zero_mean_projection119（核心141–166） |
| models/joint_dependency_v2/exact_lexicographic_assignment.py | lift_fp32_to_integers72；solve_exact_persistent_tie278；exact_persistent_refinement358 |
| models/social_encoder.py | _SocialMessageLayer.__init__18 / forward35；SocialMotionEncoder.__init__104 / forward203 |
| models/relation_inference.py | RelationInference.__init__38 / forward103 |
| models/model.py | 构造408–448；freeze494–506；A strict-no-z2757–2912；B loss3230–3322；inference identity3781–3902 |
| trainer.py / models/goal_pretrain.py | checkpoint/test/train/evaluate 的具体保存与恢复点见第4节；不是只检查配置名称 |
| metrics.py / jmm_protocol.py | empty joint292–321；Collision_Rate371–436 / official collision439–503、CR-JADE565–571 |

核心固定链接：[energy](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/src/models/joint_dependency_v2/joint_energy.py#L118)、[sampler](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/src/models/joint_dependency_v2/joint_sampler.py#L446)、[geometry](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/src/models/joint_dependency_v2/dynamic_relation.py#L46)、[V4先前实现](https://github.com/WuYanXingege/RSJG/blob/1f07e7a5b81374033377c5c057d5c6ab3673b02a/docs/MULTIWAY_COUPLING_V4.md)。
