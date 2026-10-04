# 创新、理论与网络架构：决策审查

日期：2026-10-04，Asia/Shanghai。源码 79ea3fa5e94afe6818f8c8299b89aba1d677589e；src tree a2db08541129ba93e16d288867e7ec4fbd8e8c95。本轮仅设计，不是实验结果。状态 DESIGN_COMPLETE_EVIDENCE_PENDING：设计规格完成，收益、新颖性强度和外部复现均未获实验认证。

## 1. 决策

**主线 A：保留现有网络，条件性试验“对硬邻居条件损失取期望”的学习目标，新增参数 0。备选 C：复用已有 V4 的完整轨迹有限集合耦合；不是新架构。B 的 geometry6 仅作失败案例驱动的第三顺位条件方案。现在不扩展 Stage-B、不改 diffusion、不训练。**

A 不是修复一个已证明导致性能下降的 bug。现目标作为 soft-neighbor surrogate 合法，但它不是 soft-label 下 exact conditional pseudo-likelihood 的期望。建议用受控的 MC 目标对照测试这一差异是否重要；不得把数学差异当作收益证据。实现前不要求完成历史 CPU exact-margin 审计。

当前最可信贡献是：冻结 bank 的跨 agent 对应关系含有联合指标信息；以及已实现、已认证的覆盖与数值身份工程合同。最缺的是 learned pair term 超过 unary/ranking/coverage 的贡献和干净数据协议。不能把 joint metrics、未来交互、diffusion+energy 或 V4 换名列为首创。

交付导航：[逐层架构](CURRENT_AND_PROPOSED_ARCHITECTURE.md)、[理论](THEORY_REVIEW.md)、[原始文献](LITERATURE_COMPARISON.md)、[伪代码/配置](IMPLEMENTATION_PSEUDOCODE.md)、[实验预注册](EXPERIMENT_PREREGISTRATION.md)、[机器结果](RESULTS.json)、[静态参数与数学检查](STATIC_CHECKS.json)。

## 2. 主张—证据—缺口

| 主张 | 状态 / 现有证据 | 缺口 | 允许表述 | 禁止表述 |
|---|---|---|---|---|
| A 的配对有信息 | SUPPORTED_WITH_LIMITS；139窗×5 inference seeds，固定每人完整P20轨迹集合，独立打散 ΔJADE=0.03391811925732022m，ΔJFDE=0.0774227560742364m | 一份 raw sequence、重叠窗口；未干预 energy | 该固定 bank 对预登记随机配对 null 有优势 | learned energy 已有因果增益、五次独立训练、695独立场景、SOTA |
| 优势普遍存在 | SUPPORTED_WITH_LIMITS（仅分桶事实；普遍性不支持）；all-active 62窗 ΔJADE≈0.075801，mixed 30窗≈0.000498；47 E0均N1 | 无真实多人E0；activity与N/难度混杂 | 主要集中 all-active | 图激活导致改善、所有图桶受益 |
| h 微差不影响 IDs | CONFIRMED，仅预选 h 对、8次有界调用 | 其他输入/决策边界未审 | 本次Round0/1/2及最终IDs相同，真实Gumbel匹配历史P2 | 所有扰动均无影响、pair energy无效 |
| 全片段确定 | OPEN（逐位确定已被反例否定，原因仍开放）；固定h的Round2 score仍有7.62939453125e-6差 | observer中立性、底层原因、旧29IDs因果链OPEN | 连续变异未在该h对引起离散改变 | 已解释历史差距、整个片段逐位确定 |
| A/B/C identity | CONFIRMED，已完成修复及FP32/BF16/reload历史认证 | 无需再次设计修复 | C使用V2-A权重但correction arithmetic关闭 | 启用corrector的V2-A=D与A相同 |
| corrector有全面收益 | SUPPORTED_WITH_LIMITS（当前不支持全面收益）；真实post-fix C/D，D-C：minADE +0.001804，minFDE +0.000992，JADE +0.001631，JFDE −0.003882m | 选择协议污染、marginal trade-off | original-protocol的混合效果 | 从A配对结果证明B必要 |
| 联合分布/推断一致 | 有限Gibbs定义CONFIRMED；算法同分布身份不成立 | sampler不是Gibbs/globalMAP | learned conditional compatibility的覆盖分配 | exact joint likelihood sampler |
| geometry6/MC有益 | HYPOTHESIS | 没有新训练/错误案例试验 | 可否证、预算受限设计 | 已提升、已发现新机制 |
| V4是新设计 | CONFIRMED（已有，非本轮新设计）；源码/既有ETH与UNIV报告均存在 | 新clean协议未复现 | 可复用完整轨迹耦合基线 | 本轮首次全轨迹关系或全图同步 |

历史 bank 原指标 .413185/.701908 与 post-fix C/D 的 C .413328/.702051 是不同实际执行记录，不拼接成同一次 paired run。原结果全部保留。五个推理 seed 不增加独立训练数；单一 raw source 不能产生可靠跨来源置信区间。

## 3. 五个优先瓶颈

| 排序 | 问题、证据与类型 | 状态 | 最小动作 / 新模块必要性 | 否证与失败动作 |
|---|---|---|---|---|
| 1 | 不可分解 pair contribution 未识别；固定bank只打散对应关系，评价解释 | OPEN | 对真实cost做单边/交互分解，配合energy-off、unary等预算；无需新网 | 若纯交互无益，撤回交互贡献，保留配对/coverage结论 |
| 2 | dataset_valid=test 139/139，源为biwi_eth；泛化证据 | CONFIRMED | raw-source分组的嵌套LOSO，含基座的provenance；无需新网 | 基座见过holdout则只报开发结果，不补“clean”标签 |
| 3 | mean-energy CE与expected conditional CE不同，硬/软邻居暴露差异 | 数学CONFIRMED，性能影响OPEN | A：S=4硬邻居MC，保留teacher梯度和其余loss；0新参数 | 不超过基线阈值则NO_CHANGE，不增加S/epochs补救 |
| 4 | goal几何看不到实际途中曲线、时序；但能量已读完整goal向量 | 输入边界CONFIRMED，任务损害OPEN | B仅添加直线路径最近距离/时刻；+64参数；有对应失败案例才启动 | surrogate误导或收益只来自marginal则放弃B |
| 5 | 端点推断与轨迹条件生成脱节，B correction收益混合 | 结构CONFIRMED，容量不足未证 | C复用V4 hard bank coupling；不用更大denoiser | bank oracle headroom小/soft-hard差大则停止，不能称坐标生成失败已解决 |

CUDA微差是复现解释的独立限制，不是创新不足或容量不足的证据。

## 4. 路线决策表

| 路线 | 理论针对性 | 新颖性/最近邻 | 结构与训练改动 | 资源 | 证据 | 决策 | 顺序 |
|---|---|---|---|---|---|---|---|
| A：MC conditional composite | 保留邻居不确定性经过normalizer的影响 | 概率目标修订，非首创；JFP是重要图模型近邻 | 331845注册参数不变；S4替换两套局部CE；重训A | frozen候选、edge chunk256；最多约4倍局部归约，不是全模型4倍 | 反例成立，性能未知 | CONDITIONAL_GO；先纯函数实现/CPU验收 | 主线1 |
| B：geometry6 | 候选直线途中接近的归纳偏置 | 已有TTC/V4路径几何；原创性弱 | 4→32改6→32；+64；其他模块不变；独立重训 | 几乎无参数成本，不代表零训练成本 | 未有对应失败案例 | 当前NO_GO；通过错误案例门槛后CONDITIONAL_GO | 3 |
| C：原V4 full-bank coupling | 明确有限marginal集合保护 | 就是仓库已有V4；相对JFP仍是约束与训练/推断区别 | GDTS冻；Ky=P20，GRU128，M4 rank8，4同步×8 Sinkhorn，hard Hungarian | active301998，注册390579；多项E×400×128激活 | 历史ETH可见增益、UNIV很小，非本轮验证 | 作为新创新NO_GO；作为备选复现CONDITIONAL_GO | 备选2 |

A训练改变目标但不自动保持trajectory marginal。C仅hard双射后精确保留每人的有限轨迹多重集合；不是保留整个连续分布的证明。C在固定bank不能创造缺失的合理轨迹。

## 5. 十个直接答复

1. **是否理论合理？** 有限分布可归一化；训练是带未来teacher的双局部surrogate；推断是injective局部配置，不是globalMAP；端点联合能诱导路径相关，但无条件于端点的显式路径互动保障。
2. **最可信/最弱？** 固定bank配对优势最可信；“显式joint概率被精确采样”“关系模式已具社会语义”“A与启用B相同”必须撤回。
3. **可能新增机制？** MC条件目标在本仓库是实质变化，但作为科学新颖性不足；真正可形成主线的是固定预算下分离纯交互与单人ranking的机制证据。V4不能重报为新机制。
4. **目标还是结构？** 先归因与目标。现有网络已有history、GRU、social message、候选条件关系和rank8能量，没有容量不足证据。
5. **具体网络？** A保留Social 4→128+单层GRU128+单层message，Unary64，relation4→32→16/M4，energy270→128→64/rank8，Kg21/P20，两轮原exact；训练S4 MC；GDTS完整冻结。
6. **模块？** 保留所有A层及共同学习teacher，不盲目detach；corrector冻结全零；scene latent继续禁用；diffusion不改。A需同协议重训，旧权重只可作为兼容性回归或明示warm start。
7. **最小单元？** 一个纯张量conditional-composite loss及双分支接线，新增参数0；主要新增开销在S×N×K局部条件和边gather/归约。
8. **如何归因？** 同bank/common noise、真正score干预、retrained no-energy、unary/physical/null relations、纯交互分解、objective×structure 2×2、clean-source协议，详见预注册。
9. **最可能三类失败？** a)原优势主要是unary/coverage→保留当前A、降级科学主张；b)MC方差/目标更保守致指标变差→NO_CHANGE；c)目标gap不是瓶颈、真实路径冲突占主导→C，但只有bank有headroom才继续。
10. **下一步单独授权？** 只实现loss纯函数+解析开关默认off+CPU合成测试，见伪代码接口；预算CPU≤1小时、GPU0、训练0，不修改canonical默认。不附带GPU诊断或新training许可。

## 6. 历史文档需要的更正（仅本附录，不改旧文件）

旧 THEORY/Final Specification/clean-design 的scene z、posterior scene GRU和采样scene mode是历史路线，不是strict_no_z。旧缓存设计“缓存social features”不能沿用到训练：当前cache拒绝trainable-network outputs。旧架构文档的“SDD正在运行”是历史状态，本轮SDD保持停止。旧A/B比较应改用identity修复后C/D解读。V4 ETH自动报告的recovery_ratio≈−64195.78及平均“max_component_size”等字段有口径警报，不据这些列宣称可靠恢复率或大component泛化。

## 7. 本轮实际工作与边界

读取源码/配置/指定历史报告与机器结果；CPU读取A/B checkpoint state、参数计数与零头检查；独立scalar数学反例/静态内存计数。STATIC_CHECKS脚本实际执行两次（第一次显示被截断，第二次完整捕获）；没有模型导入/forward、随机payload重放、solver或exact-margin操作。另以validate_artifacts.py检查JSON解析、逐层参数汇总、链接及源码树未变，记录见DOCUMENT_QA.json。所有新GPU调用、训练、完整A、encoder、message、sampler、diffusion调用均0；未干预/恢复SDD，未修改Skills/Apps、生产代码、配置、数据、checkpoint或旧结果。

nature-academic-search用于primary来源核验（专用MCP缺失时直读论文），nature-reviewer用于主张—证据/技术风险检查；没有多审稿人独立性声明。网页CVF访问部分403，M²Traj已通过官方PDF直接下载补齐方法核验；作者实现未找到/未核对的项明确标注，不当作不存在。无新benchmark。

本轮普通文档commit后发布；固定SHA由实际提交结果提供，不在文档内虚构自引用commit。
