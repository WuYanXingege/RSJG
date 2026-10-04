# 原始文献与新颖性对照

检索截止/实际读取日期：2026-10-04。nature-academic-search多源流程；本会话无arXiv/Crossref专用MCP，改用arXiv、CVF、PMLR及作者GitHub原始页面。按标题/arXiv ID去重，同一论文不同版本不计独立支持。不用博客/自动摘要判断方法；下表均标注可读原文位置。无任何外部训练、复现、排行榜比较。

## 1. 必选八篇与两个直接近邻

| 论文 / 版本、发表场合 / primary来源 | 实际方法、目标、joint层级 | 与RSJG区别及证据边界 |
|---|---|---|
| GDTS；Sun等；arXiv2311.14922v3，2025-03-03；作者repo标IROS2025。[原文§III](https://arxiv.org/html/2311.14922v3)；[官方代码](https://github.com/Winderting/GDTS) | Goal heatmap条件+历史encoder+ε-denoising；tree共用trunk再goal条件分支，BCE/noise损失。ETH/UCY、SDD、inD；原目标以marginal预测为主。 | RSJG冻结此基座，增加离散依赖层；树采样/goal-guided diffusion均继承，不能列新贡献。仓库当前实现190调用按源码核验，不能拿论文笼统DDIM步数代替。 |
| Joint Metrics Matter；Weng等；ICCV2023。[原文](https://arxiv.org/abs/2305.06292)；[作者代码/协议](https://github.com/ericaweng/joint-metrics-matter) | common sample的JADE/JFDE及碰撞评测，joint损失用于既有预测器；ETH/UCY与SDD。作者格式以8+12滑窗、最多20samples、共享frame/agent ID评测。 | RSJG的joint metrics及“联合优于单人指标解释”不新。JMM是评价/训练近邻，不等于RSJG覆盖assignment。其SDD版本/世界坐标转换不自动等于本仓库legacy17-scene协议。 |
| Joint Pedestrian Trajectory Prediction through Posterior Sampling；arXiv2404.00237v1（本轮以preprint身份，不从二手元数据确认会议）。[§III–V原文](https://arxiv.org/html/2404.00237v1) | GFTD对history+future完整轨迹的PCA latent联合扩散；GAT128×3，noise目标；推理history重建guidance，可加repeller和RePaint。ETH/UCY8/12、K20。 | 不是冻结GDTS候选重排，改变坐标、无有限marginal集合合同；是joint-path建模的直接近邻。论文有公式/伪代码排版不一致，本审查只采共同明确的方法描述，未获官方实现逐行复核。 |
| MotionDiffuser；Jiang等；CVPR2023，arXiv2306.03083v1。[§3原文](https://arxiv.org/html/2306.03083v1) | PCA轨迹空间，跨agent set self-attention+scene cross-attention；denoised-target L2、preconditioning，可微cost guidance；WOMD。 | 已有joint diffusion、未来interaction、constraint sampling。RSJG不同在冻基座/有限候选/覆盖合同，而非首次joint扩散。车辆地图信号和任务规模不同，不能与ETH数字排名。作者完整训练代码未在此次来源中核实。 |
| MotionLM；Seff等；ICCV2023，arXiv2309.16534v1。[§3原文](https://arxiv.org/html/2309.16534v1) | p(A₁:T|S)=∏_t∏_i p(a_it|A_<t,S)，离散motion token、teacher-forced token NLL；autoregressive联合rollout，后聚合代表模式；WOMD。 | 有显式时序条件、同t各agent条件独立；不是潜在关系mixture/finite candidate assignment。不应称其无条件同时建模所有agent动作，也不能将temporally causal直接解释观察性因果。未复现代码。 |
| Energy-based Potential Games for Joint Motion Forecasting and Control；Diehl等；CoRL2023，PMLR229。[原文§3–4](https://proceedings.mlr.press/v229/diehl23a/diehl23a.pdf)；[作者代码](https://github.com/rst-tu-dortmund/diff_epo_planner) | 神经网络预测初始化/目标/代价权重，unrolled可微优化potential game，连续控制产生轨迹；模拟与真实驾驶。 | pairwise能量、物理/目标代价、可微优化均已有。RSJG无potential game最优性/均衡证明，不能借其术语称当前同步assignment为博弈均衡。作者仓库存在，不等于本轮跑通。 |
| CODA：Diverse Yet Consistent…；Chu & Zhao；arXiv2605.22017v1，2026-05-21；**CVPR2026 Workshops/MEIS**，不是主会。[§3原文](https://arxiv.org/html/2605.22017v1)；[CVF正式记录](https://openaccess.thecvf.com/content/CVPR2026W/MEIS/html/Chu_Diverse_Yet_Consistent_Context-Guided_Diffusion_with_Energy-Based_Joint_Refinement_for_CVPRW_2026_paper.html) | DCGC历史局部/全局context；ACIM cross-attention注入denoising；JDR以diffusion密度×exp(−energy)修正；训练noise/regression/diversity/EBM；ETH/UCY、SDD、NBA、JRDB。 | 是“强marginal+energy joint refinement”直接近邻。所谓保个体合理性不是精确保轨迹集合。§3.4的energy最小化式未写完整normalizer/negative项；单看正样本energy和不够确立normalized MLE。不能据此宣布作者实现错误或RSJG胜出。未定位到可验证作者训练repo。 |
| M²Traj：A Unified Diffusion-Based Framework…；Yang等；WACV2026，pp6442起。[官方PDF§3–4](https://openaccess.thecvf.com/content/WACV2026/papers/Yang_A_Unified_Diffusion-Based_Framework_for_Multi-Agent_Trajectory_Prediction_Integrating_Structured_WACV_2026_paper.pdf) | agent/map/light tokens、history条件、DiT/self-conditioning；行为cost gradient注入reverse过程，speed/comfort/collision/map约束。§3给noise loss，§4还给均值/协方差分阶段训练；Waymo/HighD/MoCAD。 | 途中反馈/可学习物理guidance不是新方向；无需把其完整stack移植到ETH。论文标称contrastive的clean/noisy cost平方差单项容许constant解，需其他约束/实现判断；没有作者代码证实完整loss如何闭合。不要照抄不充分目标。 |
| 补充 JFP；Luo等；CoRL2022（PMLR205出版2023）。[§3原文](https://proceedings.mlr.press/v205/luo23a/luo23a.pdf) | marginal候选+unary/pairwise potentials+交互图；belief propagation计算/利用marginals，联合classification梯度与回归；WOMD及内部数据。 | **概率结构上最接近**。RSJG潜在relation混合、goal层、冻GDTS、injective exact局部分配为具体差异；不能主张首次在候选上建立pairwise joint模型。原文不是本仓库两轮sampler。 |
| 补充 QCNeXt；Zhou等；arXiv2306.10508v1技术报告、CVPR2023 AV挑战路线。[原文](https://arxiv.org/html/2306.10508v1) | query-centric场景编码，多agent DETR-like decoder显式建模未来互动，Argoverse2。 | 未来互动本身已有；任务/map/decoder和预算不同。仅作新颖性边界，不填未经同协议复现的baseline数字。 |

八篇要求均已获得primary方法证据。CVF HTML/PDF部分经网页工具403；M²Traj通过官方PDF直接下载到/tmp/rsjg_m2traj.pdf并pdftotext读取§3–4补齐，未提交论文PDF。默认shell代理不可解析，直接只读下载成功。其余主要用arXiv HTML/PMLR PDF；官方GDTS/JMM/EPO仓库页面核实可获得性，未clone/训练。GFTD/MotionDiffuser/MotionLM/CODA/M²Traj的“作者代码未核实”不是“代码不存在”。本文不声称穷尽2026年全部文献。

## 2. 最接近的机制矩阵（本审查推断，不是作者自述）

| 维度 | 当前A | JFP | CODA/GFTD | 本仓库V4 |
|---|---|---|---|---|
| 冻结强单人生成器 | 是GDTS | 端到端candidate/backbone训练路线 | 学joint diffusion/refinement，非同样冻结合同 | 默认Stage0 GDTS |
| interaction support | goal21×21 | 完整候选轨迹pair | noisy/完整连续path | fullpath20×20 |
| normalized模型/训练 | finite Gibbs定义，dual local surrogate | 图模型联合目标、message-passing近似 | diffusion目标+指导/refinement；具体normalization各异 | task/assignment surrogate，不宣称likelihood |
| 输出约束 | P20 distinct goal IDs，可更换被排除candidate | 原文推断，不是本repo P20injective合同 | 坐标可变 | hard每人全20轨迹双射，集合精确保持 |
| 可作本轮新主张 | zero-param MC目标只是训练方案；待实证 | 不能重报pairwise图模型首创 | 不能重报future social/energy diffusion首创 | 不能重报fullpath encoder/sync首创 |

最近邻排序取决于主张：finite energy→JFP；diffusion+joint refinement→CODA；完整路径生成→GFTD/MotionDiffuser；有限集合保护→仓库V4。不存在因任务换成人群或模块换名就自动获得新颖性的捷径。

## 3. 已有 / 有实质区别未验证 / 有证据区别

- **已有**：joint metrics、future interaction、稀疏图/GRU/message、low-rank pair factors、energy mixture、fullpath coupling、Sinkhorn/hard permutation、zero-head residual，均不能单独构成强原创主线。
- **实质区别但未验证**：在固定GDTS预算下，将单边ranking与不可约交互分离，并检验MC conditional目标的贡献；是可操作研究问题，不是已经获得的新机制。
- **有证据的区别（限定仓库合同）**：exact lexicographic per-agent coverage与literal-zero arithmetic identity已完成工程认证；旧bank的配对破坏有joint代价。没有证据证明这些合同优于所有最近邻或足以构成顶会创新。
- **B**：geometry6是轻量归纳偏置，已有V4/物理guidance可覆盖类似几何；单凭+64参数不够。
- **C**：承认复用V4，是fallback，不包装为“新统一框架”。

诚实研究主线：在强且冻结的marginal候选上，回答哪些不可分解的交互信息真正改善joint worlds，并给有限集合/coverage与marginal损失的可检验边界。要升级成方法贡献，必须通过预注册的因果干预、目标对照及新来源验证；不比较不同单位、不同K、不同map/预训练或不同test selection的表格数字。
