# 最小实验预注册（计划，不执行）

日期2026-10-04；source79ea3fa。没有训练、评测、GPU峰值测量或CPU margin授权。本文件预先规定未来决策，不把设计阈值包装成已验证临床/物理阈值，也不把旧139窗用于新的最终模型选择。

## 1. 三个不同问题

Q1：任意cross-agent pairing是否超过随机打散？固定完整bank已给单sequence正证据，新增来源验证仍待做。
Q2：learned pair/relations是否超越unary、pair单边ranking、candidate coverage和简单物理？当前OPEN。
Q3：A目标/B结构是否超越当前A0及已有V4、可复现最近邻？当前OPEN。Q1不能替代Q2/Q3，Q2也不证明Stage-B有效。

## 2. 数据与冻结

历史回归集：ETH biwi_eth.txt的139个重叠窗×2035–2039，仅开发/回归。旧epoch13基于相同valid/test选择，不能靠新评测消除选择偏差。旧bank与其hash不变；oldbase上的对照只标original-protocol。

未来clean pilot只承诺**一个outer fold：HOTEL原始来源家族**。所有biwi_hotel及内容重复别名整体holdout，窗口必须在split后生成；其余来源中uni_examples.txt整文件为inner validation；训练排除这两类来源的全部agent/窗口和预训练标签。最终文件清单先按内容hash确认、冻结。若指定文件在本地不存在或别名无法消歧，停止数据阶段，不改用随机窗口划分。20步窗口不能跨raw source；相邻窗口有重叠，仅在同一partition内。

生成器也要采用这一协议训练/选择，或提供已有base从未使用两类holdout标签的可信provenance。本轮没有这样的clean base SHA，所以proposed base_checkpoint=null。外部地图/semantic segmentation预训练来源亦记录。旧checkpoint SHA699336…用于历史回归，不代替clean基座。

同一个clean GDTS基座供所有A0/A/B/AB/noenergy/C使用；三个Stage-A训练seed3101–3103是独立初始化/训练，仍条件于同一基座，不声称覆盖base训练方差。扩展到五fold/SDD不在预算内；一个fold正结果仍不能宣称跨数据集泛化。

候选G[N,21,2]与p0 frozen，原sample索引、图、GT/mask/source/frame/agent IDs、dtype、base/config/source/cacheSHA统一记录。noise采用同source/window/agent/branch/timestep索引的原语义payload；每个臂保存实际使用值或可回读摘要。不同调用顺序不能只依赖“同seed”声称common noise。固定完整bank的shuffle不重新diffuse；改变goal或retrain后Y会变化，必须报告marginal。

## 3. 有界对照矩阵

| ID | 对照 / 单一变化 | 控制与重训 | 评价与可否证门槛 |
|---|---|---|---|
| E1 pairing null | 完整轨迹每agent独立置换；common同置换负控；inverse还原 | 同一真实bank，100个固定置换seed625872300–399；无重训、无新生成 | 原multiset/逐人marginal精确相同；N1/common不变；报告all-active/mixed；只支持Q1 |
| E2 inference energy-off | 真正令每轮local pair cost=0，score=u；保留h/unary/Round0/覆盖/exact次数 | 不用energy_weight=0；原checkpoint，same G+noise，E2无重训 | 若learned不优于off且方向不稳定，Q2不成立；goal/轨迹集合可能变，不能声称marginal不变 |
| E3 retrained no-energy | 训练Cprior/Cpost置0，关闭无作用teacher/KL；同social/unary、预算 | 独立同seeds重训；报告主动少了pair学习路径/有效参数 | 区别E2部署分布外与E3最优noenergy。若无energy同样好，learned交互必要性降级 |
| E4 unary-only | 与E2相同有效score，同时区分“零energy但保留调用成本”与剪枝实现 | 主比较保留相同P/K、round、candidate约束及solver预算；无重训 | 不把删除expensive模块的加速误当表示收益；若仅coverage解释，保留coverage结论 |
| E5 物理cost | dmin直线候选路径；Cphys=max(0,1−dmin/0.4m)² | graph不改；η仅从训练集无GT score RMS匹配learned C的一次常数，无heldout tuning；无重训 | dmin的计算如B；单位无量纲，非保避碰；与learned同scale/coverage；若同样有效则不主张复杂relation必要 |
| E6 relation nulls | 同scene边内shuffle完整p(k,l,m)；随机固定simplex；uniform4 constant | 冻E/h/u/G；每臂在训练calibration cache匹配C的RMS尺度一次；标签不用；log归一化后混合，mode轴同约定 | shuffle跨edge而非仅mode同时重标；identity mode共同置换负控应数学不变；scale未对齐不得归因关系语义 |
| E7 单边/交互 | C=I+a+b；分别full / I-off(仅a+b) / C-off；完整gauge补偿负控（固定Round0 IDs，仅比较refinement；否则u′会改变初始化） | 参考weight均匀valid，不用GT；冻结其余；同G/noise | 看joint与marginal变化；仅norm大不证明贡献；gauge补偿只理论等价，GPUroundoff不冒称bitwise |
| E8 目标×结构 | A0 old objective/geom4；A MC/geom4；B old/geom6；AB MC/geom6 | 四臂同初始化共同旧列；新增两列zero；同训练和候选预算，three training seeds | A唯一主测试；B/AB只有预先观察内训错误案例后启用；否则记NOT_RUN，不移预算去加大A |
| E9 完整轨迹 | 同clean base raw vs原V4 C hard；另将A输出bank给同V4结构作为未来非核心扩展 | 核心C单独重训，Ky=P20；保证strict bijection；A+V4不在本次预算自动执行 | 区分goal改support与fullbank仅coupling；若raw bank oracle headroom<0.01m JADE，停止C efficacy claim |
| E10 corrector | 仅历史post-fix C/D引用；若未来有clean B需另授权 | 本轮及本预注册核心预算不扩展B训练 | 不用A配对收益作B证据；保留marginal trade-off |
| E11 nearest method | 首选JMM Joint AgentFormer同协议；JFP为机制对照，车辆baseline不硬迁移；GFTD/CODA按官方实现可用性列待复现 | 当前均未跑；author code、数据/单位/K/训练源均一致才比较 | 如作者代码无法核验，在资源表标UNREPRODUCED，不能用paper表数字填same-protocol基线 |
| E12 退化/一致性 | E0多人/N1/mixed/all-active、mask、断开组件、packing、candidateflip/agentflip | 先CPU合成合同；真实验证使用既有分桶，不为显著性挑窗口 | 违反跨scene隔离/coverage/strict集合合同直接停；数值微差不当效应解释 |

E2与E4有效score相同，是一致性/成本控制，不假装两个独立科学证据。E6随机关系使用固定master seed791004，softmax normal(0,1)后M维归一化；shuffle不跨scene/source，E≤1记无可交换边而不是随意跨窗。随机logits数量随边变，要存实际payload。统一训练calibration子集从训练raw sources各取首个完整window（按文件名/帧排序），不看指标；η=RMS(Clearned centered)/max(RMS(Cnull centered),1e−8)，全calibration聚合一次。constant uniform relation依然可经mode E产生candidate-dependent cost，不能称energy-off。

E7进一步报告各臂candidate被排除的第21项、ID slot变化、fullpath marginal。对fixed full-bank实施独立shuffle才有精确保有限集合结论；goal成本消融不能自动复用这一保证。

B门槛：仅training/inner-validation上预登记最多20个“scene-mean endpoint误差≤0.3m，但预测与GT边路径最近距离差>0.2m或最近时刻差>1.2s”的案例，按source/frame固定顺序取首20，不挑最大收益图；至少10例且横跨≥2 raw sources才启用B/AB。它只是结构必要性线索，失败后不得改阈值继续；旧139窗不用于选这20例。

## 4. 指标、统计与可被投机之处

Primary endpoint：同clean source协议下，A相对A0的scene-mean JADE下降。每window先在P20上对agent/time平均后取min；每training seed先平均5个inference seeds，再在source/block聚合。JFDE同一slot约束但endpoint；minADE/minFDE先每agent取min再平均。不把二者混成同一误差。

共同报告：marginal、JADE/JFDE、固定定义relative path error、JMM CRmean/CR-JADE、最近接近距离/时间差、goal endpoint、coverage20unique IDs、trajectory diversity与trajectory duplicate rate、solver/fullforward时间（未来获授权实测）。碰撞主阈值0.2m且报告连续线段最近距离；0.4m仅作为预声明敏感性，不事后挑阈值。物理energy的.4m安全缓冲≠碰撞判定半径。

CR-JADE使用first-index精确tie规则，并报告tie发生率/其他tie候选范围；用oracle选出安全轨迹会掩盖其他slot碰撞。JADE/JFDE可通过更广的单人support或样本数提高；coverage不保证质量；endpoint改善可能绕行/中途碰撞更差；diversity过高可只是在浪费预算。E0本身也不是统一负控（多人E0仍可改变joint pairing）。

分桶固定：activity=all-active/mixed/E0；N=1,2,3–5,6–10,>10；degree=0,1,2–4,≥5；component=1,2,3–5,6–10,>10；拥挤桶按observed min-distance<1m与否则分。各桶给窗/agent/source counts；不把桶内window当独立样本。

重叠窗处理：主要effect是paired per-source均值；若将来≥5独立raw sources可cluster bootstrap原始source 2000次（整source/全部seeds共同采样）。当前单outer HOTEL source可能只有1–2独立文件：**不伪造source总体CI**；另按原时间顺序做长度至少20帧的不重叠block，moving-block bootstrap只作该sequence条件区间并标依赖假设，不推跨source。三training seeds分别列，5inference seeds不是training重复。只有一个test来源时升级结论限定该新来源，不宣称普遍泛化；跨fold确认另行注册。

另需区分本仓库历史139窗与官方JMM ETH253窗/364 agent实例（src/jmm_protocol.py常量及pin）。使用JMM指标定义不等于采用官方JMM样本协议；nearest-method必须重新核对source/frame/agent集合，不直接把139窗结果与论文ETH(1.4)比较。

## 5. 成功、失败及停止

A最小有意义改善：JADE下降≥0.01m且≥2%相对A0，两者都满足；是与0.4s/米制误差、旧bank0.0339m效应量作量级参照的**设计门槛**，不是由新数据优化。三training seeds至少2/3同方向；若有足够source，paired source CI上界需<0；source不足则不给confirmatory泛化结论。

Guardrails：minADE恶化≤0.003m且≤1%，minFDE≤0.005m且≤1%；JFDE不恶化>0.01m；CRmean不增>0.005绝对比例；全agent20/20 ID覆盖不减、有限值/scene isolation通过。C additionally每agent轨迹多重集合inverse-gather byte exact，marginal数值差≤1e−6（归约浮点容差，不允许坐标扰动）。候选ID唯一不保证坐标唯一，二者分别报。

A没通过：拒绝“MC目标在此任务改善”的主张，保留合法surrogate/数学不等价结果，回到A0；不增加S、rank、epochs或换更好seed。若I-off无损：拒绝“纯交互是主要收益”，即使A总JADE变好也如此。B没通过：不再加geometry维度。C没通过：归因支持/soft-hard不足，不自动转joint diffusion或Stage-B。

选择：全部checkpoint由inner-validation JADE，JFDE再早epoch tie-break；guardrails预先应用；每epoch一次、40epoch cap、patience5、min_delta .001m。最终模型及analysis代码hash固定后才解封outer labels，一次正式5seed评测；失败的finaltest不能反馈调参。

## 6. 将来待授权预算（不是本轮资源使用）

最小下一实现只CPU≤1h、GPU0、训练0。

完整**单foldclean pilot上限**：单GPU204 GPU-hours，最长400 base epochs（150goal+250joint）并受96h硬cap；基座1次训练，A0/A/B/AB/noenergy/C各3training seeds最多18次、每次≤40epochs且≤4h，共≤72h；cache/fullbank/所有冻结干预/最终评估≤12h；最多一个nearest-method复现≤24h。总96+72+12+24=204h；是停止上限，不是运行时预测。基座阶段未在cap内达成可用合同则pilot停止，不拿半成品冒充clean baseline。B门槛失败就少跑6次，不把省下预算用于别的网络。

每种调用类建立attempted/completed/failed/unknown计数；一旦OOM/NaN/合同破坏停止该臂，记录失败，不无限重试。不得自动扩展sources、五fold、SDD、模型容量、训练轮数；若预算不足以完成预注册，报告INCOMPLETE并请求新授权，不能删掉不利对照保留正结果。

复现风险处理：保持既有dtype与RNG策略、不以切换确定性制造新方法；common payload和未观察输出同时留证；对已有score微差如未改变结论仅列限制。**CPU exact-payload margin审计不是这些创新试验的强制前置，也不在本预算中自动执行。**
