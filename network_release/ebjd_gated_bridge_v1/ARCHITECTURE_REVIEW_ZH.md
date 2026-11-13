# EBJD 整体逻辑与创新评审

评审日期：2026-11-13。依据：实际源码、已有正式五折/碰撞结果、
20261109门控主干和加速度桥收据，以及下列一手论文。

**结论：核心方法逻辑成立，有明确的问题动机和联合预测证据，适合继续作为论文主线。
新版8层可作为受控研究候选；目前不足以替代已验证6层，更不能把门控、QK归一化或加深本身写成理论创新。**
本次整理不进一步堆叠结构，也不启动新的长期训练。

## 1. 主线为何合理

“预测每人的可靠终点”与“把哪些人的候选放在同一world”是不同问题。
先冻结已验证的有限终点集合，再通过双射学习联合组合，使联合分布变化和边缘终点覆盖
可分开检验。随后对该联合终点条件生成整场景轨迹，借助社交注意力和CTF处理路径交互。
这与JMM强调共同world的评价问题一致，但JMM已提出联合预测/联合损失，不能把问题定义算作首创。[1]

严格成立的结构性质是：双射保持逐人候选经验边缘；固定端点解码使G/H的minFDE相同；
对固定g，桥变量到前11帧的映射可逆。这些是可陈述、可证明的设计性质，
不是预测性能提升定理。更完整推导见METHOD.md。

## 2. 模块选择及创新边界

| 部分 | 逻辑评价 | 可以主张 | 不能主张 |
|---|---|---|---|
| 冻结原生终点先验 | 将已有可靠终点能力与联合组合分开 | 明确资产来源的模块化先验 | 独创目标引导；Goal-pretrain可任意替代matched权重 |
| 可学习双射联合配对 | 本系统最清晰的研究中心 | 保持有限边缘候选的场景条件联合组合 | 首创Sinkhorn/Hungarian；最优真实交互耦合 |
| GT一致条件桥 | 训练目标与条件端点一致，可精确约束输出端点 | 与固定终点配对组合的可逆路径参数化 | Schrödinger bridge；校准概率；自动提高minFDE |
| CTF未来几何/地图反馈 | 让精修看到预测路径相关的上下文，保持可微 | 明确的数据流与待消融的交互建模 | 自动保证避碰、物理可行、每个槽都有严格物理时间意义 |
| 终点条件门控DiT | 让不同终点、噪声和历史控制每类更新 | 针对条件联合桥的结构适配假设 | AdaLN/QK-RMS本身是新理论；8层必然胜6层 |
| 共享MSCE和一次训练 | 共用上下文、避免旧多阶段手工衔接 | 统一优化循环和共享表征 | 完全摆脱先验训练成本；路径损失直接训练了coupler |

目标条件的路径分解已有Y-Net等工作[2]，goal-guided diffusion已有GDTS[3]。
Sinkhorn学习排列已有系统研究[4]，DiT条件调制已有明确来源[5]。
因此最可信的论文表述是**“边缘候选保持的学习联合组合，与固定端点条件联合桥协同建模”**。
这种具体组合具有可研究价值；此次针对性检索不构成完整查新，不能声称首次。

## 3. 必须正面处理的逻辑限制

1. **配对监督与轨迹目标存在间隔。** `bridge_and_pair_loss`用GT终点训练桥，用冻结G训练配对。
   H没有送入训练桥路径loss，coupler主要优化joint endpoint surrogate。
   共享encoder不是“路径损失通过H训练配对”的证据。CR改善可作为实验观察，不能描述为碰撞目标的必然结果。
2. **终点训练/推理分布不同。** 训练桥见GT终点，推理见有限provider样本。
   条件生成以GT训练是合理选择，但离GT较远的条件可能形成外推；当前没有分层误差证据证明其已被解决。
3. **条件终点决定边缘FDE上限。** 同一候选集合下，换主干或配对不能降低G/H minFDE。
   ETH的端点差距无法通过8层路径主干自动消失。不能同时宣传“严格保持端点”和“靠桥降低同bank边缘FDE”。
4. **ST优化是代理。** Hungarian在固定分数上解线性分配；它不直接最小化场景碰撞。
   N=1硬前向固定identity但legacy软梯度仍可能存在，这是已识别的代理不一致；
   独立identity-backward选项尚未做长期精度验收，本目录为可归因保留legacy，不能隐藏该边界。
5. **白化槽不是物理时间槽。** CTF沿物理轨迹计算再按槽索引注入，是启发式设计；
   不等于严格坐标变换。已拒绝的decoder-energy结果说明“数学动机合理”仍不能替代试验。
6. **8层存在容量混杂。** 新门控6层比旧6层参数多23.9%，8层比旧6层多55.1%。
   legacy6→gated6只能隔离深度，不能证明等参数优势；gated6→gated8检验深度收益。
   同seed也不保证不同构造顺序下共同模块初值完全相同；未来需明确共同初始化或如实登记seed-matched。
7. **数据与统计边界。** 旧五折主表每折一个训练seed，五个推理seed仅反映采样波动；
   validation/test在既有协议中成员镜像，不能称独立测试泛化认证。
8. **运行成本不是总成本。** 在线不实例化GDTS轨迹网络成立；matched训练后的Goal权重来源仍应计入先验成本。
   cached-bank时延不等于完整在线预测时延。

## 4. 现有结果实际支持什么

以下数字来自原 `legacy6`，没有赋给新8层。

- ETH/UCY五折宏平均H相对matched GDTS：minADE改善0.97%、minFDE改善2.67%、
  JADE改善7.98%、JFDE改善5.68%。四项不退化合同通过UNIV/ZARA1/ZARA2，共3/5。
- HOTEL minADE退化12.45%，G已存在该问题；ETH minFDE退化0.004054m，G/H相同。
  这支持“路径条件建模和终点先验分别有折相关短板”，不能全部归咎于配对。
- H相对G的JADE/JFDE在五折每个推理seed均改善；minFDE保持。
- JMM CR五折宏平均H为CRJADE=7.1877%、CRmean=8.4957%；
  相对GDTS分别降低1.7391和1.8390个百分点，相对G分别降低1.4810和1.6496个百分点。
  五折均值方向一致，但UNIV H仍有约23.62% CRmean，远高于其GT参照约2.13%。
  不能称安全保证或无碰撞。
- 新主干已有130项CPU测试及真实SDD短更新/20×20采样接线检查。
  没有新8层长期训练指标、CUDA/BF16收益或最优层数证据。

来源：论文工作区
`docs/results/ebjd_eth_ucy_unified_fivefold_20261101/REPORT_ZH.md`、
`docs/results/ebjd_eth_ucy_jmm_collision_20261102/REPORT_ZH.md`、
`docs/research/20261109_endpoint_gated_dit_v1/{REPORT_ZH.md,SUMMARY.json}`。
它们保持原文件和原身份，本目录没有重复打包旧大报告/日志。

## 5. 本次整理解决的工程歧义

- 一个清楚的 `network.py` API，默认研究候选gated8；三种profile显式区分。
- 通用 `train.py`只承接统一入口，不再通过名称含ZARA1的历史pilot导入loader，
  也不拉入原无条件trainer和投影优化器。
- 发现归档schema的optimizer标签与当前统一AdamW行为不同，30轮LR也由入口固定。
  快照配置把有效AdamW、5轮warmup和10%最终LR写明；启动检查拒绝不一致，
  原历史配置/实验均保留。
- 源码文件哈希、来源提交、转换位置、依赖与恢复规则完整登记。
- scipy是实际必需依赖，已列入快照；在线provider的torchvision单列。

## 6. 明确决策

**保留核心统一方案；将8层作为未验收研究实现整理交付，不把它替换成论文已证实主模型。**
适合论文主张的中心是边缘保持的联合组合及条件交互建模，而非加深网络的先进性。
唯一优先后续：在同折、同初始化约束下完成legacy6/gated6/gated8的受控主干消融，
同时报告路径误差、联合误差、CR和成本，先确认新主干是否值得采用。
本次不自动启动该实验，也不把增加参数包装成必然改进。

## 一手参考与边界

[1] Weng et al., ICCV 2023, Joint Metrics Matter:
https://openaccess.thecvf.com/content/ICCV2023/papers/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.pdf

[2] Mangalam et al., ICCV 2021, From Goals, Waypoints & Paths to Long Term Human Trajectory Forecasting:
https://openaccess.thecvf.com/content/ICCV2021/html/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.html

[3] Sun et al., GDTS (arXiv 2023; IROS 2025): https://arxiv.org/abs/2311.14922

[4] Mena et al., ICLR 2018, Learning Latent Permutations with Gumbel-Sinkhorn Networks:
https://arxiv.org/abs/1802.08665
本实现未加入Gumbel随机采样，不称为原论文完整复现。

[5] Peebles and Xie, ICCV 2023, Scalable Diffusion Models with Transformers:
https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html

参考只支持已有组件/问题背景的来源，不证明EBJD组合新颖性、8层最优或本任务性能。
