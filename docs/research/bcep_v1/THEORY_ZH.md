# BCEP：预算感知覆盖终点提供器——候选设计，不是已验证成果

## 1. 需要解决的不同问题

BCEP 将两件事情分开：
(1) 从训练标注学出未来位置的条件概率分布；
(2) 在K=20的输出预算下，从该分布选择具有较低期望最近终点距离的代表集合。

使用连续条件终点分布、独立目标监督和分阶段冻结，避免直接沿用失败的八层共享编码器＋64×64密度＋有限K损失整体联合训练。这里的改变是研究假说，不能声称它已解释/解决过去训练漂移的因果机制。

整个新模块无需训练完整GDTS，也不从它提取学习权重；旧Provider只作受保护的历史基线。模型命名不是原创性的证明，采用的normalizing flow、子模选择及目标条件预测思想仍须引用。

## 2. 最小 pipeline

Observed histories + permitted semantic scene inputs
 -> independent temporal/social/scene context
 -> horizon-conditioned continuous endpoint density
 -> finite weighted support in world coordinates
 -> K-point coverage selection
 -> existing learned coupling
 -> existing six-layer fixed-endpoint conditional bridge.

所有进入provider的条件都来自观测期。训练GT只进入loss。推理不使用GT挑目标、过滤候选或排序。

### 2.1 可逆运动坐标

b_i(h)=last_position + h*delta_t*v_i，其中v_i仅由观测求得。
g=b_i(h)+R_i S_h r。

R_i由观测方向确定，低速时取单位阵；S_h是训练折估计的正尺度，设置数值下界。该式是坐标选择，不是限制终点位于恒速附近。模型残差可表达停止、转弯和反向。

### 2.2 条件连续密度

z~N(0,I2), r=f_theta(z; context_i,h)。

q_theta(g|context,h)=phi(z)/(|det J_f(z)|*|det(R_i S_h)|)。

先用小型条件rational-quadratic spline流作候选实现，不需要图像式U-Net输出终点热图，也不需要额外终点扩散反演链。normalizing flow提供可计算密度并不保证小数据泛化、地图合规或所有真实模态覆盖。

参考：Durkan et al., Neural Spline Flows, NeurIPS 2019：
https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html

### 2.3 多 horizon 辅助监督

选择训练协议中的几个未来时刻h，共享同一个horizon-conditioned provider，用相应真实位置做NLL监督。只在最终时刻输出终点候选。这样可利用中间位置监督，但不训练完整GDTS轨迹网络。

不同horizon的边缘密度没有自动构成一致的联合轨迹分布；完整联合轨迹仍由EBJD桥生成。是否改善必须通过同预算单horizon消融验证。

## 3. 为什么不继续把有限K覆盖当成概率学习目标

负对数似然的期望等于真实分布熵与KL(p||q)之和，适当条件下由真实分布最小化。实际有限模型与有限样本不因此保证校准。

参考：Gneiting & Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation, JASA 2007：
https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

以下是一个直接推导的反例，说明best-of-K覆盖风险与概率真实性不是同一目标：
真实Y∈{0,1}, p(Y=0)=a；预测从q独立抽K个样本，q(sample=0)=b，两位置距离1。

R_K(b)=a*(1-b)^K + (1-a)*b^K。
K>1且a∈(0,1)时，最优
b*=a^(1/(K-1))/(a^(1/(K-1))+(1-a)^(1/(K-1)))。

a=.9,K=20时，b*=.528878673...，而非.9。
这是因为有限集合覆盖两种未来，与按90%/10%的频率生成样本不同。K=2为该特殊二点模型中b*=a的特例；不能把公式误写成所有K均偏置。

NLL与覆盖项相加可产生权衡，但不能保证最优分布仍等于p。这不是证明过去的finite-K损失导致了训练漂移；它只是支持“概率估计”和“决策集合选择”分开设计。

## 4. 预算覆盖选择与子模性质

用模型产生的原子z_j及归一化非负权重w_j构成q_M。原子来自q的普通采样/QMC变换时使用等权，不再乘q(z_j)。

候选S取自有限支持C，|S|=K。
R_qM(S)=sum_j w_j min_{s∈S} ||z_j-s||_2。

令D覆盖所有原子-候选距离，例如C就是支持时D=max_{i,j}||z_i-z_j||。
F(S)=sum_j w_j max_{s∈S}[D-||z_j-s||]，F(empty)=0。
对非空S，F(S)=D-R_qM(S)。

每个原子对应的max-similarity函数具有边际收益递减，非负加权和仍然如此，故F是normalized monotone submodular。

在固定ground set、非负权重、纯cardinality约束、精确边际收益下：
F(S_greedy) >= [1-(1-1/K)^K] F(S_opt) >= (1-1/e)F(S_opt)。

参考：Nemhauser, Wolsey & Fisher (1978)：
https://thibaut.horel.org/submodularity/papers/nemhauser1978.pdf

边界：这是收益F的保证，不是minFDE降低63.2%，也不是R_greedy <=1.58 R_opt。由F=D-R得到的误差界含D，可能很松。额外硬约束、随机近似贪心、任意图距离修改都需重新说明前提。

选择器不将不同模态的中心求平均，因此不会因质心运算额外创造一个不在支持集里的候选；但若密度本身给了非法位置概率，选择器仍可能选到它。不能声称保证可走性。

## 5. 风险分解：为何必须先验收Provider

固定条件输入，假设p和q_M有有限一阶矩。对任意候选集合S，
l_S(y)=min_{s∈S}||y-s||_2
是1-Lipschitz。对任意连接p与q_M的耦合，
|E_p l_S - E_qM l_S| <= E||Y-Z||。
取耦合最小值得
|R_p(S)-R_qM(S)| <= W1(p,q_M)。

设Shat在同一个有限候选宇宙上相对经验最优的加法误差为epsilon_sel，S*_C是真风险在该宇宙中的最优，则三角分解给出：
R_p(Shat)-R_p(S*_C) <= 2 W1(p,q_M)+epsilon_sel。

进一步W1(p,q_M)<=W1(p,q_theta)+W1(q_theta,q_M)。
若与所有连续位置上的最优S*比较，还须加上：
epsilon_support=R_p(S*_C)-R_p(S*) >=0。

这将误差分为分布学习、数值支持逼近、候选限制和选点优化。p未知，不能将这些不可观测量直接伪装成实测证据；NLL小也不无条件推出W1小。

固定终点的EBJD桥无法补回未覆盖终点，因此阶段验收必须在桥前完成。只在可靠候选上训练/评估联合组织，才能分清贡献。

## 6. 真正解除GDTS训练依赖的条件

仅换Provider、不重训旧H/bridge，不足以证明完整系统独立：旧权重可能继承GDTS候选训练的影响。

最终无GDTS训练依赖流程应是：
1. 自有Provider随机初始化，仅用许可训练标注监督。
2. 冻结通过验收的Provider，生成训练候选，不读teacher bank。
3. H与六层桥随机初始化，在相同严格数据合同下训练。
4. 旧GDTS-derived版本只保留为已披露来源的对照。

这是分阶段从零训练，不是必须一次性端到端共同训练。是否端到端、使用何种地图预处理/语义先验，应如实描述。

## 7. 相关工作与创新边界

目标条件分解已有TNT、Y-Net、GDTS等工作；流模型也已用于轨迹密度学习，例如FlowChain。因此，不把“使用目标”“使用flow”“多个horizon”“贪心选点”本身称为新发现。

- TNT: https://proceedings.mlr.press/v155/zhao21b.html
- Y-Net: https://openaccess.thecvf.com/content/ICCV2021/html/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.html
- FlowChain: https://arxiv.org/abs/2308.08824
- GDTS: https://github.com/Winderting/GDTS

待验证贡献是：为固定终点联合桥构建可独立训练的连续终点先验；明确区分概率学习和有限预算覆盖；用受控实验说明覆盖候选与集合保持联合配对的协同价值。

既有方法组合不自动构成足够的论文创新。需要系统相关工作比较、同密度不同选择器消融、同数据不同Provider对照、独立训练seed以及完整下游联合指标。没有实验前不保证超越GDTS或达到某一期刊门槛。
