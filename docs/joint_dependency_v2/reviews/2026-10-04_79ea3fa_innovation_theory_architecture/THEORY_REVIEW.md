# 理论审查：定义、训练、算法与路径必须分开

源码：79ea3fa。数学检查为独立小标量例子，绝非读取真实exact payload或执行sampler。证据见[STATIC_CHECKS](STATIC_CHECKS.json)。

## 1. 实现公式与四种身份

固定候选G和历史X，合法离散状态s_i∈{valid candidates}：

\[
u_i(k)=\log p_{0i}(k)/\tau+\delta u_i(k),\quad
E^m_{ij}(k,l)=-A^m_{ij}(k)^\top B^m_{ij}(l)/\sqrt8,
\]
\[
p^m_{ij}(k,l)=\operatorname{softmax}_m
[b_{ij}+\langle W_{\rm geom}(\phi_{ij}(k,l)),Q_m\rangle/4+d_m],
\quad C_{ij}(k,l)=-\operatorname{LSE}_m(\log p^m_{ij}(k,l)-E^m_{ij}(k,l)).
\]
\[
p_\theta(s\mid X,G)=Z(X,G)^{-1}
 \exp\{\sum_i u_i(s_i)-\sum_{i<j:(i,j)\in E}C_{ij}(s_i,s_j)\}.
\]

规范边只计一次；只要每人非空有限support、有效分数有限，Z有限且正。λ_E当前隐式为1，不读取legacy energy_weight/pair_energy_normalization；不是配置的mean归约。mask项为−∞，不能含NaN；在canonical P20设置还需每人≥20个valid candidate。

| 对象 | 成立条件 / 精确身份 | 不成立的推论 | 修改必要性 |
|---|---|---|---|
| 有限Gibbs模型 | 对固定候选集合定义明确 | 连续轨迹normalized density已被拟合 | 无需为了可归一化改网络 |
| dual训练 | 条件logsoftmax有normalizer；soft邻居surrogate | full-joint MLE、missing partition因此必错 | 修正术语；A仅条件性改变surrogate |
| P个world分配 | without-replacement+局部exact injective assignment | Gibbs采样、global MAP、P独立同分布 | 不在本轮替换sampler |
| GDTS路径生成 | 端点条件的implicit stochastic algorithm | pθ(s)一定等于实际algorithm分布 | 明确分布/算法区分 |

candidate_log_prior是冻结heatmap候选打分，经候选生成/归一化形成离散prior；不是已校准的完整连续endpoint密度。最佳P排名、coverage及JADE都不证明校准/likelihood。

## 2. strict-no-z当前双目标和梯度

源码model.py:2757–2913、set_losses_coeffs:1518；joint_goal_loss.py:45,810,833,894。

q_i(k)∝1_valid exp(−‖g_i^k−y_i^*(T)‖²/(2σ²))，σ=1m（真实resolved args）；每人归一化；frozen候选/GT无梯度。令q^r_{ij}=future_teacher(descriptor6)，无scene latent。

prior使用p^m(k,l)，post将同一套E混合权重替为q^r(m)，不依赖候选：
\[
C^{post}_{ij}(k,l)=-\log\sum_m q^r_{ij}(m)e^{-E^m_{ij}(k,l)}.
\]
\[
\bar C_i^a(k)=\sum_{j\in N(i)}\sum_l q_j(l) C^a_{ij}(k,l),\quad
\ell_i^a=-\sum_kq_i(k)\log\operatorname{softmax}_k(u_i(k)-\bar C_i^a(k)).
\]
反向边用C(l,k)，不把有向edge重复计数。
\[
L_a={1\over B}\sum_b{1\over N_b}\sum_{i\in b}\ell_i^a,\quad
L_r={1\over E}\sum_{(i,j)}\sum_{k,l}q_i(k)q_j(l)
 \sum_m q^r_{ij}(m)(\log q^r_{ij}(m)-\log p^m_{ij}(k,l)).
\]
E=0时KL为可回传的0；不是除0。**L_a场景平衡，L_r全batch边平均，不是scene-balanced edge mean**。dense scenes会影响KL相对权重。
\[
L_A=.5L_{post}+.5L_{prior}+\beta(t)L_r,\quad
\beta(t)=.1\min(t/.2,1).
\]
t为已完成stage optimizer steps占预定总steps的进度，不是epoch13/当前best epoch；resume需保留progress。两CE包含局部logsumexp，未包含global log Z，因此不称joint MLE。

| 参数族 | post CE | prior CE | KL | 边界 |
|---|---|---|---|---|
| unary、social | 是；social经unary/energy | 是；也经relation | social经base relation | 不detach h |
| energy因子 | 是 | 是 | 否 | shared E，不能称两独立模型证据 |
| deployable base/dynamic relation | post混合不经它 | 是 | 是 | dynamic relation embedding64本阶段无loss路径 |
| future relation teacher | **是** | 否 | **q和logq两侧均可导** | future descriptor固定≠teacher冻结 |
| GDTS/候选/GT | 否 | 否 | 否 | frozen + detach |
| hard sampler/corrector | 不在A CE图中/冻结 | 同 | 同 | 不可对hard assignment偷偷straight-through |

teacher logq又做log_softmax不等于detach。未来输入提供训练期额外信息，但没有独立语义标签/固定teacher最优性的保证：q和p可共适应、共享E可退化。四模式任意同时置换embedding、query、teacher/base输出和能量模式，整体cost不变；label switching意味着不能把mode0叫“避让”当发现。单条GT对多种合理未来的识别本来不足；soft标签只缓解离散候选分配，不提供未观测未来的真实性标签。不要无理由固定随机teacher；若未来尝试stop-gradient必须定义可靠预训练teacher来源并另做消融，本方案不这样做。

## 3. mean-energy surrogate的精确差别与A设计

对hard邻居z_{−i}~∏q_j，定义
\[
L_{\rm EC}=\mathbb E_{z_{-i}}\left[-\sum_k q_i(k)\log p_\theta(k\mid z_{-i},X,G)\right].
\]
当前损失为CE(q_i,softmax(E_z[u−C]))，而不是E_z CE。因logsumexp凸、target线性，
\[
L_{\rm current}\le L_{\rm EC}.
\]
这是一种Jensen gap，不是缺失条件normalizer。邻居q为one-hot时相等并恢复普通categorical PL；更一般若不同z的logit向量只差候选无关常数也相等。不要把两个softmax的平均与上述loss期望混为一谈。

反例：两候选、target=(1,0)、等概率两个邻居状态，条件logits分别(0,0)、(−4,0)。当前CE=softplus(2)=2.126928；expected CE=.5[softplus(0)+softplus(4)]=2.355649，严格不等。不是实现bug的数值测试，是不同目标的数学例子。

**A的唯一目标改动**：每个训练scene采S=4组z_j~Categorical(q_j)，对每组hard邻居算完整logsoftmax条件CE再平均；同一组z用于post/prior配对；q_i仍soft，KL保持原精确candidate加权形式。新增参数0。不采model predictions，不调用sampler，不用GT测试期teacher；q固定所以MC梯度对L_EC无偏（随机数独立于参数），方差随S及邻居数变化。

它把邻居状态变为硬值并保留不确定性经normalizer的影响，**但分布仍是GT附近q而非部署时模型选择**，没有解决整个exposure gap，也没有使覆盖约束推断等价训练。更高loss不等于更好JADE；现目标可能是有益平滑。MC后teacher训练图不变。若改善不达预注册阈值，回到NO_CHANGE，不强加teacher冻结、gauge normalization或新λ。

## 4. 纯交互、低秩与可识别性

给定固定边cost矩阵C及预先声明的参考权重w_i,w_j（默认有效candidate均匀，不用GT），定义
r(k)=Σ_l w_j(l)C(k,l)，c(l)=Σ_k w_i(k)C(k,l)，μ=Σ_kw_i(k)r(k)，
\[
I(k,l)=C(k,l)-r(k)-c(l)+\mu.
\]
I行/列加权均值为0，C=I+a+b，取a=r−μ,b=c。将每条边的a/b从端点u扣除得到u'，则
Σu−ΣC = Σu'−ΣI。分布、exact条件primary目标只差邻居状态相关常数，数学上同一模型；finiteprecision/初始化/ties不因此逐位相同。

**仅把C替I且保持u不变会改变模型**，不是无害规范化；相反做完整gauge补偿不应期待真实分布性能提升。推荐只用于归因，不改变生产训练。energy-off去掉全部C，I-off保留a+b，unary-only仅u，三个干预不同；I-only且固定u必须标注同时删除单边项。判断单边项主导不能只看矩阵Frobenius norm，还要干预同一score路径。

尤其是当前Round0直接读u，补偿后u′会改变初始化，即使实数精确条件分布不变，有限两轮算法也可能给出不同worlds。E7的gauge负控必须固定同一Round0 IDs和persistent keys，仅从refinement入口比较补偿前后条件score；不能重新运行变动后的unary初始化再把差异判为代数失败。局部primary score的逐行常数平移不改变injective最优，但浮点/tie路径仍单列。

P=K完全置换且u仅候选相关时，各slot求和的单边项常数；canonical P20<K21时选哪个candidate被丢弃仍受单边项影响。改goal集合再diffuse不保持trajectory marginal；固定fullbank重排才有硬合同。

每个mode的E矩阵rank≤8，但候选相关p、非线性logsumexp之后C不再rank≤8。即rank1 E₁=−abᵀ和E₀=0的均匀mixture得到−log((1+exp(abᵀ))/2)，逐元素非线性一般增加矩阵rank。不能凭低秩因子维度宣称最终交互只能rank8，也不能据此证明容量足够。

对所有mode加同一candidate无关常数，只改normalizer；加不同mode常数则重新加权mixture，通常改变C的候选排序。若加candidate row/col项，需对应unary补偿才保持联合模型。relation概率可与energy offset相互补偿，语义更难识别。

u和C应视为无量纲log potentials；几何有m、m/s、m²，网络学尺度，不是物理能量。当前边求和使度大agent pair影响通常更强；degree-normalization改变模型、梯度与场景间尺度，不是数值等价优化。边权是否应用由真实JDV2路径而非legacy配置推断；canonicalweight=1使该区别在此不产生可识别缩放。

## 5. 同步exact为何不是全局推断

Round0独立Gumbel-Top-P给每agent不放回集合；Round1/2对冻结旧邻居，解一个P×Kg矩形injective assignment，优化(J,Cstay,Cgeom,Rpersistent)。CPU求解只保证**这一给定矩阵**的精确lex最优；不保证所有agent同时更新后的joint objective提高。第四层priority随机key跨两轮不变，非每轮Gumbel；也没有MH校正或遍历收敛保证。

最小两agent两candidate、P1、u0、cost [[0,1],[1,0]]：状态(0,1)→(1,0)→(0,1)，每次局部best response而joint cost始终1（全局0）。另一cost [[1,0],[0,2]]：(0,0)→(1,1)让cost1→2，双方在旧邻居下都改善，合起来变差。它否定通用同步单调性，不宣称本轮真实P20发生该例；多slot injectivity仍不补充全局单调证明。顺序coordinate descent对固定其他变量的同一objective才有局部单调性，也仍不保证全局MAP；Gibbs需要按真正条件概率抽样，非argmax assignment。

网络共享参数及对称聚合在实数数学下agent-equivariant，canonical edge reversal给cost转置；几何非旋转不变的MLP不能自动获得旋转等变。固定种子按agent/window编号派生payload：重新编号但不重排payload/key，不必逐位equivariant。图组件的Gibbs因子化与有限world列的对齐不同；packed batch RNG consumption可能影响sample path，不能从无跨边直接声明pack/unpack bitwise equality。边/候选mask、至少P有效值及E0旁路必须独立测试。

旧h试验只见同一选定h对的IDs稳定；固定h score差仍在。没有margin认证前不能给鲁棒半径；这里不需要新margin实验才能讨论上述数学身份。

## 6. goal联合与路径联合

理想化单world若采用pθ(s)并且给定X、goals的agent噪声独立，
p(Y|X,G)=Σ_s pθ(s|X,G)∏_i p_GDTS(Y_i|X_i,map,g_i^{s_i})。
当前更准确地应将pθ替成实际algorithm的Q_alg，且描述P-world集合分布：coverage使slots相关、tree共享同一agent的trunk latent，故P个world不iid。噪声batch有N轴不等于agent共享相同值；没有显式跨agent base denoiser attention。

相同goals可对应提前/推迟相遇、绕行方向不同；A在给定goals后不再看其他agent noisy path。B的corrector每步读relative noisy world位置/速度并交换边消息，确实增加路径耦合，但不是跨时间attention；cumsum提供时序前缀信息。投影删除component公共residual是一种经验no-harm归纳约束，不是守恒律。零均值、每人端点固定、每人完整轨迹多重集合不变三者不等价；后两者B都无保证。

C硬双射精确保留有限经验marginal与坐标端点，但不能突破bank支持；均匀有限集合coupling对罕见事件概率表达有P预算限制。single GT per scene仍不足以校准完整联合分布。pairwise potentials可通过图传播形成高阶统计依赖，但任意三元不可约约束（如三binary变量parity）不由一般仅pairwise log-potentials精确表示；当前数据尚未证明需要显式高阶网络。加入大Transformer不是该问题的证据。

## 7. 科学表述建议

“在冻结的goal-guided生成器和固定候选预算下，以候选条件潜在关系和稀疏低秩compatibility构造受覆盖约束的joint-world配置。对一个历史ETH bank，保持每人轨迹集合的配对破坏降低联合指标表现。学习到的不可分解交互贡献及独立来源泛化仍待受控验证。”

升级条件：干净来源/基座协议、真实energy/I/单边控制、等预算和完整轨迹基线、独立训练重复与source级不确定性同时支持；不能仅凭公式可归一化、Jensen反例或JADE提升宣布理论创新。
