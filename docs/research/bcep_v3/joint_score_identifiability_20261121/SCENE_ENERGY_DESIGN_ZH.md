# 拟议结构：由同一场景能量生成候选替换评分

状态：数学设计与 CPU 验证性原型；**没有接入正式网络，没有训练，没有新模型指标**。这是本轮唯一建议进入下一阶段工程实现的方向。保留 final_coordinate，不通过温度、人工评分倍数或随机 epsilon 解决近并列。

## 1. 需要修复的结构问题

当前 S0 由场景注意力后的 world 向量、候选特征和行人上下文拼接进 MLP。实测 world 向量的差异远小于候选差异，完整评分的绝大部分又是对双射选择无效的行列偏置。旧反馈及 HCRA 分别输出接收者评分，再同时重新分配所有人；源码没有约束这些评分是同一个场景能量的条件变化，也没有保证各人的独立最优修正合在一起仍有益。

拟议模块把 world 表示为**当前确实被选中的各人候选集合**，不靠彼此几乎相同的查询向量独立命名 world。所有候选替换都向同一个能量询问代价；顺序接受的替换共享该能量。这样解决的是评分来源与协调约束，不能保证训练后 JADE/JFDE 改善，也不能保证 Sinkhorn 自动变成接近硬匹配。

## 2. 输入、形状与固定边缘

对一个场景省略 batch 维：N≤57，K=20；批处理时加 B 维并沿用 valid mask。

| 张量 | 形状 | 来源 |
|---|---|---|
| G | N×20×2 | 原 refined bank；只读、detach，不生成或移动候选 |
| U | N×20×192 | 原候选特征；可复用 CSEE 输出，不新增注意力层 |
| O | N×8×2 | 原观测历史与有效行人掩码 |
| P | N×20×20 | 每人的 world→candidate 双射或对应 ST 张量 |
| z | N×20×16 | 共享小型投影 φ(U, 观测运动、相对终点) 的候选潜变量 |
| R | E×20×20×12 | 原候选对参考路径/几何关系；**先在真实候选对上计算** |
| C | E×20×20 | 一个无向行人对共享的候选关系能量表 |

16 是拟议工程容量，不是本轮搜索得到的超参数。原型直接接收 z、C，没有实现或拟合 φ、ρ。关系来源适配现有 `geometry.reference_paths/pair_features`，不导入其他论文网络。

为明确接口，可取观测摘要 qᵢ=[Oᵢ,7−Oᵢ,4；Oᵢ,7−Oᵢ,0；Oᵢ,7−2Oᵢ,6+Oᵢ,5]/原 goal_scale，形状 N×6；候选偏移 gᵢₐ=(Gᵢₐ−Oᵢ,7)/同一尺度，形状 N×20×2。拟议 φ 为 Linear(200,64)→SiLU→Linear(64,16)，输入 [Uᵢₐ,qᵢ,gᵢₐ]；ρ 为 Linear(408,64)→SiLU→Linear(64,1)，输入 [Uᵢₐ,Uⱼᵦ,Rᵢⱼₐᵦ,qᵢ,qⱼ]。这些是结构设计值，没有在本轮权重或真实数据上拟合、筛选或验收。

观测图取原八邻居图的无向并集，保留边界距离严格并列的所有邻居。边 e=(i,j), i<j，只存一张表：

\[
C_{ij}(a,b)=\tfrac12[\rho(U_i(a),U_j(b),R_{ij}(a,b),O_i,O_j)
+\rho(U_j(b),U_i(a),R_{ji}(b,a),O_j,O_i)].
\]

两个方向共享同一个 ρ；反向查表严格为转置。原型用任意固定表验证此公共势的代数性质，**未声称已验收神经几何编码器的交换对称性**。φ、ρ 的上述拟议宽度、初始化及逐层计算成本须在下一阶段实现验收时冻结，不能把原型当作正式可训练入口。

## 3. 一个公共能量

令硬分配选出的潜变量为 xᵢʷ=zᵢ(πᵢ(w))，dᵢ 为无向图度数，ωᵢⱼ=1/√(dᵢdⱼ)。定义

\[
E_w(P)=\frac1{N-1}\sum_{i<j}\|x_i^w-x_j^w\|^2
+\sum_{(i,j)\in\mathcal E}\omega_{ij}C_{ij}(\pi_i(w),\pi_j(w)),\quad N>1.
\]

总场景能量 \(\mathcal E(P)=\sum_{w=1}^{20}E_w(P)\)。单人沿用原 identity bypass。

第一项让所有有效行人参与 world 相容性，使用完整图的低秩计算；第二项刻画局部候选关系。1/(N−1) 是每人的其他成员平均，ω 是同一无向势的度归一化，不是为制造尖锐 Sinkhorn 而任意放大评分。两项都属于评分结构，不向训练目标添加新损失或新损失权重。能量不承诺具有米单位；最终监督仍是原米单位 final_coordinate。

硬分配时第一项可精确写为

\[
E_w^{global}=\frac{N\sum_i\|x_i^w\|^2-\|\sum_i x_i^w\|^2}{N-1}.
\]

软/ST 扩展必须对**候选对代价**取期望：

\[
E_w^{global}(P)=\frac1{N-1}\sum_{i<j,a,b}P_{iwa}P_{jwb}\|z_i(a)-z_j(b)\|^2.
\]

不要将它误写成两个混合坐标距离。令 μᵢ=ΣₐPᵢₐzᵢ(a)、mᵢ=ΣₐPᵢₐ‖zᵢ(a)‖²，在行和为 1 的条件下，快速式为
\(\sum_i m_i-[\|\sum_i\mu_i\|^2-\sum_i\|\mu_i\|^2]/(N-1)\)。局部项同样使用 \(P_i C_{ij} P_j^\top\) 的逐 world 对角值。

## 4. 候选替换评分与可辨识性

固定其他人，输出 Sᵢ∈R²⁰ˣ²⁰：

\[
S_i(w,a)=-[E_w(P_{-i},i\leftarrow a)-E_w(P)].
\]

减去当前 E 只产生行常数，既便于核对，也不改变 Hungarian。每人完整候选集合在每个 world 各用一次，因此全体行偏置、列偏置都无法决定最优双射。

记 T₋ᵢʷ=Σⱼ≠ᵢzⱼ(πⱼ(w))，全球项的四点交互差为

\[
S_i(w,a)-S_i(w,a')-S_i(w',a)+S_i(w',a')
=\frac2{N-1}\langle z_i(a)-z_i(a'),T_{-i}^w-T_{-i}^{w'}\rangle
+\text{局部共享势的四点差}.
\]

只要候选差异和其他成员的 world 差异在潜空间有相互作用，就不等于单纯行列偏置。区别在于这项交互由实际候选集合产生，而不是要求一个通用拼接 MLP 自发找出微弱 world 差异。**这不是无条件不退化定理**：φ 坍缩、各 world 的邻居集合不可区分、关系势变成可加形式，都仍会退化；下一版必须继续记录有效谱、间隔、软硬偏离。

所有 world 同时重编号始终是能量的对称性；目标是识别跨人候选对应关系，而不是给 world 编号赋予物理意义。

## 5. 接原 Hungarian、Sinkhorn、final_coordinate

最小迁移设计保留当前初配作为 P⁽⁰⁾，新模块替代旧反馈与 HCRA 的两个独立评分残差；CSEE 候选特征、冻结 Provider/bank、六层桥、GRG-CTF 及有效损失不动。当前初配仍可能近并列，故 P⁽⁰⁾ 必须作为有物理候选键的显式 incumbent 传递，不宣称新反馈已经修好了首配。

第一版只进行一次有界顺序 sweep。每次固定其他人，为一位行人的 20×20 评分调用**原 SciPy Hungarian 算法**，并使用原 20 次 Sinkhorn 产生 ST 梯度。若同时独立更新所有人，公共能量也可能回升；顺序替换且只接受全能量严格下降，才得到原型已验证的单调性。它只针对学习到的能量，不是对真实 JFDE 的保证。

```text
P_hard, P_ST, P_soft = original_initial_assignment(S0)
for i in fixed_observation_member_order:  # 最多 N 次
    S_i = negative_conditional_energy_change(U, O, G, P_ST, i)
    proposal_hard = original_hungarian(S_i.detach())
    proposal_soft = original_sinkhorn(S_i, iterations=20)
    gap = complete_best_minus_second_best(S_i.detach())
    accept = (gap > 0) AND (common_energy(proposal_hard) < common_energy(P_hard))
    if accept.detach():
        P_hard[i] = proposal_hard
        P_soft[i] = proposal_soft
        P_ST[i] = proposal_hard + proposal_soft - proposal_soft.detach()
    else:
        keep incumbent hard / ST / soft together
goals_hard = original_candidates_gather(G, P_hard)
goals_ST = goals_hard + einsum(P_soft,G) - stopgrad(einsum(P_soft,G))
loss_pair = original_final_coordinate(goals_ST, truth, valid)
bridge = original_six_layer_bridge(goals_hard, original_keyed_noise)
```

伪代码中的神经/ST 路径尚未实现，不能把 CPU 硬能量原型等同于正式前后向验收。初配 ST 必须保留到条件能量，不可 detach 掉其他人的 P_ST；接受判定、访问顺序与硬选择才是离散控制。未来应在无 optimizer 更新验收中核对每条路径。

该接口保留了有偏的 coordinate ST，不宣称消除了 Hungarian 与 Sinkhorn 的数学差异。公共能量使条件评分有统一含义，后续仍需测量软硬差异、真实交换方向和参数梯度，不能先声称它解决了所有梯度问题。

## 6. 并列、编号与噪声

- gap 使用完整匹配的最优/次优差，不用逐行 top2 差。主评分、Hungarian、损失温度都不加 epsilon，不人工缩放。
- 严格并列时拒绝此次替换，保留 incumbent；若最优新方案有多个，即使都低于旧能量也保守弃权。精确相等不当作改善。
- 近并列只标记与报告，不改变评分或评测规则。原型使用 FP64 进行能量核对；未来 FP32 实现须记录求和误差与后验能量，不将微小差宣称为可靠效果。
- 候选编号变化必须同时映射 G/U/C、incumbent 和候选物理键；所有 world 统一换编号要同时映射所有人和噪声键。输出始终映射回原 bank 编号，不能重新生成 keyed noise。
- 在条件最优唯一且输入 incumbent 一并映射时，候选重编号与统一 world 重编号测试通过。原始 S0 严格并列产生的不同 incumbent、完全重复候选、行人访问顺序，以及浮点近并列都限制无条件等变性。**不承诺从任意重编号后重新求得的 S0 都返回相同最终组合**。
- 对单人仍直接旁路，不让求解器或原型改变其候选边缘。

## 7. 57 人成本边界

全局项每轮统计 Σx、Σ‖x‖²，候选替换的主要乘法为 O(NK²r)，不形成 N²K² 的全局神经关系。局部图通常约 O(8N) 条无向边；边界并列最坏 E=N(N−1)/2=1596。20×20 FP32 关系表最坏 2,553,600 字节；完整 12 维几何缓存约 30.64 MB，可按原 64 边分块生成；全部 N×K×K 评分约 91.2 KB。神经中间激活与反传内存另计。

ρ 的64边拼接输入为64×20×20×408，约41.78 MB，64维隐层约6.55 MB；两个方向可顺序计算。仅分块前向不代表反向已受控，正式实现应在关系构造函数内采用确定性激活重算，并测量其峰值与代价；本CPU原型没有替代这项GPU验收。

一次 sweep 最多 57 次主 Hungarian；保守的精确次优检查每人另 20 次，总计 1197 次 20×20 求解。这里明确披露这部分成本，不把其说成免费替换。CPU 原型的实际耗时、平台和缓存字节见 [PROTOTYPE_CHECK.json](PROTOTYPE_CHECK.json)，并未测量 φ/ρ 前后向或 GPU 训练速度。完整性能评估属于后续工程验收。

## 8. 与旧反馈和 HCRA 的实质差别

| 项目 | 旧反馈 / 当前 HCRA | 拟议公共能量 |
|---|---|---|
| world 条件 | S0 查询；初配候选引导的多个消息/残差 | 同一个 incumbent 的实际候选集合 |
| 接收者评分 | 各自的 MLP 兼容性与残差；无公共势约束 | 同一共享无向势和全场景潜能的精确候选替换差 |
| 多人协调 | 同时重求所有人的匹配 | 有界顺序替换，共享能量严格下降才接受 |
| 重复/冲突修正 | 两组残差可相互抵消或覆盖 | 一套条件能量；原型可验证各人对同一能量负责 |
| 软梯度 | 原 Sinkhorn + 坐标 ST | 保留接口及其偏差，不能将结构修复当作无偏梯度证明 |

原型证明了候选替换代数、低秩全局能量等价、双射边缘不变、顺序单调性、条件编号一致性、严格并列弃权及 57 人可容纳。它没有证明学习到的能量与真实人群意图一致，也没有评估新 JADE/JFDE。结论是值得做下一版正式实现与零更新验收，而不是现在有权启动下一次训练。
