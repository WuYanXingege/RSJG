# EBJD clean v1 — complete frozen design

- Design version: `EBJD-clean-v1 / 2026-10-08`
- Frozen source document SHA256: `9296937cc81f0e64a94b7c774a76e04d489fffe76e8ee5b64ca9bda3ff3af1af`
- Implementation base commit: `092ed91f1f4117f50aa835ec8e47cd2c9ce74a98`
- Implementation branch: `research/ebjd-clean-v1`

The text below is the complete frozen source design. The metadata above is not
part of the hashed source document; no design section was removed.

**RSJG EBJD 一次性完整实现提示词：独立目录与新分支交付版**

请在 https://github.com/WuYanXingege/RSJG 中，一次性完整实现本提示词后附的 EBJD 网络、监督目标、可导采样、优化器、指标适配及必要消融开关。交付实际可运行的代码，而不只是规划、占位函数或伪代码。理论与网络规格以附后的设计正文为准；本段与正文第 17 节明确工程隔离方式，替代原先接入仓库公共 src/ 和 configs/ 的落地建议。

**必须执行的目录与分支约束**

1. 在仓库根目录新建 `ebjd_clean/`，作为这版干净设计的独立项目根目录。模型源码、数据适配、训练与评估入口、配置、必要验证、设计文档和运行说明均放在此目录。采用 `ebjd_clean/ebjd/` 作为独立 Python 包，通过包内显式导入运行，不依赖修改公共 src/、全局 PYTHONPATH 或旧 JDV2 的训练入口来启用新模型。
2. 在独立工作树或独立 checkout 中建立新分支 `research/ebjd-clean-v1`，目标远端为 `WuYanXingege/RSJG`。默认以已审查提交 `092ed91f1f4117f50aa835ec8e47cd2c9ce74a98` 为可复现的基准；读取仓库约束并记录实际 base SHA。若同名分支已用于其他工作，保留它并新建带明确后缀的分支，记录最终名称。不要在原工作目录切换分支而干扰现有任务。
3. 修改与新增的版本控制文件限定在 `ebjd_clean/`。保留仓库其他目录及既有实验结果；新目录自带 `.gitignore`、依赖说明、训练器和指标适配器。原设计中的公共 trainer、loader、U-Net 或 metrics 接口改动，应在新目录内部实现对应功能。
4. 数据、语义地图和已接受来源的初始化 checkpoint 可以通过显式路径只读引用。需要复用的最小 U-Net 或指标实现，复制到新目录的明确位置，记录源文件、源提交和许可；禁止复制整个 JDV2 项目形成新的隐式依赖。新模型不得通过旧 Stage-A/JDV2 分支、冻结设置、离散 goal bank 或后处理模块完成核心训练和采样。
5. 新训练输出统一写入 `ebjd_clean/outputs/<run_id>/`，每次运行有独立目录。日志、配置快照、权重和评价结果不得复用旧实验输出位置；缓存、数据、完整权重和大体积输出由目录内 `.gitignore` 排除，仅提交必要的小型验证摘要。
6. 保留完整设计正文为 `ebjd_clean/docs/DESIGN.md`，写明设计版本、内容 SHA256 和实施基准提交；实现差异、实际参数量、依赖版本及验证结果写入 `ebjd_clean/docs/IMPLEMENTATION_RECEIPT.md`。设计文档中的既有 NumPy 检查只属于设计代数核对，不能冒充本次真实 PyTorch 验证或训练结果。若发现设计与代码必须调整，记录具体原因及影响，不得静默删减网络模块、rollout 梯度链或优化器约束。
7. 完成设计验收中的真实 forward、backward、可导 rollout、指标与优化器验证，并给出可复现的训练/评估命令。正式多折训练如尚未执行，交付中应准确标明状态；不能宣称已提高 minADE、minFDE、JADE、JFDE。
8. 验证完成后，检查 Git diff，显式暂存本次 `ebjd_clean/` 内文件、提交，并将实际实现推送到 GitHub 新分支：`git push -u origin <实际新分支名>`。不覆盖远端既有历史，不把其他未提交改动带入本次提交，也不合并进其他分支。交付时提供实际分支链接、提交 SHA、独立目录路径、运行命令和已完成的验证摘要。只有推送成功后才能报告已推送；权限或连接阻断时准确报告已完成的本地状态及未完成的推送。

**监督与随机性必须按定稿实现**

- 监督来自同步场景的真实未来 12 帧轨迹，以及程序生成并保存的扩散噪声；未来真值仅进入 target/loss，不作为自由生成或推理条件。
- 对可逆终点—桥表示白化后的 latent 使用独立标准正态噪声 `epsilon ~ N(0, I)`；同场景共享噪声级 t，不共享相同噪声向量。独立性适用于白化 latent，解码后的桥噪声沿时间相关。
- pre/final 两类预测头均学习 `v* = alpha * epsilon - sigma * Z0`，v 不是行走速度。监督还包括正文的相对运动、地图热图及自由采样的边际/联合误差。
- 推理每个完整世界只在初始状态抽一次噪声，20 步 DDIM 使用 eta=0；世界之间不交叉 attention，不在每一步额外加入随机噪声。

以下为完整技术设计。与原定稿相比，只调整第 17 节的文件位置与入口组织；理论、网络、损失、采样、训练协议及验收标准保持一致。

---

**RSJG 完整网络设计：终点—桥联合扩散模型**

日期：2026-10-08  
设计代号：EBJD（Endpoint–Bridge Joint Diffusion，暂定工程名）  
依据：上一轮审查固定版本 research/joint-dependency-v2-clean / 092ed91f1f4117f50aa835ec8e47cd2c9ce74a98；官方 GDTS 对照 297d508558c10831983ea4b19c2b3e657459a449。  
性质：完整设计规格与算法伪代码；未接入仓库、未启动训练、未证实指标收益。本文给出一个确定的主模型，不要求先逐项试出若干残差补丁再决定最终结构。

**1．决策及研究主张**

选择一条主线：**可学习连续终点 + 端点一致的路径表示 + 同一世界的未来交互联合生成 + 自由采样下的边际/联合多目标优化**。

保留 GDTS 的语义地图 U-Net 作为可训练的条件编码器；替换原历史 context、时间去噪器和 tree sampling 主生成路径。目标不再由冻结热图中的 K21 离散候选决定，而是与整条未来路径一起扩散生成。未来社会信息直接参与去噪，能改变轨迹本身及终点支持。

“改动一步到位”在这里指：第一版就实现下文所有模块、训练目标、接口和消融开关，结构不再悬而未决。层数与超参数是一个可执行的初始设计，不是由理论唯一决定的最优值；收益和新颖性仍需要实验检验。已有 HOTEL 正式队列保留原配置并独立归档，不能把新架构混入旧 A0/A1 的结果。

论文主问题应改为：**如何在有限的 P20 场景预测预算下，同时学习个人未来覆盖与可共同发生的场景未来？** 可以争取的贡献是这种完整生成与优化机制及其证据，而不是单独宣称“Transformer、联合 diffusion、Brownian bridge 或梯度投影是新理论”。

| 理论环节 | 对应模型自由度 | 可验证的结果 |
|---|---|---|
| 联合误差 = 边际误差 + 世界协调缺口 | 同时优化覆盖与世界一致性 | 同权重 M、J、J−M 分开改善 |
| 完整轨迹可逆地分成终点与桥残差 | 连续终点扩散；完整 11 维时间残差 | 不受 K21 bank 限制，goal FDE = trajectory FDE |
| 邻居的 noisy future 可降低条件去噪风险 | 同世界跨 agent attention | 与纯时间模型相比，存在提升个人去噪的理论可能 |
| 高噪声几何不可靠 | v 参数化；预测 clean future；随噪声门控 | 高噪声几何贡献有界，机制由分噪声诊断检验 |
| 联合与边际目标可能冲突 | 自由采样指标损失；实际更新方向的两约束投影 | 约束更新的一阶边际代理风险不增 |
| 场景是无序的 agent 集合 | 共享网络、相对特征、明确场景/世界轴 | agent 置换等变、世界间不串线 |

最后一行是结构性质；“一阶不增”是局部训练代理目标的性质。二者都不等于测试 minADE/minFDE 永远不增。

**2．输入、输出和唯一主配置**

沿用 ETH/UCY 的 8→12 设置。不同 agent 必须来自同一个真实时间窗口，而不能将独立窗口拼成场景。

| 量 | 规格 |
|---|---|
| 观测 | X：[B,N,8,2]，米制坐标 |
| 地图输入 | 每个 agent 以最后观测位置为中心裁剪；6 语义通道 + 8 历史热图 |
| 裁剪 | 32 m × 32 m，256 × 256 像素，中心与范围只依赖观测及固定配置 |
| 训练标签 | Y*：[B,N,12,2]，只用于构造扩散 target 和 loss |
| 有效 mask | [B,N]；完整 8 步历史、12 步目标的 benchmark agent；padding 不参与注意力或均值 |
| 时间间隔 | dt = 0.4 s；预测跨度 4.8 s |
| 外部采样预算 | P = 20 个完整场景世界 |
| 输出 | Yhat：[B,P,N,12,2]；Ghat：[B,P,N,2] |
| 隐藏维度 | D = 128；4 heads；每 head 32 |
| 未来时空块 | 6 个：初步去噪 3 个 + 未来几何细化 3 个 |
| 推理采样 | 20 步确定性 DDIM；20 个独立初始场景噪声 |
| 训练噪声级 | k ∈ {1,…,100}，t=k/100；同场景所有 agent 使用同一个 t |

同一 p 是一个完整的共同未来，不是每个人各取第 p 大概率轨迹。不同 p 使用独立的初始噪声；同一 p 中各 agent 的初始高斯变量也独立，但后续去噪由场景网络耦合。共享噪声级不等于所有人使用同一个噪声向量。

地图 crop 不是生成边界，输出目标不做 map clipping；越出 crop 时可用地图信息减少，但连续目标仍然合法。使用原地图坐标到米制坐标的变换生成 crop 和标签，保证训练、推理一致。原 checkpoint 在这里是 CNN 初始化，不是固定目标候选的来源。

**3．从评价目标建立理论链条**

对一个场景及同一有效 agent 集，令 e_ip 为第 i 人、第 p 个世界的 ADE：

\[
M_A=\frac1N\sum_i\min_p e_{ip},\qquad
J_A=\min_p\frac1N\sum_i e_{ip}.
\]

FDE 同理定义 M_F、J_F。总有

\[
J_A=M_A+\Delta_A,\quad\Delta_A\ge0;\qquad
J_F=M_F+\Delta_F,\quad\Delta_F\ge0.
\]

个人候选集合固定时，单纯重排只能改变 Δ，不能改变 M。新模型直接学习终点和路径分布，可以改变 M；同世界社会去噪及联合损失针对 Δ。两部分需要同时训练，不能用 J 下降自动替代 M 下降。

训练使用可微 softmin：

\[
s_\tau(e_1,\ldots,e_S)
=-\tau\log\left(\frac1S\sum_{p=1}^S e^{-e_p/\tau}\right).
\]

它满足

\[
\min_p e_p\le s_\tau(e)\le\min_p e_p+\tau\log S.
\]

softmin 是凹函数，因此同权重、同样本下的 soft J 仍不小于 soft M。所有 gap 分析必须按 scene 内平均，再对 scene 平均；仓库正式 marginal 表的 agent weighting 和 joint 表的 scene weighting 仍分别保留，不能直接拿不同权重的汇总表相减。

这里不是优化“边际收益”这一经济概念，而是同时改善 marginal prediction metrics 和 joint prediction metrics。

**4．可逆终点—桥表示：把目标从条件改成生成变量**

取最后观测位置 x_i0。取最近两个观测速度的平均为 vbar_i，只由历史计算：

\[
b_i(\tau)=x_{i0}+\tau\,dt\,\bar v_i,\quad
a_\tau=\tau/T,\quad T=12.
\]

给定完整未来轨迹，定义

\[
G_i=Y_i(T),\quad d_i=G_i-b_i(T),
\]
\[
r_i(\tau)=Y_i(\tau)-b_i(\tau)-a_\tau d_i,\quad\tau=1,\ldots,11.
\]

令 r_i(0)=r_i(12)=0。反向解码：

\[
Y_i(\tau)=b_i(\tau)+a_\tau d_i+r_i(\tau),\quad\tau=1,\ldots,11,
\]
\[
Y_i(12)=G_i=b_i(12)+d_i.
\]

最后一帧在代码中直接拼接 G，避免靠浮点乘零实现终点约束。

原来每人 12×2=24 个实数；现在 2 个终点坐标 + 11×2=22 个桥残差，仍是 24 个实数。没有 PCA 截断、没有低秩路径限制，任意完整未来都能唯一表示。

因此严格成立：

\[
\operatorname{minFDE}(Yhat)=\operatorname{minFDE}(Ghat),
\quad
\operatorname{JFDE}(Yhat)=\operatorname{JFDE}(Ghat).
\]

在固定终点变量时，任意修改桥残差不影响最终位置：

\[
\frac{\partial Y_i(12)}{\partial r_i(\tau)}=0.
\]

这只是表示上的终点解耦。共享去噪器的权重或邻居状态改变时仍可能改变 G，不能据此宣称整个网络训练保证 FDE 不变。

**5．Brownian bridge 白化：设计噪声几何，不假设真实行为独立**

对 a_1,…,a_11 定义

\[
C_{\tau\nu}=\min(a_\tau,a_\nu)-a_\tau a_\nu,\qquad C=LL^\top.
\]

C 是 11×11 正定矩阵，L 为 Cholesky 因子。使用

\[
z_i^g=d_i/s_g,\qquad z_i^r=L^{-1}r_i/s_r.
\]

s_g、s_r 是每个训练折上计算的固定标量：

- s_g = 训练终点相对 CV 基线偏移的每坐标 RMS，最低 0.5 m；
- s_r = L^{-1}r 的每坐标 RMS，最低 0.1 m。

具体 RMS 均按训练折的全部有效 agent、两坐标计算；s_r 还平均 11 个时间系数。使用 FP64 计算统计量，固定后作为模型 buffer。验证和测试不参与统计，不使用测试未来进行归一化。由于都是标量，不人为赋予 x/y 不同缩放。

解码时

\[
G_i=b_i(T)+s_gz_i^g,\qquad r_i=s_rLz_i^r.
\]

等距网格下可直接推导：

\[
C^{-1}=T\,\operatorname{tridiag}(-1,2,-1),
\]
\[
\|z_i^r\|^2
=\frac{T}{s_r^2}\sum_{\tau=1}^{T}
\|r_i(\tau)-r_i(\tau-1)\|^2.
\]

于是各向同性 latent noise 对应一条端点为零的桥噪声，而非将任意 noisy velocity 积分成不受控制的几何。随机终点项和桥项对应的参考轨迹噪声协方差为

\[
\operatorname{Cov}[\eta_\tau,\eta_\nu]
=a_\tau a_\nu s_g^2I+s_r^2 C_{\tau\nu}I.
\]

这是用于扩散参数化的高斯参考结构，不是假设真实的 G 与 r 相互独立，更不是假设人的真实路径是 Brownian motion。模型学习的是完整的 p_theta(G_1:N,r_1:N|X,map)，其中终点、路径及不同人之间都可以相关。

桥白化是可逆线性变换，本身不增加模型表达能力；它提供端点分工、噪声尺度和可审计的结构。Brownian bridge 已用于其他 diffusion 工作，不能写成首次提出桥扩散。[P5]

**6．为什么联合去噪有可能改善个人精度**

令 V_i 为 diffusion 的 v target。对同一个观测条件 C_obs：

\[
\mathcal F_i=\sigma(Z_{t,i},C_{obs}),\quad
\mathcal F=\sigma(Z_{t,1:N},C_{obs}),\quad
\mathcal F_i\subseteq\mathcal F.
\]

在无限数据、可实现最优条件均值的比较中，

\[
\inf_f E\|V_i-f(\mathcal F)\|^2
\le
\inf_{f_i} E\|V_i-f_i(\mathcal F_i)\|^2.
\]

减少的风险等于

\[
E\|\ E[V_i|\mathcal F]-E[V_i|\mathcal F_i]\ \|^2.
\]

如果 noisy 邻居未来中存在与本人的未来相关的信息，联合网络就有可能降低个人去噪不确定性；这解释了“联合生成为何可能同时改善边际”，而不是把共同未来约束当成只能牺牲个人精度的惩罚。

它不证明有限容量的 D128/6-block 网络一定达到这个下界，也不把 noise MSE 优势直接等同于 best-of-20 ADE/FDE 优势。预测 clean future 是已有输入的确定函数，不新增信息；它是帮助有限网络读取几何的归纳偏置。

训练的 noisy future 由目标加噪构成，这是 diffusion 的正常学习过程。推理的 Z_t 来自模型的 reverse sampling，模型没有访问邻居真实未来。

**7．完整网络拓扑**

~~~mermaid
flowchart TD
  X["同步历史 X"] --> H["1 层 GRU + 2 个历史社会块"]
  M["语义地图 + 8 张历史热图"] --> U["可训练 U-Net"]
  U --> C["历史与地图条件 memory"]
  H --> C
  Z["终点与桥的 noisy latent"] --> D1["3 个初步时空块"]
  C --> D1
  D1 --> V0["初步终点头与桥头"]
  V0 --> G["重建 clean future 与门控几何"]
  D1 --> D2["3 个交互细化块"]
  G --> D2
  C --> D2
  U --> G
  D2 --> V["最终终点头与桥头"]
  V --> S["联合 DDIM 更新"]
  S --> Y["P20 完整场景轨迹"]
~~~

6 个块共同组成一个去噪器，每次 diffusion forward 都执行完整的 3+3 路径。前半部输出用于估计 clean future，后半部在该估计上进行社会交互；最终终点和桥输出更新同一个联合 latent。

**8．每个模块的网络、维度和层数**

全网络除保留的 U-Net 内使用 ReLU 外，MLP 使用 SiLU；Transformer FFN 使用 GELU。所有 dropout=0；小型 ETH/UCY 初版不加 agent ID embedding。所有 attention 均为 4 heads、D128、共享权重。

| 模块 | 确切结构 | 输出/说明 |
|---|---|---|
| 地图 U-Net | encoder 通道 14→32→32→64→64→64，5 个 DoubleConv；decoder 64→64→64→32→32，4 个 up block；每个 DoubleConv 两个 3×3 conv；12 通道 1×1 head | 保留原通道规格；新增返回最后 decoder 的 F32 和 logits12 |
| U-Net 深度 | 10 个 encoder conv + 4 个 up conv + 8 个 decoder conv + 1 个 head conv | 总计 23 个 conv；BN=false；ReLU；所有层训练 |
| 地图 memory | F32 adaptive average pool 到 8×8；Linear32→128；坐标 MLP2→64→128 | 每人 64 个 map tokens；坐标相对其 x0 |
| 历史输入 | 自身相对 x0 位置2 + 速度2 + 加速度2；Linear6→128→LayerNorm→SiLU | 位置按 s_g；速度/加速度按 1 m/s、1 m/s² |
| 历史时序 | 1 层 GRU，input128、hidden128，h0=0 | 保存 8 个历史 tokens，以及最后 h_i |
| 历史社会编码 | 2 个 pre-LN social Transformer blocks；每块 MHA128 + FF128→512→128；两处 LN | 对同一 scene 的 h_i 做跨人注意力 |
| observed attention bias | 历史相对几何10维；MLP10→64→4；输出 2 tanh(.) | 所有 observed/future blocks 共用该几何编码器 |
| 条件 memory | 8 个自身历史 tokens + 64 个地图 tokens；添加 history/map 两种 type embedding；共享 LN128 | [B,N,72,128]；历史固定 sinusoidal time embedding |
| noisy state stem | Linear2→128 + LN128；添加 bridge/goal 两种 type embedding、sinusoidal future-time embedding、h_i | 11 个 bridge tokens + 1 个 goal token |
| diffusion time | sinusoidal128→MLP128→256→128 | 输入 t；每个 scene 一份后 broadcast |
| 各未来块调制 | 拼接 time128 与 h_i128；独立 MLP256→128→256 | 输出 gamma、beta；用于该块四个 pre-LN 分支 |
| 初步去噪 | 3 个 factorized future blocks | 使用历史社会 bias；不读取 GT goal 或未来几何 |
| 初步输出 | 独立 goal MLP128→64→2；独立 bridge MLP128→64→2 | v_g_pre；v_r_pre，bridge head 在 11 tokens 间共享 |
| clean 状态注入 | 位置相对 CV2、速度相对 vbar2、终点偏移2；Linear6→128 + LN128 | 乘 gate(t) 后加入后半部 tokens |
| clean 地图采样 | 从 F32 在预测 clean 位置双线性采样32维 + in-crop flag1；Linear33→128 + LN128 | 乘 gate(t)；越界返回 zero feature 和 flag0 |
| predicted future bias | 同世界未来相对几何12维；MLP12→64→4；输出 2 tanh(.) | 再乘 gate(t)；没有离散关系类别 |
| 交互细化 | 3 个与前半部同规格的 factorized blocks | 使用 observed bias + gated future bias |
| 最终输出 | goal MLP128→64→2；bridge MLP128→64→2 | v_g、v_r；用于 DDIM |
| 参数初始化 | 新 GRU/attention/MLP 按标准初始化；block modulation 最后一层零初始化；v heads 最后一层小方差初始化 | modulation 初始为 identity；v heads 不做被冻结的恒零 corrector |

D128、4 heads、FF512 用来控制短序列和小数据的容量；6 个 future blocks 提供 3 个基础去噪块与 3 个交互细化块的完整路径。这里的“6 个块”不是 6 层单一 self-attention：每个块包含 3 次 attention 和 1 个 FFN，另有 2 个历史 attention blocks。层数是合理工程选型，需通过消融验证，不能写成理论推出必须为 6。

按表逐项计算的参数估算为 **3,612,764**，其中 U-Net 613,772；这是规格的解析计数，不是实际 PyTorch 实例化结果。正式实现应导出 named_parameters 与 active grad 列表核实。可训练参数减少不等于推理一定更快。

**9．时空块的计算顺序与轴**

主 hidden state：

\[
H\in R^{B\times P\times N\times12\times128}.
\]

每个块按以下顺序运行：

1. **时间 attention**：每个 scene/world/agent 单独看 12 tokens，包括终点 token。
2. **社会 attention**：同一 scene、同一 world、同一未来 token 位置上，跨 agent 看邻居。
3. **条件 cross-attention**：每个 agent 的 12 个 queries 读取其 72 个历史/地图 tokens。
4. **FFN**：128→512→128。

每个分支有独立 LN，其输出共同使用本块的 modulation：

\[
\operatorname{Mod}_\ell(q)=(1+\gamma_\ell(t,h_i))\odot q+\beta_\ell(t,h_i).
\]

分支采用 residual 更新，不在块外整体替换 H。主设计为同 scene 全连接社会 attention；历史邻居阈值仅决定辅助监督边，不限制未来生成器的消息范围。

| attention | 送入 MHA 的布局 |
|---|---|
| temporal | [B·P·N,12,128] |
| social | [B·P·12,N,128] |
| context query / key-value | [B·P·N,12,128] / [B·P·N,72,128] |

社会 attention bias 是 [B·P·12,4,N,N]。所有 padding key 被 mask，padding query 输出显式置零。每个 scene 至少有一个有效 agent，避免全 masked softmax 产生 NaN。

复杂度按每个 future block 的 attention interaction 计约为

\[
O(BPD[NT^2+TN^2+NT\cdot72]).
\]

线性投影与 FFN 另含 O(BPNTD²)。时间/社会分解减少了对 NT 全 token 做平方 attention 的开销；全连接社会部分仍为 N²，没有声称主配置是稀疏线性复杂度。

**10．预测干净未来与几何门控**

使用 VP cosine schedule：

\[
\alpha(t)=\cos(\pi t/2),\quad\sigma(t)=\sin(\pi t/2).
\]

t=1 在代码中显式设置 alpha=0、sigma=1。对 goal 和 bridge 一起加噪：

\[
Z_t=\alpha Z_0+\sigma\epsilon,\qquad
v^*=\alpha\epsilon-\sigma Z_0.
\]

前 3 个块得到 v_pre 后：

\[
\widehat Z_0^{pre}=\alpha Z_t-\sigma v_{pre}.
\]

将它解码为 clean future estimate Y_pre、G_pre，再计算相对位置、速度和相遇几何。v 参数化避免 epsilon→x0 转换中除以 alpha 的高噪声放大；它是现有 diffusion 参数化，不是本课题新提出。[P4]

门控固定为

\[
g(t)=\alpha^2(t)=\frac{\mathrm{SNR}}{1+\mathrm{SNR}}.
\]

它是噪声级门控，不是校准过的预测置信度。初始纯噪声时 g=0，后半块仍可通过历史社会条件耦合 agent；噪声减小时逐渐引入预测未来几何。

社会 attention：

\[
A^{(h)}_{ij,\tau}
=\frac{Q_{i,\tau}^{(h)}K_{j,\tau}^{(h)\top}}{\sqrt{32}}
+b^{obs,(h)}_{ij}
+g(t)b^{future,(h)}_{ij,\tau}.
\]

由于 b_future=2 tanh(MLP)，其直接 logit 贡献严格被 2g(t) 限制。这个界不保证 clean path 正确，只避免任意高噪声几何产生无限强的 bias。

10 维历史几何：相对位置2、相对速度2、距离1、closing rate1、CPA 时间1、CPA 距离1、速度方向 cosine1、两人速度方向有效 flag1。CPA 由历史速度计算，时间裁到 [0,4.8] s；静止速度不强行构造朝向。

12 维未来几何：相对 clean 位置2、相对 clean 速度2、距离1、closing rate1、相对生成目标2、目标距离1、未来时间比例1、CPA 时间1、CPA 距离1。米制位置按 s_g 缩放，速度按 1 m/s，时间按 4.8 s；数值特征 clip 到 [-10,10]。

所有 predicted future 信息来自同世界的 Y_pre/G_pre。两个 phase 之间的 clean 重建、连续几何、地图双线性采样保留梯度。几何在该次 forward 的后 3 个块共用，不为每层添加另一套迭代 teacher。

由于 raw x/y MLP 与 raster CNN 没有严格 SE(2) 等变设计，只宣称 agent 置换等变与一致坐标变换；旋转稳健性通过训练增广验证，不宣称数学上的旋转等变。


**11．损失：生成学习、几何学习与真实自由采样分开**

(a) 基础联合 diffusion loss

将 goal 与 bridge 各自按有效 agent/坐标求均值，bridge 再按 11 tokens 求均值：

\[
L_{diff}
=E\|v_g-v_g^*\|^2+E\|v_r-v_r^*\|^2
+0.25\left(E\|v_g^{pre}-v_g^*\|^2+
E\|v_r^{pre}-v_r^*\|^2\right).
\]

goal 与 bridge 两块各占相同标量权重，避免 endpoint 只占 12 tokens 中的 1/12。正的固定分块权重不改变理想条件均值解，但会改变有限容量训练的取舍。初步 heads 用独立监督，保证后半部读取的 clean estimate 有学习目标。

(b) 可解释几何辅助监督

监督边集合 E_obs 只由历史决定：最后距离≤6 m，或历史 CV 在未来 4.8 s 内的 CPA 距离≤2 m。社会 attention 仍覆盖全部 scene agents。

在 provisional 与 final 的 clean reconstruction 上，分别计算

\[
L_{geo}(Yhat)
=g(t)\ \operatorname{mean}_{(i,j)\in E_{obs},\tau}
\left[
h_1\left(\frac{(\widehat Y_i-\widehat Y_j)-(Y_i^*-Y_j^*)}{1m}\right)
+\frac12 h_1\left(\frac{(\widehat V_i-\widehat V_j)-(V_i^*-V_j^*)}{1m/s}\right)
\right].
\]

h_1 为逐坐标 smooth-L1，再对坐标平均。预测速度由相邻位置与 dt 计算，第一步使用最后观测位置。最终 L_geo 是 provisional/final 两项的均值。无有效边时该项为 0。

这直接监督相对运动场，避免把未识别的四种 latent labels 写成“避让/同行/超越”。不加入另一个可训练 future teacher，也不把 projected crossing 强行解释成所有行人的真实交互类别。该辅助任务不是独立原创理论。

(c) 地图辅助损失

U-Net 保留 12 个未来热图 logits，使用 BCEWithLogitsLoss，target 是以真实未来坐标为中心、标准差 0.5 m 的 Gaussian heatmap。只在训练中构造，按有效 agent、12 帧与像素求均值：

\[
L_{map}=\operatorname{BCEWithLogits}(logits,H^*).
\]

crop 外 target 对应为零图；连续生成目标不被强行投回 crop。地图热图只训练 condition encoder，没有 TTST/top-k/离散目标分配进入主生成链。

(d) 自由生成指标损失

真实自由采样从独立标准高斯开始，使用与推理一致的 20 步 DDIM，无 GT goal、GT path condition。训练每次采 S=4 个完整世界；正式评价 P=20。对采样结果定义

\[
e^A_{ip}=\frac1{12}\sum_\tau\|\widehat Y^p_i(\tau)-Y_i^*(\tau)\|,
\quad
e^F_{ip}=\|\widehat G^p_i-G_i^*\|.
\]

每个 scene 内：

\[
M_A^\tau=\frac1N\sum_i s_\tau(e^A_{i,:}),\quad
M_F^\tau=\frac1N\sum_i s_\tau(e^F_{i,:}),
\]
\[
J_A^\tau=s_\tau(\operatorname{mean}_i e^A_{i,:}),\quad
J_F^\tau=s_\tau(\operatorname{mean}_i e^F_{i,:}).
\]

ADE 和 FDE 各自有其 winner，不把两者强制用同一个 winner。J 的 softmin 始终在 agent 平均之后，M 则在 agent 平均之前。

\[
L_{roll}
=J_A^\tau+0.5J_F^\tau+
M_A^\tau+0.5M_F^\tau.
\]

以 1 m 为固定单位归一化上述标量。按分解可写成

\[
L_{roll}=2M_A^\tau+M_F^\tau+
\Delta_A^\tau+0.5\Delta_F^\tau.
\]

因此显式兼顾个人覆盖与世界协调。单纯加权仍可能产生梯度冲突，下节处理实际更新方向。

每 4 个 optimizer updates 做一次 rollout，并在该次将该项乘 4，以使平均权重与预设 lambda_roll 对齐：

\[
L=L_{diff}+0.05L_{geo}+0.10L_{map}
+\mathbf1_{roll}\ 4\lambda_{roll}L_{roll}.
\]

训练 S4 的风险不是正式 P20 风险，softmin 也不是 hard min。这里消除了 GT endpoint conditioning 与自由生成之间的结构差异，但没有声称训练与正式指标完全相同。最终判断只能来自 P20 的实际完整采样。

**12．边际约束：投影实际参数更新，不在 Adam 前做错误保证**

rollout updates 计算两条独立的边际梯度：

\[
g_A=\nabla_\theta M_A^\tau,\qquad
g_F=\nabla_\theta M_F^\tau.
\]

先以 total loss 梯度生成 AdamW 候选的实际参数减量 u；u 已包含当前每参数组学习率、Adam moments/preconditioning 和 weight decay。然后求：

\[
d^*=\arg\min_d\frac12\|d-u\|^2,
\qquad
g_A^\top d\ge0,\quad g_F^\top d\ge0.
\]
\[
\theta_{new}=\theta-d^*.
\]

仅两条约束，可枚举 0/1/2 条 active constraints，求最多 2×2 的 Gram system，无需大型 QP 库。约束梯度可各自单位归一化，半空间不变；近零梯度忽略。

如果直接投影原始梯度，然后再执行 AdamW，预条件和 decay 可能改变方向，不能使用这里的一阶结论。也不能在手动应用 d* 后再次 optimizer.step()。

若代理风险在当前邻域 L-smooth，则

\[
M(\theta-d^*)\le
M(\theta)-g_M^\top d^*+\frac{L}{2}\|d^*\|^2.
\]

约束保证一阶项不增加，并不消除二阶项、不保证有限步长下严格下降。不同目标完全冲突时，d=0 仍可行，可能出现更新受阻。

保护范围明确：

- 是当前训练 minibatch、当前 S4 噪声、当前 softmin 下的两项 marginal 风险；
- 只发生在 rollout optimizer updates；其他 diffusion updates 没有此保证；
- 不保护未见场景的 hard minADE/minFDE，也不自动优于原 GDTS；
- 不保证 joint 目标每次下降；
- 需监控投影比例、更新范数、冲突率及 P20 validation 四项指标。

梯度投影/约束优化有已有工作，例如 GEM；不能把这一通用数学工具本身作为原创算法。[P6] 此处的设计要点是约束**同一完整联合生成器的自由采样边际风险**，并约束 AdamW 的实际减量。

**13．一次确定的训练协议**

全部新结构从第一轮已接入，所有参数均 requires_grad=True。不再沿用 Stage-A 冻结 generator 的训练开关。

| 项目 | 唯一初版值 |
|---|---|
| epochs | 100 |
| batch | 4 个完整 scenes；gradient accumulation 4，effective 16 scenes |
| denoiser/history/context lr | 1e-4 |
| 初始化的 U-Net lr | 1e-5 |
| LR schedule | 前 5 epochs warm-up，之后 cosine 到各初始 lr 的 1% |
| AdamW | betas=(0.9,0.999)，eps=1e-8，weight_decay=1e-4；bias/LN 无 decay |
| total gradient clip | norm 5；保护用的 g_A/g_F 保留未 clip 值 |
| dropout | 0 |
| mixed precision | BF16 可用于 CNN/attention；几何、decode、loss、QP、optimizer state 使用 FP32 |
| epochs 1–20 | L_diff + 0.05L_geo + 0.10L_map；全部模块训练 |
| epochs 21–40 | lambda_roll 从 0.01 线性增加到 0.20；每 4 updates 一个 rollout update |
| epochs 41–100 | lambda_roll=0.20；保持完整目标 |
| softmin 温度 | epochs21–40 从 0.10 m 线性减到 0.05 m；之后 0.05 m |
| rollout | S4；20-step DDIM；所有采样步保留梯度 |
| gradient constraint | 每个 rollout optimizer update 保护 soft M_A、M_F |
| activation checkpointing | future blocks 与 rollout steps；use_reentrant=False |
| EMA | 本初版不使用，以免额外改变更新/选模解释 |
| 增广 | 同 scene 的轨迹与地图一起旋转；p=0.5 镜像；worlds 共用该次增广 |
| random seeds | 至少 3 个训练 seeds；正式推理 seeds 2035–2039 |

显存不足时减少 microbatch scenes 并相应增加 accumulation，effective batch16、网络结构和 P20 预算保持一致。不能为减少显存而静默删 agent、截断桥维度、减少正式世界数或 detach 采样步。

每个普通 diffusion update 每 scene 约 1 次 world-denoiser evaluation。每个 rollout update 另有 S4×20=80 次 world-denoiser evaluations，world 维可并行；平均每 optimizer update 约另增 20 次。Backward 和 checkpoint 重算进一步增加计算，不能将此训练方案称为与原 GDTS 等计算成本。正式训练前需要实际 profile，本文未测 GPU 显存或耗时。

学习率、loss weights、warm-up、S4 等均是可执行起点，没有声称唯一最优。先按完整规格建立一次主模型，再用预先定义的消融验证贡献，不继续无目标地枚举潜在类别数/MC 数。

**14．完整推理流程**

1. 只读同步历史与语义地图。
2. 运行一次可训练 history/map encoder，得到 h、memory、F32、CV baseline b。
3. 独立采 Z_1∼N(0,I)，形状 [B,20,N,12,2]；前11 tokens 为 bridge、最后1 token 为 goal。
4. 对 j=20,…,1，t=j/20、s=(j−1)/20，运行完整 6-block denoiser。
5. 使用 final heads：
\[
\widehat Z_0=\alpha(t)Z_t-\sigma(t)v_\theta,\quad
\widehat\epsilon=\sigma(t)Z_t+\alpha(t)v_\theta,
\]
\[
Z_s=\alpha(s)\widehat Z_0+\sigma(s)\widehat\epsilon.
\]
6. s=0 时 Z_0 即最终生成状态；decode 得到 G、Y，最后一帧直接使用 G。
7. 输出 P20 完整世界，不再按每个人排序，不再解 joint assignment，不再运行后置 residual corrector。

推理共 20 次包含 P20 的 batched denoiser forward，即每 scene 400 次 world-denoiser evaluations。原 GDTS tree 有 trunk 复用，因此即使新 hidden 维度更小，也不能依据参数量或 sampler steps 宣称推理一定更快。分别报告真实延迟、峰值显存、FLOPs 和生成预算。

P 个样本不带已校准的概率评分。当前主任务是 best-of-P marginal/joint 评价；若未来要求 Brier/top1/NLL，需另做概率评估设计，不从 best-of-20 好看推导概率更准确。

**15．PyTorch 风格伪代码**

以下为实现规格伪代码。masked attention、scene packing、map 坐标变换、AdamW proposal/state commit 等 helper 需按对应说明实现；并非已在本环境实例化的完整软件。

(a) 表示与网络

~~~python
class EndpointBridgeRepresentation:
    # L: Cholesky(C), C[j,k]=min(a[j],a[k])-a[j]*a[k]
    # sg/sr: training-fold constants; all decode geometry in FP32
    def encode_target(self, Y, baseline):
        G = Y[..., -1, :]                  # [B,N,2]
        d = G - baseline[..., -1, :]
        r = (Y[..., :11, :] - baseline[..., :11, :]
             - a[:11, None] * d[..., None, :])
        zg = d / sg
        zr = triangular_solve(L, r / sr)   # L @ zr = r/sr
        return cat([zr, zg[..., None, :]], dim=-2)

    def decode(self, Z, baseline):
        baseline = baseline[:, None, :, :, :]  # B,1,N,12,2 -> broadcast P
        zr, zg = Z[..., :11, :], Z[..., 11, :]
        d = sg * zg
        G = baseline[..., -1, :] + d
        r = sr * matmul_time(L, zr)
        Y_first = (baseline[..., :11, :]
                   + a[:11, None] * d[..., None, :] + r)
        Y = cat([Y_first, G[..., None, :]], dim=-2)
        return Y, G                       # literal equality Y[..., -1, :] == G


class FactorizedBlock:
    def forward(self, H, ctx, time_emb, bias):
        gamma, beta = modulation(time_and_agent_condition(
            time_emb, ctx.social_h          # explicitly -> B,1,N,1,256
        ))
        mod = lambda x: (1 + gamma) * x + beta

        # reshape temporal: B*P*N,12,D
        H = H + temporal_attn(mod(ln_t(H)))

        # reshape social: B*P*12,N,D; bias: B*P*12,4,N,N
        H = H + social_attn(
            mod(ln_s(H)), bias=bias, key_mask=ctx.agent_valid
        )

        # query B*P*N,12,D; memory B*P*N,72,D
        H = H + condition_cross_attn(
            mod(ln_c(H)), ctx.memory
        )
        H = H + ff_128_512_128(mod(ln_f(H)))
        return zero_padded_queries(H, ctx.agent_valid)


class JointEndpointBridgeDenoiser:
    def forward(self, Zt, t, ctx):
        # explicitly [B,1,1,1,1], including scalar t -> B scenes
        alpha, sigma = broadcast_noise_coefficients(t, Zt.shape[0])
        time_emb = time_mlp(sinusoidal(t, 128))
        H = (state_stem(Zt) + fixed_future_time_embedding
             + goal_or_bridge_type_embedding
             + ctx.social_h[:, None, :, None, :])

        for block in self.blocks[:3]:
            H = block(H, ctx, time_emb, bias=ctx.observed_bias)

        vpre = combine_heads(
            self.pre_bridge_head(H[..., :11, :]),
            self.pre_goal_head(H[..., 11, :])
        )
        Zclean_pre = alpha * Zt.float() - sigma * vpre.float()
        Ypre, Gpre = representation.decode(Zclean_pre, ctx.baseline)

        # no detach: final losses may update pre head via geometry
        gate = alpha.square()
        clean_features = relative_to_cv_features(Ypre, Gpre, ctx)  # 6
        map_features = bilinear_map_sample(ctx.F32, Ypre)          # 32+flag
        H = H + gate * (
            clean_injection(clean_features)
            + map_injection(map_features)
        )
        bias_future = 2 * tanh(future_bias_mlp(
            pair_geometry12(Ypre, Gpre, ctx)
        ))
        bias = combine_attention_bias(
            ctx.observed_bias, bias_future,
            gate_per_scene=alpha_scene(t).square()
        )  # observed B,4,N,N -> combined B,P,12,4,N,N

        for block in self.blocks[3:]:
            H = block(H, ctx, time_emb, bias=bias)

        vfinal = combine_heads(
            self.final_bridge_head(H[..., :11, :]),
            self.final_goal_head(H[..., 11, :])
        )
        return vfinal, vpre, Ypre, Gpre
~~~

alpha/sigma/gate、social_h、baseline、mask 等按 [B,P,N,time,feature] 明确 broadcast。不得依赖 PyTorch 的偶然 shape 广播，将 P 或 scene 误当作社会 attention 轴。

combine_heads(bridge, goal) 明确执行 cat([bridge, goal[...,None,:]], dim=-2)。time_and_agent_condition 将 time embedding 和 h_i 显式扩成 [B,1,N,1,128] 后拼接。combine_attention_bias 将历史 bias 扩成 [B,1,1,4,N,N]，future bias 为 [B,P,12,4,N,N]，gate 为 [B,1,1,1,1,1]，再相加并 reshape。history/map memory 在 world 维 expand 后使用，不跨 world 交换内容。

(b) 无 GT condition 的可微采样

~~~python
def differentiable_sample(model, ctx, initial_noise):
    Z = initial_noise                     # [B,S,N,12,2]
    for j in range(20, 0, -1):
        t, s = j / 20, (j - 1) / 20
        at, st = cosine_vp(t)
        av, sv = cosine_vp(s)
        v, _, _, _ = model.denoiser(Z, t, ctx)
        clean = at * Z.float() - st * v.float()
        eps = st * Z.float() + at * v.float()
        Z = av * clean + sv * eps
        # NEVER: Z = Z.detach() inside training rollout
    return model.representation.decode(Z, ctx.baseline)

@torch.no_grad()
def predict(model, X, maps, valid, generator):
    ctx = model.encoder(X, maps, valid)
    noise = randn([B,20,N,12,2], generator=generator)
    return differentiable_sample(model, ctx, noise)
~~~

@no_grad 只用于推理。训练调用同一 sampler 时保留整条反传链；checkpoint 是重算策略，不是 detach。

(c) 训练与实际更新投影

~~~python
for update, microbatches in enumerate(scene_loader):
    do_roll = epoch >= 21 and update % 4 == 0
    total_grad = zeros_like_parameters(model)
    marginal_A_grad = zeros_like_parameters(model)
    marginal_F_grad = zeros_like_parameters(model)

    for batch in microbatches:             # accumulate 4 scenes*4
        ctx = model.encoder(batch.X, batch.maps, batch.agent_valid)
        Z0 = model.representation.encode_target(
            batch.Y.detach(), ctx.baseline
        )[:, None]                         # [B,1,N,12,2]

        # one t per scene; shared by all agents
        t = randint(1, 101, [B]) / 100
        alpha, sigma = broadcast_noise_coefficients(t)
        eps = randn_like(Z0)
        Zt = alpha * Z0 + sigma * eps
        vtarget = alpha * eps - sigma * Z0

        v, vpre, Ypre, _ = model.denoiser(Zt, t, ctx)
        Zclean = alpha * Zt - sigma * v
        Yclean, _ = model.representation.decode(Zclean, ctx.baseline)

        Ldiff = goal_mse(v, vtarget) + bridge_mse(v, vtarget)
        Ldiff += 0.25 * (
            goal_mse(vpre, vtarget) + bridge_mse(vpre, vtarget)
        )
        geo_pre = relative_motion_huber_per_scene(Ypre, batch.Y, ctx)
        geo_final = relative_motion_huber_per_scene(Yclean, batch.Y, ctx)
        Lgeo = valid_scene_mean(
            alpha_scene(t).square() * (geo_pre + geo_final) / 2
        )
        Lmap = heatmap_bce(ctx.logits12, make_targets(batch.Y, ctx))
        Lbase = Ldiff + 0.05 * Lgeo + 0.10 * Lmap
        L = Lbase

        if do_roll:
            # only encoder context + independently drawn noise enter sampler
            noise = randn([B,4,N,12,2])
            Yfree, Gfree = differentiable_sample(model, ctx, noise)
            MA, MF, JA, JF = scene_soft_metrics(
                Yfree, Gfree, batch.Y, batch.metric_mask, temperature(epoch)
            )
            Lroll = JA + 0.5 * JF + MA + 0.5 * MF
            L = Lbase + 4 * rollout_weight(epoch) * Lroll

            marginal_A_grad += grad(MA, model.parameters(),
                                    retain_graph=True) / accumulation
            marginal_F_grad += grad(MF, model.parameters(),
                                    retain_graph=True) / accumulation

        total_grad += grad(L, model.parameters()) / accumulation

    # candidate is an actual parameter decrement, including group lr & decay
    candidate, pending_adam_state = optimizer.propose_adamw(
        parameters=model.parameters(),
        gradient=clip_global_norm(total_grad, 5)
    )
    if do_roll:
        decrement = project_two_halfspaces(
            candidate, marginal_A_grad, marginal_F_grad
        )
    else:
        decrement = candidate

    optimizer.commit_state(pending_adam_state)  # advance moments once
    subtract_from_parameters(model, decrement) # no second optimizer.step()
~~~

伪代码的 grad 返回与 parameters 顺序一致的 tensor list，无梯度参数置零。Lgeo 最终要按有效 scene 聚合为 scalar，不能保留 B 维就传入 grad。microbatch 中 forward 的参数不更新，先累计梯度再统一投影。

(d) 两约束 QP 的 active-set 解

~~~python
def project_two_halfspaces(u, gA, gF):
    A = stack_nonzero_unit_rows([gA, gF])
    if A.num_rows == 0:
        return u

    candidates = []
    for active_set in subsets_of_rows(A):   # {}, {A}, {F}, {A,F}
        if not active_set:
            d = u
        else:
            G = A[active_set]
            lam = pinv(G @ G.T) @ (-G @ u)
            if any(lam < -tolerance):
                continue
            d = u + G.T @ lam
        if all(A @ d >= -tolerance):
            candidates.append(d)

    candidates.append(zeros_like(u))       # feasible fallback; report if used
    return argmin(candidates, key=lambda d: squared_norm(d - u))
~~~

Gram matrix/constraint inner products 使用 FP64 累计，参数更新回 FP32；处理重复、零梯度与相反梯度。数值 tolerance 按单位化后的约束和 update norm 设置，不把严重违反约束的解当作成功。


**16．概率解释及可以严格写入论文的命题**

新模型是场景级生成分布：

\[
Z_1\sim N(0,I),\quad Z_0=F_\theta(Z_1;X,M),\quad
Y=D_X(Z_0).
\]

F_theta 是 20 步联合 sampler；D_X 是可逆终点—桥解码。这个采样映射定义一个归一化的生成分布（pushforward measure），不要求额外求离散 joint partition function。P20 是它的有限样本，不是分布只包含 20 种可能未来。

v 的理想最优条件均值对应

\[
\widehat\epsilon=\sigma Z_t+\alpha v_\theta,\qquad
s_\theta(Z_t,t|X,M)=-\widehat\epsilon/\sigma(t),\quad t>0.
\]

所有 agents 的 noisy state 都进入 v_theta，因此学习的是 joint 去噪，不再是固定个人生成器加事后 coupling。有限网络的向量场未必严格保守，不宣称任意 v_theta 就是某个已知标量 energy 的精确梯度；有限 20 步 DDIM 也不是精确数据分布或精确 NLL。

建议论文中清楚区分：

| 性质 | 可以证明的范围 |
|---|---|
| metric gap 非负 | 相同 scene、mask、P 和加权方式下，hard/soft M≤J |
| 表示可逆 | 完整 12 帧目标与 goal2+bridge22 一一对应，统计缩放非零 |
| 终点一致性 | Y_T=G；goal/trajectory minFDE、JFDE 相等 |
| 桥能量结构 | whitened bridge norm 对应端点为零的残差增量能量 |
| 条件信息风险 | 理想可实现条件均值下，增加 noisy 邻居信息不会提高最优 MSE |
| agent 置换等变 | shared 网络、相对特征、无 agent ID、mask/输入同步置换 |
| world 隔离 | 各 world 独立运行；没有 cross-world attention 或逐人重排 |
| 几何 bias 界 | 每个 head 的 predicted geometry logit 贡献≤2alpha² |
| 局部 marginal 保护 | 指定 rollout 更新的当前两项 soft marginal loss 的一阶项不增加 |

不能写成定理的结论：测试四项指标必降；个体每条轨迹误差不增；所有目标同时全局最优；训练不会模式集中；地图及碰撞一定物理合法；6 层比 4/8 层最优；已经达到某论文档次。

**17．独立目录内必须一次实现的接口与文件**

以仓库根目录的 `ebjd_clean/` 为唯一新增项目目录。以下文件均在新分支实现，不改动公共 src/ 或既有实验入口。路径相对于仓库根目录。

| 文件/接口 | 实现责任 |
|---|---|
| ebjd_clean/ebjd/__init__.py | 独立包入口 |
| ebjd_clean/ebjd/model.py | EBJD 顶层模型；encoder、denoiser、representation、sample 接口 |
| ebjd_clean/ebjd/representation.py | target encode / world decode；C、L、s_g/s_r buffers；literal goal endpoint |
| ebjd_clean/ebjd/encoders.py | U-Net decoder feature 与 logits；history GRU；2 个历史社会块；context memory |
| ebjd_clean/ebjd/denoiser.py | 6-block factorized attention；4 个 v heads；clean geometry 与 map sampling |
| ebjd_clean/ebjd/objectives.py | 分块 v-MSE、相对运动 Huber、BCE、soft M/J、mask 与 scene 权重 |
| ebjd_clean/ebjd/sampling.py | 训练/推理共享的 20-step DDIM；训练可导、推理 no_grad |
| ebjd_clean/ebjd/optimizer.py | AdamW actual-decrement proposal、两半空间投影、一次 state commit |
| ebjd_clean/ebjd/trainer.py | 独立训练器；新 loss、optimizer、日志和选模；不调用旧 Stage-A freeze setter |
| ebjd_clean/ebjd/data.py | 显式 synchronized_world 数据布局；同步 packing、米制 crop、mask；只读外部数据 |
| ebjd_clean/ebjd/metrics.py | 包内米制适配与原评价定义；保留 marginal/joint 原权重，额外导出 scene-weighted gap |
| ebjd_clean/ebjd/vendor/ | 必要的最小 U-Net/指标来源；记录源 SHA 和许可，不依赖旧模型的运行时逻辑 |
| ebjd_clean/ebjd/train.py、evaluate.py | 独立 CLI 入口；分别支持 python -m ebjd.train / python -m ebjd.evaluate |
| ebjd_clean/configs/endpoint_bridge_joint.yaml | 下一节完整主配置及显式输入/输出路径 |
| ebjd_clean/tests/ | 第 20 节要求的真实网络、梯度、采样、指标和优化器验证 |
| ebjd_clean/docs/DESIGN.md | 完整设计定稿、版本与 SHA256 |
| ebjd_clean/docs/IMPLEMENTATION_RECEIPT.md | 设计落实情况、来源、实际参数量与验证结果 |
| ebjd_clean/README.md、依赖文件、.gitignore | 独立安装/运行说明；排除缓存、数据、权重和大体积输出 |
| ebjd_clean/outputs/<run_id>/ | 隔离的新实验输出；不纳入 Git 大文件提交 |

安装/运行命令必须由实际依赖文件与入口验证后写入 README；推荐在 `ebjd_clean/` 下安装独立包，再使用包内 CLI。包内核心源码不引用旧 Stage-A/JDV2 的训练器、模型类或隐式全局配置。

需要从新模型运行路径消除的旧逻辑：

- 不调用 _configure_training_stage 的 GDTS 冻结逻辑；
- 不调用 _jdv2_contexts 中的 goal.detach，连续 G 本身就是生成 latent；
- 不经过 hard TTST/top-k、K21→P20 solver、Gumbel assignment；
- 不加载 FutureRelationTeacher、M4 latent relation、rank8 energy、DependencyCorrector；
- 不通过 get_loss(joint_finetune) 转回旧 _jdv2_dependency_losses；
- 不对可训练 UNet features/logits 做跨 update 的静态缓存；
- 不在训练 rollout 的 reverse steps 用 no_grad/detach；
- 不把新输出再次当 noisy velocity 积分，也不乘两遍 dt/down_factor。

静态语义地图可缓存；CNN feature 必须用当前权重在线计算。当前 batch 的 context 在 20 个采样步间复用并保留计算图，参数 update 后重新计算。

独立 metrics 适配保留原评价定义：

1. 将有效 scenes 的 agent 顺序打包为 A，保留 scene_index[A]；
2. 重复观测 8 帧，拼接生成未来 12 帧；
3. 转成与原 src/metrics.py 定义一致的 [P,20,A,2]；GT 为 [20,A,2]；计算在新目录内部完成，不修改原文件；
4. 两者均为 metres，obs_length=8；
5. 原 GDTS 对照使用其既有像素→米转换；EBJD 已输出米制，不再经过 _future_predictions_world 的 down_factor/坐标变换；
6. marginal 继续按 agent 返回，joint 继续按 scene 返回，另导出同 scene 权重的 gap 诊断。

导出必须标记 coord_system=world、unit=m。包内 loader 明确选同步窗口，不靠 jdv2_active:null 推断类型。新入口、新输出目录和新分支共同隔离新实验；既有正式评估继续使用既有代码与配置。

**18．完整配置草案**

~~~yaml
model_type: endpoint_bridge_joint_diffusion
data_layout: synchronized_world
coordinate_system: world
coordinate_unit: m

data:
  obs_length: 8
  pred_length: 12
  dt_seconds: 0.4
  complete_horizon_required: true
  scene_agent_padding: true
  do_not_truncate_scene_agents: true

map_encoder:
  semantic_channels: 6
  history_heatmap_channels: 8
  crop_width_m: 32.0
  crop_pixels: 256
  enc_channels: [14, 32, 32, 64, 64, 64]
  dec_channels: [64, 64, 64, 32, 32]
  future_heatmap_channels: 12
  heatmap_sigma_m: 0.5
  batch_norm: false
  memory_grid: [8, 8]
  decoder_feature_dim: 32
  trainable: true
  initialization: accepted_fold_matched_gdts_unet

history_encoder:
  input_dim: 6
  gru_layers: 1
  hidden_dim: 128
  social_blocks: 2
  social_heads: 4
  social_ff_dim: 512
  observed_geometry_dim: 10
  observed_bias_bound: 2.0
  dropout: 0.0

representation:
  type: invertible_endpoint_brownian_bridge
  goal_tokens: 1
  bridge_tokens: 11
  bridge_cholesky: true
  retain_all_coefficients: true
  train_fold_rms_scaling: true
  goal_rms_floor_m: 0.5
  bridge_rms_floor_m: 0.1
  literal_endpoint_concatenation: true

denoiser:
  hidden_dim: 128
  heads: 4
  ff_dim: 512
  coarse_blocks: 3
  refinement_blocks: 3
  sublayers: [temporal_attention, social_attention, context_cross_attention, ffn]
  social_scope: all_valid_agents_in_same_scene_and_world
  memory_tokens_per_agent: 72
  agent_id_embedding: false
  cross_world_attention: false
  dropout: 0.0
  target_parameterization: v
  coarse_heads: [bridge, goal]
  final_heads: [bridge, goal]
  output_head_width: 64
  future_geometry_dim: 12
  future_bias_bound: 2.0
  geometry_gate: alpha_squared
  clean_future_detach: false
  predicted_position_map_sampling: true

diffusion:
  schedule: cosine_vp_alpha_cos_sigma_sin
  exact_terminal_snr_zero: true
  training_noise_bins: 100
  shared_time_within_scene: true
  independent_initial_noise_per_agent: true
  sampler: ddim
  ddim_eta: 0.0
  inference_steps: 20
  output_worlds: 20

objectives:
  goal_v_weight: 1.0
  bridge_v_weight: 1.0
  coarse_v_weight: 0.25
  relative_motion_weight: 0.05
  map_bce_weight: 0.10
  relative_velocity_factor: 0.5
  rollout_joint_ade_weight: 1.0
  rollout_joint_fde_weight: 0.5
  rollout_marginal_ade_weight: 1.0
  rollout_marginal_fde_weight: 0.5
  supervision_edge_radius_m: 6.0
  supervision_edge_cv_cpa_radius_m: 2.0
  supervision_edge_cv_horizon_s: 4.8

training:
  epochs: 100
  batch_scenes: 4
  gradient_accumulation: 4
  new_module_lr: 0.0001
  unet_lr: 0.00001
  lr_warmup_epochs: 5
  lr_final_fraction: 0.01
  optimizer: adamw_actual_step_projection
  optimizer_betas: [0.9, 0.999]
  optimizer_eps: 0.00000001
  weight_decay: 0.0001
  exclude_bias_and_layernorm_from_decay: true
  total_gradient_clip_norm: 5.0
  gradient_constraint_uses_unclipped_marginal_gradients: true
  base_training_epochs: 20
  rollout_start_epoch: 21
  rollout_ramp_end_epoch: 40
  rollout_weight_start: 0.01
  rollout_weight_end: 0.20
  rollout_every_optimizer_updates: 4
  rollout_frequency_loss_compensation: 4.0
  rollout_worlds: 4
  rollout_steps: 20
  rollout_detach_between_steps: false
  softmin_temperature_start_m: 0.10
  softmin_temperature_end_m: 0.05
  marginal_constraint: actual_update_two_halfspaces
  protected_losses: [soft_marginal_ade, soft_marginal_fde]
  activation_checkpointing: true
  checkpoint_use_reentrant: false
  mixed_precision: bf16_with_fp32_geometry_loss_optimizer
  ema: false
  train_all_modules: true
  augmentation: synchronized_rotation_and_reflection

legacy_paths:
  frozen_generator: false
  discrete_goal_bank: false
  hard_joint_assignment: false
  future_teacher: false
  latent_relation_classes: false
  energy_rank_factors: false
  dependency_corrector: false
  zero_mean_residual_projection: false
  tree_sampling: false

evaluation:
  worlds: 20
  steps: 20
  inference_seeds: [2035, 2036, 2037, 2038, 2039]
  minimum_training_seeds: 3
  metrics: [minADE, minFDE, JADE, JFDE]
  diagnostics: [scene_weighted_marginals, coordination_gap, CRmean, CRJADE, CRJFDE]
  per_agent_reordering: false
  final_test_used_for_checkpoint_selection: false
~~~

配置中的 accepted_fold_matched_gdts_unet 需要在实现时解析为现有已接受来源、训练折一致的 checkpoint 路径，并写入 receipt；不是本次新建的权重文件。所有模块 trainable，source 仅负责初始化。现有先验来源决策沿用此前约定。

**19．一次实现的消融与可证伪标准**

主模型一次建齐，以下是预定义的证据设计，不是要求做一轮改动才允许实现下一轮。

| 对照/消融 | 目的 |
|---|---|
| 原 GDTS | 完整保留原方法，输出同 P20，同同步窗口/metric masks |
| GDTS end-to-end fine-tune | 排除“只是解冻/再训练”的收益；报告 step-matched 与 compute-matched 预算 |
| EBJD，去掉 future social attention | 同历史/地图条件下，检验 future joint 去噪本身 |
| EBJD，未来几何改用 decode(Z_t) | 检验 predicted clean geometry 对原 noisy geometry 的价值；容量尽量匹配 |
| EBJD，Cartesian/velocity 表示 | 相同网络容量与采样预算，检验 endpoint/bridge 表示作用 |
| EBJD，去掉 geometry loss | 检验可解释相对运动监督是否有增量 |
| EBJD，去掉自由采样 loss | 区分单步去噪学习与真实预测风险学习 |
| EBJD，去掉 actual-step constraint | 检验改善是否来自对边际梯度冲突的控制 |
| EBJD 完整主模型 | 最终主结果 |

所有必要开关在第一版实现。4/8 个 future blocks、D64/256 等容量搜索不作为主理论，最多在核心机制成立后用少量 sensitivity study 验证设计稳健性。

数据与预算：

- 五个 ETH/UCY held-out folds；每 fold 至少三训练 seeds，推理 seeds2035–2039；
- 选模使用预先确定的 validation，最终 test 不参与选模；若保留 mirrored validation/test 的旧协议，明确把它视为内部配对开发对照；
- 同样真实窗口、agents、地图来源、输出 P20；记录实际生成次数，不能生成 P40 再筛成 P20 而不报告；
- 新模型 rollout 训练更贵，同 epochs 不等于同 compute；GDTS finetune 必须有预算控制；
- 不把同一 checkpoint 的五推理 seeds 当成五个独立训练复现；
- 重叠窗口的置信区间按原始序列/连续块 bootstrap，不能简单假设每个窗口独立；
- 原 A0/A1 结果作为旧方法结果归档，不覆盖负结果，也不改动正在运行的正式配置。

选模规则：

1. 记录 validation 四项指标和 Pareto frontier。
2. 相对 matched GDTS validation，先筛 marginal minADE/minFDE 均未恶化的候选。
3. 在候选中按 JADE+0.5JFDE 选模；不足时保留最小约束违反的候选并标明“未达到同时改善”，不能伪装成功。
4. 正式 test 只评预先选定 checkpoint；报告每训练 seed 的结果、平均与不确定性。

论文若主张“同时改善”，至少需要五折汇总的四项主指标都下降，并展示训练 seed 方向；若置信区间跨零，只能报告观察到的均值改善，不能写成稳定显著改善。2% 等阈值可预注册为工程目标，但不是论文录用门槛。

机制诊断至少包括：

- goal minFDE 与 trajectory minFDE 的逐场景差，应仅为数值误差；
- 同 scene 权重的 M_A/M_F、J_A/J_F、两类 gap；
- endpoint 随训练的生成支持变化，不能只看固定候选 CE；
- 分噪声级的 provisional clean RMSE、future bias 范数、attention 贡献；
- marg/joint 梯度 cosine、投影频率、||d−u||/||u||、更新接近零的比例；
- P20 endpoint/trajectory diversity，检查是否仅收缩样本提高部分指标；
- N=1、N≥2、历史交互边存在的分组；防止数据组成掩盖联合机制；
- CRmean 及选中 joint worlds 的碰撞指标，不能只对距离误差做论文结论。

预测 clean geometry 改善不成立时，不能以其形式优美为理由继续宣称有效；梯度约束长期阻塞 joint 改善时，也应如实报告。这些是候选研究假设的证伪条件。

**20．本次核对与实现验收**

已运行 NumPy 代数检查与一个小型 factorized-attention 轴原型，结果如下。它们检查设计等式与轴语义，没有实际 PyTorch 模型训练。

| 检查 | 结果 |
|---|---:|
| 完整轨迹 → latent → 完整轨迹 max error | 1.78e−15 |
| literal endpoint 与 G 的差 | 0 |
| 修改桥变量后 endpoint 改变 | 0 |
| Brownian bridge precision 等式误差 | 7.11e−14 |
| bridge energy 等式误差 | 2.50e−12 |
| v→Z0/epsilon 重建 max error | 3.55e−15 |
| goal/trajectory FDE 差 | 0 |
| hard、soft M≤J | 均通过 |
| 两约束 QP 随机试验 | 2,000 次；最差 dot≈−8.52e−15，浮点精度范围 |
| attention 轴原型 agent 置换误差 | 1.78e−15 |
| world 置换误差 | 0 |
| 改其他 worlds 后第一个 world 的变化 | 0 |
| 按完整规格解析参数计数 | 3,612,764 |

这些数值不能替代 GPU forward、真实网络的梯度检查、采样稳定性、性能评估或方法有效性实验。

正式实现的必要验收：

1. 用 N=1、N=2、多个 scenes、不等长 agent padding 测试完整 forward/rollout。
2. 实例化 named_parameters 并核实所有主模块 requires_grad、预期参数量与梯度路径。
3. 在多人 batch 验证 UNet、history、6 个 blocks、pre/final goal/bridge heads 可获得梯度；检查 graph 中不存在 legacy freeze/no_grad/detach。
4. 检查全部未 padding 输出有限，geometry/logsumexp/optimizer FP32，t=1、t≈0 无除零。
5. 对真实网络重复 agent/world permutation 与 world isolation 检查；比较共同 seed 下 batched/chunked sampler 的一致性。
6. 对 actual-step optimizer 做一次有限差分验证，确认约束的是实际 delta，而不是 clip/Adam 前的梯度。
7. 验证 packed metric adapter 与手算 marginal/joint 定义一致；防止单位转换两次、world 轴错置、重复积分。
8. profile 完整训练更新和 P20 inference 后记录实际资源，再进行正式训练。

**21．来源与新颖性边界**

代码依据：

- [当前审查固定源码](https://github.com/WuYanXingege/RSJG/tree/092ed91f1f4117f50aa835ec8e47cd2c9ce74a98)
- [当前模型及冻结/goal/context/采样路径](https://github.com/WuYanXingege/RSJG/blob/092ed91f1f4117f50aa835ec8e47cd2c9ce74a98/src/models/model.py)
- [复用的 U-Net](https://github.com/WuYanXingege/RSJG/blob/092ed91f1f4117f50aa835ec8e47cd2c9ce74a98/src/models/model_utils/U_net_CNN.py)
- [现有 metrics layout 与 scene weighting](https://github.com/WuYanXingege/RSJG/blob/092ed91f1f4117f50aa835ec8e47cd2c9ce74a98/src/metrics.py)
- [官方 GDTS 固定对照](https://github.com/Winderting/GDTS/tree/297d508558c10831983ea4b19c2b3e657459a449)

本次核对的论文均为作者论文或正式论文页：

- [P1 MotionDiffuser, CVPR2023](https://arxiv.org/abs/2306.03083)：已存在 joint trajectory diffusion，以及 PCA 表示；不能把 joint diffusion 本身作为首次贡献。
- [P2 QCNeXt, 2023](https://arxiv.org/abs/2306.10508)：未来社会交互与联合预测已有先例，也已有联合模型改善 marginal metrics 的报告。
- [P3 Future-Interactions-Aware Trajectory Prediction via Braid Theory, 2026](https://arxiv.org/abs/2603.22035)：future interaction 辅助监督已有先例；这里选择连续相对运动场，未宣称其三类 crossing 是本设计新贡献。
- [P4 Progressive Distillation for Fast Sampling of Diffusion Models, ICLR2022](https://arxiv.org/abs/2202.00512)：采用已有 v 参数化、VP cosine 与稳定的 clean reconstruction；本方案不要求进行 distillation。
- [P5 BBDM, CVPR2023](https://arxiv.org/abs/2205.07680)：Brownian bridge 在 diffusion 中已有应用；本文使用的是未来时间轴上的可逆桥白化，不等同于其 image-to-image diffusion bridge。
- [P6 Gradient Episodic Memory, NeurIPS2017](https://arxiv.org/abs/1706.08840)：约束梯度方向/QP 有已有来源；本方案保护自由生成 marginal risks 的实际 AdamW update。
- [P7 DDIM, ICLR2021](https://arxiv.org/abs/2010.02502)：20-step sampler 使用已有确定性更新。
- [P8 GDTS](https://arxiv.org/abs/2311.14922)：原方法的 goal-guided diffusion 与 tree sampling 作为对照来源。

上述核对不是完整的新颖性检索。设计目前最有价值的研究主张是：**在同一个可学习、端点一致的联合生成器中，使连续终点覆盖、未来交互路径生成与边际约束的自由采样优化形成闭合梯度链，并通过误差分解证明收益来自生成能力和共同未来学习。** 只有结构、理论命题和机制实验共同成立，才能支撑论文贡献；单纯把已有技术拼接进表格不能证明原创性。




