# RSJG / Joint Dependency V2 当前网络结构与问题清单

## 1. 文档目的

本文档整理当前仓库中实际存在的三层模型结构、它们之间的冻结关系，
以及截至 2026-10-03 仍需关注的问题。它是工程/研究 handoff 文档，
不是新的实验结论，也不改变任何已冻结的方法。

与 SDD 当前运行状态配套阅读：

- docs/joint_dependency_v2/SDD_BASELINE_CURRENT_STATUS.md
- docs/joint_dependency_v2/GDTS_SDD_PAPER_ALIGNED_PROTOCOL.md
- docs/joint_dependency_v2/JDV2_STAGE_A_FREEZE.md
- docs/joint_dependency_v2/JDV2_STAGE_B_V2A_ORIGINAL_PROTOCOL_ADOPTION_FREEZE.md

最重要的边界是：

> 当前正在运行的 SDD 实验是原始 independent GDTS baseline。
> 它尚未启用 JDV2 Stage-A 的 relation/energy/sampler，也未启用
> Stage-B DependencyCorrector。

因此不能把 ETH 上已经冻结的 JDV2 结果理解成当前 SDD 训练的实时结果。

## 2. 当前模型矩阵

| 层级 | 当前用途 | 主状态 | 是否用于当前 SDD run |
|---|---|---|---|
| 原始 GDTS | marginal trajectory baseline | SDD 两阶段训练进行中 | 是 |
| JDV2 Stage-A | joint goal dependency | ETH strict-no-z 已冻结 | 否 |
| JDV2 Stage-B V2-A | trajectory dependency correction | ETH original protocol 已冻结 | 否 |

当前 SDD run 的配置中：

- goal_model_type = independent；
- training_stage = baseline；
- trajectory_coupling = none；
- jdv2_active = null；
- amp_enabled = false，FP32；
- num_samples = 20。

虽然全局配置文件中仍能看到 use_scene_latent、use_dynamic_relation、
use_joint_energy、use_dependency_corrector 等通用 parser 字段，但在
goal_model_type=independent 且 jdv2_active 未启用时，这些 JDV2 模块不会
构造或执行。这些字段的存在不代表 SDD baseline 已经开启 JDV2。

## 3. 当前 SDD 原始 GDTS 网络

### 3.1 总体数据流

    6-channel scene raster + 8 observed trajectory maps
      -> Goal U-Net
      -> 12 future heatmaps
      -> final-step heatmap
      -> TTST endpoint sampling
      -> 20 deployed goals + 1 internal trunk goal
      -> goal-relative history representation
      -> LSTM history encoder, D=256
      -> transformer diffusion denoiser
      -> 12 future velocity steps
      -> cumulative integration
      -> 20 trajectory samples

输入/输出时间长度固定为：

- observed length：8；
- predicted length：12；
- complete sequence length：20。

### 3.2 Goal U-Net

Goal U-Net 的输入通道数为 14：

- 6 个 scene/map 通道；
- 8 个 observed trajectory heatmap 通道。

编码通道：

    (14, 32, 32, 64, 64, 64)

解码通道：

    (64, 64, 64, 32, 32)

输出为 12 个 future-step raw-logit heatmaps。最后一个 future heatmap
用于 endpoint sampling。网络采用普通 double-convolution、max-pooling、
bilinear upsampling 和 U-Net skip connection。

Goal-only 预训练模型参数量为 613,772。当前 SDD 正式 pipeline 首先只训练
这个模块 150 epochs，目标为 masked Goal BCE；best checkpoint 按
validation Goal BCE 选择。

### 3.3 Goal sampling 与 tree branches

测试时，最后一步 heatmap 经过 sigmoid 后由 TTST 产生多模态 endpoint。
外部评估使用 20 个 sample branches；另有 1 个 heatmap MAP/argmax endpoint
作为内部 diffusion tree trunk condition。内部 trunk 不作为第 21 个预测样本
报告。

原始 independent GDTS 对每个 agent 独立生成 goal，因此这一层本身没有显式
multi-agent joint relation 或 joint energy。

### 3.4 Goal-relative history encoder

每个 goal branch 都构造一个 shape 为 [N,8,8] 的历史输入：

- 前 2 维：观测位置减去该 branch 的 goal endpoint；
- 后 6 维：缓存中的运动/历史增强特征。

历史编码器为：

- input size：8；
- single LSTM；
- hidden size / context dimension：256。

输出 [N,256] context，并扩展为 [N,1,256] 给 diffusion denoiser。

### 3.5 Diffusion denoiser

基础 denoiser 是 TransformerConcatLinear：

- 输入：noisy future velocity [N,12,2]；
- context：goal-relative history embedding [N,1,256]；
- diffusion timestep embedding：beta、sin(beta)、cos(beta)；
- first projection：2 -> 512；
- positional encoding：d_model=512；
- Transformer encoder：2 layers、4 heads、feed-forward width 1024；
- conditioned projections：512 -> 256 -> 128 -> 2。

训练使用 100-step linear diffusion schedule，直接回归 future-velocity
noise，diffusion loss 为 MSE。

推理采用 tree-style sampling：

1. internal trunk condition 共享运行前段 diffusion；
2. 在 trunk_stage_step=30 处分支；
3. 20 个 goal conditions 各自运行后段 DDIM branch；
4. 将 12 个预测 velocity 累积为 trajectory。

当前 SDD 完整 GDTS 模型参数量为 7,826,588。

### 3.6 原始 GDTS joint-training objective

在 paper-aligned joint stage 中，基础 loss 为：

    L_GDTS = L_diffusion + 20 * L_goal_BCE

其中：

- L_diffusion：预测 noise 与 sampled Gaussian noise 的 MSE；
- L_goal_BCE：12 个 future heatmap 的 masked BCE；
- history encoder、Goal U-Net、diffusion denoiser 一起训练；
- 不包含 relation、joint energy 或 DependencyCorrector loss。

### 3.7 当前 SDD 两阶段训练边界

当前正式 SDD pipeline：

1. Goal U-Net：150 epochs，Adam，lr=1e-3，
   ExponentialLR gamma=0.99；
2. 选择 best Goal BCE checkpoint；
3. 新建完整 GDTS；
4. 只严格加载 goal_module 权重；
5. history LSTM 和 diffusion denoiser fresh initialize；
6. joint training：250 fresh epochs，Adam，lr=1e-3，
   ExponentialLR gamma=0.995；
7. 最终 checkpoint 做 5 次 stochastic evaluation。

Goal 阶段的 optimizer、scheduler、epoch 和 best-selection state 不会传入
joint stage。

## 4. 已冻结的 JDV2 Stage-A 网络

### 4.1 总体结构

ETH 上冻结的 Stage-A 是 strict_no_z：

    History X
      -> frozen Goal U-Net candidate bank, K=21
      -> parameter-free sparse radius+TTC graph
      -> SocialMotionEncoder
      -> hypothesis-conditioned relation
         p(r_ij | X, g_i, g_j), M=4
      -> relation-specific low-rank joint energy, rank=8
      -> structured P=20 joint goal worlds
      -> frozen goal-relative LSTM + frozen GDTS diffusion tree

不存在 global categorical scene latent z。冻结主架构中没有：

- p(z|X)；
- q(z|X,Y*)；
- gamma；
- scene-mode sampling；
- scene embedding；
- z-indexed relation 或 energy branch。

仓库仍保留历史 latent-objective 实现用于旧 checkpoint、审计与 ablation
兼容，但它们不是当前冻结主方法。

### 4.2 Sparse interaction graph

图为 parameter-free radius+TTC proposal graph：

- radius：6 m；
- TTC threshold：8 s；
- dt：0.4 s；
- edge feature width：14；
- edge 不能跨 scene；
- canonical undirected edge 使用 src < dst。

E=0 和 degree=0 路径具有显式 identity/bypass 语义。

### 4.3 SocialMotionEncoder 与 relation

SocialMotionEncoder 输出：

    agent feature [N,128]

基础 relation 有 M=4 个 latent modes，随后使用 agent history、
candidate-pair geometry 和 base relation 构造 hypothesis-conditioned
relation：

    full relation probability [E,21,21,4]

relation 是 candidate-pair dependent，而不是单纯 history-only relation。

### 4.4 Unary 与 low-rank pair energy

Unary score：

    frozen candidate log prior + learned unary residual
    shape [N,21]

每条 edge、每个 relation mode 的左右 low-rank factors：

    [E,4,21,8]

组合后得到 candidate-pair energy。系统没有枚举 K^N joint combinations；
训练和采样只在 sparse edges、candidate pairs 和 P=20 worlds 上工作。

### 4.5 冻结 structured sampler

正式 policy：

    exact_lexicographic_persistent_tie

Round 0：

- weighted Gumbel-Top-P without replacement；
- K=21，P=20；
- 每个 agent 初始保留 20 个不同 candidate。

两轮 synchronous refinement：

- degree=0：candidate IDs tensor-exact identity；
- degree>0：对 [P,K] conditional score 做 injective assignment；
- score 为 unary 减去 accumulated pair energy；
- exact lexicographic objective 为
  (J, C_stay, C_geom, R)；
- persistent exchangeable resolver R 在两轮间保持一致；
- 每轮继续保持 20/20 coverage。

这个 sampler 是 sample-set-level structured allocation，不应描述成 exact
K^N likelihood，也不应描述成全局最大化 multi-agent compatibility。

### 4.6 Stage-A 输出合同

核心张量：

- candidate bank：[N,21,2]；
- unary：[N,21]；
- relation：[E,21,21,4]；
- sampled IDs：[N,20]；
- joint goals：[N,20,2]；
- selected relation probability：[E,20,4]；
- relation embedding：[E,20,16]。

20 个 deployed worlds 加 1 个 internal trunk 进入冻结 diffusion tree，
最终只输出 20 个 trajectory samples。

Stage-A checkpoint：

- epoch：13；
- SHA256：
  699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb。

## 5. 已冻结的 JDV2 Stage-B V2-A

### 5.1 数据流

Stage-B 不改变 Stage-A candidate IDs、joint goals、relation/energy 或
diffusion backbone。它只在 diffusion branch 的已批准 timesteps 上预测
dependency residual：

    epsilon_joint = epsilon_base + projected_delta_epsilon

完整路径：

    frozen Stage-A worlds and relation embeddings
      -> frozen GDTS denoiser epsilon_base
      -> DependencyCorrector delta_raw
      -> parameter-free component_zero_mean projection
      -> identity-safe active-agent routing
      -> corrected DDIM transition

### 5.2 DependencyCorrector

DependencyCorrector 有 30,851 个参数，固定配置：

- hidden width：64；
- relation embedding：16；
- input pair state：relative position 2 + relative velocity 2
  + distance 1 + relation 16 = 21；
- timestep sinusoidal embedding：32；
- FiLM、gate/value aggregation、2D residual head；
- output shape：[N,12,2]。

只训练 corrector。Goal U-Net、history encoder、base denoiser、Stage-A
relation/energy/sampler 全部 frozen。

### 5.3 V2-A projection 与 numerical identity

V1 audit 发现 component-common residual 是主要有害模式。V2-A 在 corrector
之后、epsilon addition 之前，对每个 interacting connected component 做
parameter-free zero-mean projection。

同时保留严格 numerical identity：

- whole-window E=0：直接走 Stage-A arithmetic path；
- mixed scene 中 degree=0 agent：直接保留 Stage-A transition；
- 只有 degree>0 agent 进入 correction arithmetic；
- baseline/corrected path 共享同一个 diffusion noise，不额外消费 RNG。

Stage-B V2-A checkpoint：

- epoch：5；
- SHA256：
  e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73。

## 6. 冻结/训练关系

| 阶段 | Goal U-Net | History LSTM | Base denoiser | Relation/Energy/Sampler | Corrector |
|---|---|---|---|---|---|
| SDD Goal pretrain | train | absent | absent | absent | absent |
| SDD GDTS joint | train | train | train | absent | absent |
| JDV2 Stage-A | frozen | frozen | frozen | train/frozen checkpoint | frozen/off |
| JDV2 Stage-B V2-A | frozen | frozen | frozen | frozen | train |
| JDV2 inference | frozen | frozen | frozen | frozen | frozen |

这张表也是 checkpoint 兼容性的核心：SDD Goal checkpoint 只能初始化
goal_module；Stage-A/Stage-B checkpoint 不应反向覆盖 GDTS baseline。

## 7. 当前问题清单

### 7.1 当前阻塞或未完成事项

#### P0 — SDD paper-aligned baseline 尚未产出结果

当前 run 仍处于 150-epoch Goal U-Net pretraining。尚无完成 epoch 的正式
Goal checkpoint，250-epoch joint stage 也未开始，因此现在没有可与论文
7.42/11.57 直接比较的 paper-aligned SDD ADE/FDE。

旧 gdts_baseline_sdd_seed2035 是 joint-from-scratch diagnostic，训练协议
不一致，不能用它判断 paper reproduction 是否成功。

#### P0 — SDD 运行时间与恢复粒度

完整 cache 有 2440 train batches/epoch。当前受控配置只给 4 CPU cores、
num_workers=1，并使用 FP32。单 epoch 很长，150+250 epochs 是明显的工程
时间风险。

当前 checkpoint 以 epoch 为恢复边界。如果机器在 epoch 中途重启，已完成的
batch 不能从 batch-level state 精确恢复，只能从最近完整 epoch 重跑。

在本次正式 run 中不应为了提速临时改变 worker、precision、batch semantics
或 scheduler；若要优化吞吐，应先单独做 protocol-preserving benchmark。

#### P0 — SDD validation/test 不是 clean held-out

legacy 30/17 protocol 中 validation 和 test 使用同一组 17 scenes。它可以
用于复现原始 published protocol，但 best-checkpoint selection 后的 test
不是独立 held-out estimate。

报告必须明确标注：

- legacy/original SDD protocol；
- ADE/FDE 单位为 pixels；
- 不宣称 clean independent-test generalization。

### 7.2 JDV2 当前研究限制

#### P1 — JDV2 尚未在 SDD 上训练/验证

ETH 的 Stage-A/Stage-B 结果不能外推为 SDD 结果。当前 SDD 只在建立可靠
GDTS baseline。后续是否进入 SDD JDV2，必须等 baseline checkpoint 与协议
先冻结。

#### P1 — Stage-B V2-A 增益较小且有 trade-off

ETH original-protocol 五种子均值中，V2-A 相比 Stage-A：

| Metric | Stage-A | V2-A | Relative |
|---|---:|---:|---:|
| minADE | 0.278201 | 0.279971 | +0.636% |
| minFDE | 0.386215 | 0.387217 | +0.259% |
| JADE | 0.413185 | 0.414958 | +0.429% |
| JFDE | 0.701908 | 0.698169 | -0.533% |
| Relative Motion | 0.297907 | 0.297441 | -0.156% |

V2-A 修复了 V1 的 common-mode harm，并改善 JFDE/relative motion，但对
Stage-A 的 marginal metrics 和 JADE 有小幅退化。因此目前最稳妥的结论是：

- V2-A 优于有缺陷的 V1；
- 它没有证明对 Stage-A 所有指标都普遍更优；
- 不应因为单个 joint metric 改善而忽略 marginal trade-off。

#### P1 — ETH 主要协议同样是 mirrored validation/test

ETH original GDTS protocol 的 validation/test 内容一致。方法间 paired
comparison 是公平的，但仍不是严格独立 held-out generalization。

clean-split reproduction 曾启动但中止，现有 partial checkpoint/metrics
被标记为 non-authoritative，不得用于论文结论或初始化后续阶段。

#### P1 — Cross-dataset generalization 尚未完成

Stage-A 和 Stage-B 的正式冻结证据主要来自 ETH。UNIV 工作已经开始，但
SDD/HOTEL/ZARA 的完整同协议结果尚未形成。模型是否能稳定跨 dataset 保留
joint gain，仍是开放问题。

### 7.3 已知但非阻断的工程行为

#### P2 — Stage-A synchronous two-cycle

两轮 exact structured refinement 的 observed two-cycle rate 约为 40.66%。
它没有破坏 20/20 coverage，且通过冻结验证，因此当前定义为
KNOWN_NON_BLOCKING_BEHAVIOR。需要持续记录，但冻结期不能私自修改。

#### P2 — exact sampler 成本与默认配置风险

exact persistent sampler 单独看比简单 D0 慢，但 full-forward runtime gate
已通过。

全局 parser 默认 refinement policy 仍是 categorical，用于旧配置兼容；
正式 Stage-A 必须显式加载 frozen config 中的
exact_lexicographic_persistent_tie。若手写命令遗漏该字段，可能无意中回到
旧 categorical behavior。

#### P2 — 历史模块造成配置认知负担

源码仍包含：

- 早期 global scene latent V1/V2；
- legacy social/lowrank/energy/joint ablations；
- categorical/CPSR/audit policies；
- multiway/alignment experiments。

这些模块用于兼容和审计，但增加了误配风险。当前主结论只能由明确的
goal_model_type、jdv2_active、latent objective、refinement policy 和
training_stage 联合确定，不能仅凭模块在源码中存在来判断运行结构。

#### P2 — 坐标与指标不可混报

- SDD legacy ADE/FDE：pixels；
- ETH/UCY JDV2：world metres；
- minADE/minFDE：agent-marginal best-of-K；
- JADE/JFDE：scene-level common-world joint selection；
- endpoint/compatibility/relative-motion：JDV2 joint diagnostics。

不同坐标、split 和 sample-selection semantics 的数字不能放在同一张表里直接
比较。SDD baseline 在启用 JDV2 前也不应声称已有 relation/compatibility gain。

## 8. 已解决问题，不应重新打开

下列问题已经通过实验和冻结评审解决，不是当前 SDD baseline 的待办：

1. global scene latent winner-take-all collapse：
   主架构改为 strict-no-z；
2. iid-with-replacement 初始 slot duplication：
   Round 0 改为 weighted Gumbel-Top-P without replacement；
3. categorical refinement coalescence：
   改为 exact injective structured refinement；
4. arbitrary slot/tie semantics：
   使用 exact lexicographic objectives 和 persistent exchangeable resolver；
5. degree=0 无信息重排：
   refinement identity bypass；
6. Stage-B zero residual 在 BF16 下不保持数值 identity：
   inactive agents 直接走 Stage-A arithmetic path；
7. Stage-B V1 component-common drift：
   V2-A 使用 parameter-free component-zero-mean projection；
8. SDD Goal-pretrain loss-mask crash：
   已修复并通过完整 test suite。

除非新的可复现证据违反冻结合同，否则不应在 SDD baseline 训练期间同时重开
这些设计问题。

## 9. 建议的近期检查顺序

不改变当前 run 的前提下，最小必要顺序是：

1. 等待 SDD Goal epoch 1 完成，确认 loss finite、checkpoint/last checkpoint
   可 reload；
2. 首次 validation 时确认 raw-logit Goal BCE 与 best-selection 正常；
3. Goal 150 epochs 完成后核对 best checkpoint SHA256 与 provenance；
4. 确认 fresh joint model 只加载 goal_module，history/diffusion 未继承旧 state；
5. joint stage 先检查 diffusion loss、Goal BCE、ADE/FDE 是否 finite；
6. 完成 250 epochs 后再执行五次 stochastic evaluation；
7. 只在 paper-aligned baseline 成立后，单独评审 SDD JDV2 cache/Stage-A
   preflight，不把 JDV2 训练混入 baseline reproduction。

## 10. 当前结论

当前代码库不是“一个正在训练的统一 JDV2 模型”，而是三层清晰分离的系统：

1. SDD 正在训练原始 GDTS baseline；
2. ETH Stage-A 已冻结 strict-no-z joint-goal dependency；
3. ETH Stage-B V2-A 已冻结 component-relative trajectory correction。

现阶段最大的实际问题不是网络结构再次设计，而是：

- 完成并验证 SDD paper-aligned baseline；
- 保持不同 dataset、坐标和 split protocol 的报告边界；
- 在跨数据集启用 JDV2 前先完成 dataset-specific preflight；
- 不把 Stage-B 的小幅 joint gain 夸大成所有 trajectory metrics 的普遍提升。

本文档只记录现状；没有修改模型、loss、sampler、checkpoint 或运行中的训练。
