<h2 align="center">GDTS: Goal-Guided Diffusion Model with Tree Sampling for Multi-Modal Pedestrian Trajectory Prediction</h2>
<div align="center">
  <br>Ge Sun, Sheng Wang, Lei Zhu, Ming Liu and Jun Ma*.
  <br>HKUST, HKUST(GZ)
  <br>IROS 2025
</div>

> 本分支新增的 Stage 3 v3 完整轨迹对齐、训练入口和验收标准见
> [`docs/TRAJECTORY_ALIGNMENT_V3.md`](docs/TRAJECTORY_ALIGNMENT_V3.md)。

<p align="center">
<a href="https://arxiv.org/pdf/2311.14922.pdf">
  <img src="https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=white" alt="arXiv PDF">
</a>
</p>

## Abstract

Accurate prediction of pedestrian trajectories is crucial for improving the safety of autonomous driving. However, this task is generally nontrivial due to the inherent stochasticity of human motion, which naturally requires the predictor to generate multi-modal prediction. Previous works leverage various generative methods, such as GAN and VAE, for pedestrian trajectory prediction. Nevertheless, these methods may suffer from mode collapse and relatively low-quality results. The denoising diffusion probabilistic model (DDPM) has recently been applied to trajectory prediction due to its simple training process and powerful reconstruction ability. However, current diffusion-based methods do not fully utilize input information and usually require many denoising iterations that lead to a long inference time or an additional network for initialization. To address these challenges and facilitate the use of diffusion models in multi-modal trajectory prediction, we propose GDTS, a novel Goal-Guided Diffusion Model with Tree Sampling for multi-modal trajectory prediction. Considering the "goal-driven" characteristics of human motion, GDTS leverages goal estimation to guide the generation of the diffusion network. A two-stage tree sampling algorithm is presented, which leverages common features to reduce the inference time and improve accuracy for multi-modal prediction. Experimental results demonstrate that our proposed framework achieves comparable state-of-the-art performance with real-time inference speed in public datasets.

<div align="center">
  <br><img src="img/network.png" width="80%" alt="Original GDTS network">
</div>

## Relation-aware joint-goal extension

本分支在保留原始 GDTS goal heatmap、goal-conditioned diffusion 和 tree sampling 主干的前提下，将逐行人独立目标采样扩展为：

```text
Relation-aware socially-coupled joint goal distribution
  = low-rank scene-level social modes
  + sparse relation-conditioned pairwise goal potentials
```

其概率结构可写为：

```text
p(Y, G, R, Z | X)
  = p(Z | X) p(R | X) p(G | Z, R, X) p(Y | G, X)
```

其中 `Z` 表示 scene-level latent social mode，`R` 表示 sparse edge 上的 latent relation，`G` 是所有行人的联合终点配置，`Y` 是由原 GDTS diffusion 生成的未来轨迹。关系类别不使用 friend/yield 等人工标签；soft relation 可以从数据中学习 coordination、avoidance、following 或 weak interaction，而不是将所有近邻统一视为排斥。

### Repository audit and compatibility decisions

实现前审计得到的关键结论如下：

- 原始 goal U-Net 输入为场景语义图和单个行人的 8 帧 observation heatmaps，输出未来 12 帧 heatmaps；测试时仅从最后一张 endpoint heatmap 取目标。
- 原始 TTST 对每个行人独立执行 sampling 和 k-means。旧接口请求 `K` 个 clusters 后还会附加一个 argmax，因此内部实际为 `K+1` 个点；cluster 序号跨行人没有共同语义，不能把同一旧 `sample_idx` 解释为 coherent joint future。
- diffusion denoiser 把行人数 `N` 当 batch 维，只在 12 个 future timesteps 上做 Transformer encoding；tree trunk/branch 共享发生在同一行人的多模态之间，不是行人之间。
- 原始预处理会将同一大场景中不同 agent、不同 starting frame 的单人片段混入一个 batch，训练时还会打乱。这种 batch 不能构造有效 social graph。
- 因此，`independent` 完整保留 legacy batch/cache 和 checkpoint 路径；其余四种模式使用独立的 version-2 synchronized scene-window cache。预处理先以整个 scene 的统一 frame origin 对齐时间相位，再按完全相同的 frame sequence 聚合 agent，并保留 `scene_index`/`scene_ptr`；这避免了按各 agent 首帧分别下采样而产生伪 single-agent windows。
- joint sampler 插在 candidate generation 与 goal-relative history encoding 之间。输出的每一列是一个完整 scene future，随后原 diffusion conditioning 和 tree sampler 可以继续使用同一接口。

## Pipelines

### GDTS-Base

```text
semantic map [6,H,W] + observed maps [N,8,H,W]
  -> goal U-Net -> endpoint heatmap [N,1,H,W]
  -> independent TTST goals [N,num_samples+1,2]
  -> goal-relative history LSTM contexts [num_samples+1,N,1,e_dim]
  -> original diffusion tree sampling
  -> trajectories [num_samples,20,N,2]
```

训练 diffusion 时仍使用 ground-truth endpoint 做 goal conditioning；测试时使用 TTST goals。最后一个 argmax goal 用于 shared tree trunk，前 `num_samples` 个 goals 对应 branches。

### Full-Joint path

```text
endpoint heatmap
  -> explicit candidates and heatmap prior [N,K,2], [N,K]
world observations [N,8,2]
  -> sparse interaction graph -> social encoder
  -> relation inference on sparse edges
  -> low-rank scene mode p(z|X) and p_i(g_i|z,X)
  -> sparse relation-conditioned low-rank energy
  -> latent-mode initialization + blocked Gibbs-like refinement
  -> coherent branch goals [N,num_samples,2] + one MAP trunk goal
  -> original goal-relative LSTM + diffusion tree sampling
```

`radius_ttc` graph 同时保留当前距离在 radius 内的 pair，以及未来最近接距离进入 radius 且 TTC 未超过阈值的 pair。近距离同行者因此不会因 relative velocity 接近零而被 graph gate 删除；其 formation-preserving compatibility 由 latent relation 和 group-relative endpoint feature 学习，而非硬编码排斥规则。

上图是 Full-Joint 的完整链路。Social-GDTS 将有效 mode 数固定为 1，并关闭 relation/energy；LR-Goal 使用多个 shared modes、关闭 relation/energy；Sparse-Energy 使用单一 heatmap unary、relation/energy 和 refinement。各分支的精确定义见下方 ablation 表。

### Implementation map

| Component | Location | Responsibility |
|---|---|---|
| synchronized grouping | `src/data_grouping.py`, `src/data_pre_process.py` | isolate format-v2 scene windows from the legacy cache |
| interaction graph | `src/models/interaction_graph.py` | canonical sparse pairs, 14-D edge features and edge weights |
| social/relation encoders | `src/models/social_encoder.py`, `src/models/relation_inference.py` | equivariant agent features and soft/hard-ST latent relations |
| global joint goal | `src/models/joint_goal.py` | scene-mode prior and mode-conditioned candidate marginals |
| pairwise compatibility | `src/models/goal_energy.py` | relation-conditioned low-rank sparse-edge factors |
| structured inference | `src/models/joint_sampler.py` | shared-mode initialization and local two-end refinement |
| objectives/integration | `src/joint_goal_loss.py`, `src/models/model.py` | soft targets, mode/pseudo-likelihood losses and GDTS interface |
| evaluation/diagnostics | `src/metrics.py`, `tools/visualize_joint_goal.py` | marginal, joint, collision and scene visualization tools |

## Tensor contracts

默认 `T_obs=8`、`T_pred=12`、`T=20`。下表中 `N` 是当前同步窗口的行人数，`K=num_goal_candidates`，`S=num_samples=num_joint_samples`，`R` 是有效 scene-mode 数（LR-Goal/Full-Joint 为 `num_social_modes`，Social-GDTS/Sparse-Energy 为 1），`M=num_relation_modes`，`r_e=goal_pair_rank`，`E` 是 canonical sparse pairs 数，`D=social_feature_dim`。`H,W` 随场景地图变化。

| Tensor | Shape | Coordinate/meaning |
|---|---:|---|
| `abs_pixel_coord` | `[T,N,2]` | downsampled GDTS pixel coordinates |
| `scene_index` | `[N]` | agent-to-window assignment |
| `obs_traj_world` | `[N,T_obs,2]` | world coordinates in metres |
| goal U-Net input | `[N,6+T_obs,H,W]` | semantic channels + observation maps |
| `goal_logit_map` | `[N,T_pred,H,W]` | future-position logits |
| `goal_prob_map` | `[N,1,H,W]` | final endpoint map |
| `goal_candidates_map` | `[N,K,2]` | explicit endpoint candidates in downsampled heatmap coordinates |
| `goal_candidates_world` | `[N,K,2]` | the same candidates converted to metres |
| `candidate_prob` | `[N,K]` | normalized heatmap mass at candidates |
| `agent_feat` | `[N,D]` | shared temporal encoder + social message passing |
| `edge_index` | `[2,E]` | one canonical `src < dst` pair per interaction |
| `edge_feat` | `[E,14]` | relative position/velocity, distance, heading, speed, TTC, historical-distance and formation features |
| `relation_prob` | `[E,M]` | soft latent relation distribution |
| `mode_prob` | `[num_scenes,R]` | scene-level latent-mode distribution |
| `conditional_goal_prob` | `[N,R,K]` | candidate probability conditioned on social mode |
| `left_factor`, `right_factor` | `[E,M,K,r_e]` | low-rank relation-specific goal factors |
| `JointGoalSampler` branch output | `[N,S,2]` | world-coordinate column `s` is one complete scene-level sample |
| integrated `joint_goal_points_{map,world}` | `[N,S+1,2]` | inference only; `S` branches followed by one MAP trunk goal |
| training diffusion context | `[1,N,1,e_dim]` | teacher-forced ground-truth endpoint context |
| inference diffusion contexts | `[S+1,N,1,e_dim]` | `S` branch contexts plus one trunk context |
| predicted velocities | `[S,N,T_pred,2]` | diffusion output |
| final trajectories | `[S,T,N,2]` | GDTS metric layout |

Graph construction, TTC, joint losses, collision metrics and goal-level metrics use world metres. Heatmap generation and the legacy diffusion interface retain GDTS pixel/downsampled-pixel conventions; conversion happens at their boundary.

## Model variants and ablations

`--goal_model_type` accepts the five canonical values below. Paper-facing aliases such as `GDTS-Base`, `Social-GDTS`, `LR-Goal`, `Sparse-Energy` and `Full-Joint` are normalized by the parser as well.

| Experiment | CLI value | Social context | Global low-rank mode | Pairwise energy | Goal sampling |
|---|---|---:|---:|---:|---|
| GDTS-Base | `independent` | no | no | no | original independent TTST |
| Social-GDTS | `social` | yes | no | no | socially informed but per-agent independent |
| LR-Goal | `lowrank` | yes | yes | no | shared `z`, no local energy refinement |
| Sparse-Energy | `energy` | yes | no (single effective mode) | yes | sparse relation-conditioned refinement |
| Full-Joint | `joint` | yes | yes | yes | shared `z` plus sparse refinement |

常用消融参数：

```text
--use_social_encoder       False disables neighbour messages but retains the shared temporal encoder
--num_goal_candidates      K, default 21
--num_social_modes          R, default 8
--num_relation_modes        M, default 4
--goal_pair_rank            r_e, default 8
--num_refinement_steps      L, default 2; 0 disables refinement
--graph_type                full | radius | radius_ttc
--graph_radius              default 6.0 metres
--ttc_threshold             default 8.0 seconds
--hard_relation             False = soft mixture; True = straight-through hard mode
--use_group_relative_feature
--joint_sampling_mode       sample | map
--joint_sampling_temperature
--energy_weight
```

通常无需手工设置 `--joint_goal_enabled`；它会根据 `goal_model_type` 推导。若显式设置 `--num_joint_samples`，其值必须等于 `--num_samples`，从而保证每个 diffusion tree branch 对应一个完整 joint goal configuration。

## Losses and numerical stability

总体目标为：

```text
L = lambda_goal * L_goal
  + lambda_mode * L_mode
  + lambda_PL   * L_PL
  + lambda_diff * L_diff
  + optional relation/mode regularizers
```

- `L_goal` 保留原 GDTS `BCEWithLogitsLoss` 定义，并覆盖 12 张 future heatmaps。
- `L_diff` 保留原 velocity-noise denoising MSE。
- `build_soft_goal_target` 使用以 `goal_soft_sigma` 为带宽的 Gaussian soft assignment，不使用 nearest-candidate hard label。
- `L_mode` 以 `logsumexp_r(log p(z=r|X) + sum_i sum_k q_i(k) log p_i(k|r,X))` 计算 scene likelihood。
- relation mixture 在 compatibility space 中计算 `E_eff=-logsumexp_m(log q(r=m)-E_m)`，不以平均 energy 代替多模态 mixture。
- `L_PL` 仅在每个 agent 的 `K` 个候选上归一化；邻居 GT 默认使用 soft expectation，也可用 `--pairwise_neighbor_target nearest`。
- mode/relation utilization KL 使用最近 64 个 detached scene-window usage 的滑动历史；第一个 window 的 balance loss 为 0，避免因 DataLoader 每项只有一个 scene 而把每个 posterior 强制成 uniform。
- 概率路径使用 `log_softmax`/`logsumexp`，并对 zero-mass/invalid candidate、zero-edge、NaN/Inf 做显式检查。

对应权重由 `--lambda_goal`、`--lambda_mode`、`--lambda_PL`、`--lambda_diff`、`--lambda_relation_entropy`、`--lambda_relation_balance` 和 `--lambda_mode_balance` 控制。

## Data preparation and caches

原始数据可按上游 GDTS/Goal-SAR 提供的脚本下载：

```bash
bash ./download_data.sh
```

下载脚本需要系统已安装 `curl` 和 `unzip`；它使用临时归档，并在已有 `./data` 或 `../data` 时拒绝覆盖。若数据已放在仓库内的 `./data`，可在仓库父目录建立 `data -> GDTS/data` 符号链接以匹配上游数据定位逻辑。

`main.py` 在 train/test 前都会检查预处理 cache，也可以显式预处理 Full-Joint 数据：

```bash
python main.py \
  --dataset eth5 \
  --test_set eth \
  --phase pre-process \
  --goal_model_type joint \
  --force_reprocess True \
  --data_augmentation False \
  --use_wandb False
```

缓存和 checkpoint 彼此隔离：

```text
GDTS-Base: output/<test_set>/data_batches/
           output/<test_set>/saved_models/

Other:     output/<test_set>/<goal_model_type>/data_batches_joint_v2/
           output/<test_set>/<goal_model_type>/saved_models/
```

不要把 legacy `data_batches` 当作 social windows。如果代码提示缺少 format-v2 或同步 `frame_ids`，请用对应 `goal_model_type` 加 `--force_reprocess True` 重建；这不会覆盖 baseline cache。

每个 cache 根目录包含 `cache_manifest.json`。`down_factor`、`skip_ts_window`、baseline `batch_size`、debug 状态或 schema 改变时，预处理器会自动重建所选生成缓存，避免静默复用不兼容的 maps/windows。

Structured preprocessing 的一个 pickle 对应一个完整同步 window，行人数可变且不会按 legacy `batch_size` 拆散，因此 `N` 可能大于命令行的 `--batch_size`。

Structured modes 依赖原始 scene homography 将像素坐标转换为 metres。Legacy augmentation 只变换像素/地图而不更新 homography，因此 parser 会为所有非 `independent` 模式自动关闭它并给出警告；在命令中显式传 `--data_augmentation False` 可避免警告。Baseline 的原始 augmentation 行为保持不变。

## Training and testing

### Environment

The original GDTS experiments used Ubuntu 22.04, Python 3.9.6, PyTorch 1.13.1 and CUDA 11.7. This extension introduces no graph-library dependency and uses PyTorch `index_add_` for sparse message passing.

请先按本机 CUDA/CPU 环境安装**成对兼容**的 `torch`/`torchvision` wheel，再安装其余运行、可视化和测试依赖；为避免依赖解析器替换已选择的 CUDA wheel，这两个包有意不放在 `requirements.txt` 中。原论文环境的 CUDA 11.7 参考安装命令如下，其他平台应改用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 给出的命令：

```bash
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

然后安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

主训练支持 `--optimizer Adam|AdamW|SGD`，以及 `--scheduler ExponentialLR|CosineAnnealingLR|ReduceLROnPlateau|None`；未显式指定时沿用 parser 中的 `Adam` 和 `ExponentialLR`。

### Original baseline

原始脚本仍可用：

```bash
source ./train.sh
source ./test.sh
```

使用 parser defaults 的最小 baseline 命令为（它不复写 `train.sh` 中的 250 epochs、learning rate 和 validation schedule；需要复现实验脚本时请直接运行脚本或显式传入相同参数）：

```bash
python main.py \
  --dataset eth5 \
  --test_set eth \
  --phase train_test \
  --goal_model_type independent \
  --training_stage baseline \
  --use_wandb False
```

单独测试 baseline checkpoint：

```bash
python main.py \
  --dataset eth5 \
  --test_set eth \
  --phase test \
  --goal_model_type independent \
  --load_checkpoint best \
  --use_wandb False
```

### Recommended staged Full-Joint training

Stage 0：先复现 GDTS-Base，得到 `output/eth/saved_models/best_model.pt`。Stage 1 加载它并冻结 GDTS 主干，仅训练 joint-goal modules：

```bash
python main.py \
  --dataset eth5 \
  --test_set eth \
  --phase train \
  --goal_model_type joint \
  --training_stage joint \
  --pretrain_path output/eth/saved_models/best_model.pt \
  --data_augmentation False \
  --num_epochs 80 \
  --learning_rate 0.001 \
  --use_wandb False
```

Stage 2 从 Full-Joint 的 best checkpoint 恢复，并以较小 learning rate 联合 fine-tune：

```bash
python main.py \
  --dataset eth5 \
  --test_set eth \
  --phase train \
  --goal_model_type joint \
  --training_stage finetune \
  --load_checkpoint best \
  --pretrain_path '' \
  --force_reprocess False \
  --data_augmentation False \
  --num_epochs 250 \
  --learning_rate 0.0001 \
  --use_wandb False
```

测试 Full-Joint：

```bash
python main.py \
  --dataset eth5 \
  --test_set eth \
  --phase test \
  --goal_model_type joint \
  --load_checkpoint best \
  --data_augmentation False \
  --use_wandb False
```

`--pretrain_path` 用于以旧 baseline 权重初始化新模型，`--load_checkpoint` 用于从当前 `goal_model_type` 的输出目录恢复。配置按 `<save_dir>/config.yaml` 跨 pre-process/train/test 共用并持久化，CLI 覆盖已保存值；旧版 `config_<phase>.yaml` 会在首次对应 phase 运行时自动迁移。`phase/load_checkpoint/pretrain_path/force_reprocess` 是当前命令专属控制，不会从配置继承；Stage 2 中显式的空 `--pretrain_path` 与 `--force_reprocess False` 因而只是自说明写法。对于 SDD 和 inD，`test_set` 会自动设为与 `dataset` 相同。

当前有明确语义的 stage/type 组合是：GDTS-Base 使用 `independent + baseline`，structured Stage 1 使用 `{social,lowrank,energy,joint} + joint`，structured Stage 2 使用相同类型加 `finetune`。Parser 会拒绝 `independent + joint` 和 `structured + baseline`，避免悄然运行错误的训练路径；为保持原命令兼容，默认的 `independent + finetune` 仍按 legacy baseline 执行。

### Run the five ablations

以下命令使用相同数据划分，输出目录按模式自动隔离：

```bash
python main.py --dataset eth5 --test_set eth --phase train --goal_model_type independent --training_stage baseline --use_wandb False
python main.py --dataset eth5 --test_set eth --phase train --goal_model_type social      --training_stage finetune --data_augmentation False --use_wandb False
python main.py --dataset eth5 --test_set eth --phase train --goal_model_type lowrank     --training_stage finetune --data_augmentation False --use_wandb False
python main.py --dataset eth5 --test_set eth --phase train --goal_model_type energy      --training_stage finetune --data_augmentation False --use_wandb False
python main.py --dataset eth5 --test_set eth --phase train --goal_model_type joint       --training_stage finetune --data_augmentation False --use_wandb False
```

为公平比较，应显式保持相同的 seed、`num_samples`、epoch、optimizer、scheduler 和基础 checkpoint。上面是可直接启动的单阶段命令；若使用 staged protocol，应在每个 structured ablation 的首次训练命令中加入同一个 `--pretrain_path`。较大的实验建议通过单独的 `--phase pre-process` 命令创建/重建一次同步 cache，随后训练时保持 `--force_reprocess False`。

## Evaluation metrics

与 ICCV 2023 *Joint Metrics Matter* 表格中的 `ETH (1.4)` 做严格测试协议对齐时，请使用独立工具，不要修改或抽样现有训练 cache：

```bash
python tools/evaluate_jmm_official.py prepare-data
python tools/evaluate_jmm_official.py inspect
```

模型推理和指标计算被拆成两个显式步骤，便于先导出、后评测，并防止把非官方窗口误报成论文结果。完整用法、输出位置及可比性边界见 [`docs/jmm_official_evaluation.md`](docs/jmm_official_evaluation.md)。

所有轨迹输入遵循 `[samples,time,agents,2]`。除传统 marginal 指标外，joint 指标始终以同一个 sample index 评价 scene 内全部行人：

| Metric | Selection/evaluation rule |
|---|---|
| `ADE`, `FDE` | original per-agent best-of-K, pixel coordinates |
| `ADE_world`, `FDE_world` | original per-agent best-of-K, metres |
| `minADE@K`, `minFDE@K` | explicitly named marginal best-of-K metrics, metres |
| `JADE` | one shared sample minimizes mean future error over all scene agents/timesteps, metres |
| `JFDE` | one shared sample minimizes mean endpoint error over all scene agents, metres |
| `Goal_minFDE` | per-agent minimum endpoint error over the heatmap candidate set `K`, metres |
| `Goal_Recall@K` | whether any of the `K=num_goal_candidates` heatmap candidates is within the configured metric threshold |
| `Joint_Goal_Endpoint_Error` | best shared-branch mean endpoint error per scene, metres |
| `Joint_Goal_Compatibility` | best shared-branch relative endpoint/formation error per scene, metres |
| `Collision_Rate` | fraction of sample/pair future paths crossing the configured metric threshold |

`Collision_Rate` 和 `Goal_Recall@K` 不使用未经验证的统一阈值，只有显式给出下列参数时才会记录：

```text
--collision_threshold_meter <dataset-validated value in metres>
--goal_recall_threshold_meter <dataset-validated value in metres>
```

请根据数据预处理、行人 footprint 和论文 protocol 选择阈值，并在实验报告中同时披露。当前 collision metric 使用连续线段最近距离，包含“最后观测点→第一个预测点”的边界段，能够检测离散帧之间的 crossing。Goal-level candidate metrics 使用原始 `K` 候选；两个 Joint Goal 指标使用 `S=num_samples` 个 selected branches 并排除额外的 tree-trunk MAP goal。

`--best_metric auto` 在 GDTS-Base 中保留原 pixel ADE 选模，在四种 synchronized structured 模型中默认按 world-coordinate JADE 保存 `best_model.pt`；也可显式选择 `ADE|ADE_world|JADE|JFDE`。

## Complexity

显式 joint table 有 `K^N` 个状态。本实现只保存 scene modes、candidate marginals 和 sparse-edge low-rank factors：

```text
low-rank unary representation:     O(R*N*K)
pair-factor representation:        O(E*M*K*r_e)
one refinement round/sample:       O(E*M*K*r_e)
requested structured estimate:     R*N*K + L*E*K*r_e*M
```

总 inference work 还随 joint sample 数 `S` 线性增长，但绝不枚举 `K^N`，也不创建 `[N,N,K,K]`。`estimate_joint_goal_complexity(...)` 同时报告 `K^N`、其 `log10` 大小及 structured estimate。训练 pseudo-likelihood 可在 sparse edges 上构造 `[E,K,K]` effective energy，复杂度约为 `O(E*M*K^2*r_e)`；它仍与 agent-pair 全连接表和 full partition function 无关。

## Diagnostics, edge cases and visualization

实现显式处理：

- variable `N`、single-agent scene 和 `E=0`；
- all-isolated agents、invalid candidate fallback；
- relation/mode probability normalization；
- cross-scene edge rejection；
- NaN/Inf loss and sampling checks；
- canonical edge only once，避免 pairwise energy 双计数；
- permutation-invariant scene pooling 与 permutation-equivariant social encoding。

启用默认的 `--joint_diagnostics True` 时，训练进度会显示 edge count、average degree、mode/relation entropy 和 average pairwise energy 等紧凑诊断。

可视化工具读取一个同步 scene-window `.npz`，其中 `obs [N,T_obs,2]` 为必需项，其他支持项包括 `gt`、`independent_goals`、`independent_prob`、`joint_goals`、`predictions`、`edge_index`、`relation_prob`、`mode_prob` 和 `selected_mode`：

```bash
python tools/visualize_joint_goal.py path/to/scene_window.npz \
  --output-prefix output/figures/scene_window \
  --sample-index 0 \
  --coordinates world \
  --annotate-relations
```

工具输出 SVG、PDF、TIFF 和 PNG。`--coordinates` 只设置坐标轴单位标签，不执行坐标转换，因此 archive 中所有空间数组必须预先处于同一个坐标系。工具期望 `predictions [S,T_pred,N,2]` 只包含未来段；从模型的 `[S,20,N,2]` 输出导出时应使用 `predictions[:,8:]`。若希望显示 relation annotations，请同时提供与 `edge_index` 对齐的 `relation_prob [E,M]`。

## Tests

运行全部单元测试：

```bash
python -m pytest -q
```

测试覆盖 synchronized grouping、`N=1/E=0` graph、TTC gate、social permutation equivariance、soft/hard-ST relations、low-rank mode normalization、pairwise energy、joint sampler、legacy/joint metrics、formation-preserving feature、between-frame crossing detection、五种 model-type 集成、真实 U-Net/diffusion backbone 的 synthetic Full-Joint backward 和可视化输入校验。大多数快速集成测试使用 shape-compatible lightweight doubles；真实 backbone smoke 单独确保 goal U-Net、history LSTM、diffusion 和 joint modules 都获得 finite gradient。进行正式训练前还应在目标 dataset 的真实数据上执行 one-mini-batch forward/backward smoke test。

## Checkpoint compatibility

- `independent` 保持原模块名称、shape、cache 目录和 checkpoint 目录，并使用 strict loading。
- 四种 structured models 可通过 `--pretrain_path` 非严格加载 legacy GDTS checkpoint；missing keys 应仅来自新增模块，并会打印出来。
- `--load_checkpoint` 对 baseline 和 structured 当前 checkpoint 都严格加载；缺失任一 joint module key 会立即失败，不会以随机初始化模块继续测试。
- 旧 YAML 缺少新增参数时会保留 argparse defaults，不再因 key 集不完全相等而拒绝加载。
- 不要将一种非 baseline ablation 的 checkpoint 当作另一种 ablation 的完整 checkpoint；它们使用不同输出目录和不同启用模块。

## Known limitations

- 第一版有意保持 diffusion denoiser 不变；trajectory-level denoising 本身仍无 relation-aware graph layer，社会耦合主要发生在 goal selection。
- latent relation modes 无人工语义标签，co-walking、yielding、following 等解释必须通过训练后的统计和可视化验证。
- joint inference 只能在有限 heatmap candidate set 中选择，最终上限受 Goal Recall@K 和 TTST/k-means candidate quality 约束；candidate extraction 本身不可微。
- synchronized preprocessing 当前只保留完整 20-frame windows；不完整轨迹不会用于 joint-window training。
- SDD 与 inD 沿用上游 GDTS 的 `validate_on_test` protocol，validation/test scene 集相同，因此这两套数据上的 best-checkpoint 结果不能表述为独立 held-out model selection；需要无偏选择时应另划 validation 或预先固定 epoch。
- GDTS-Base 为保持原缓存完全兼容，仍使用会混合不同 window 的 legacy batches；因此代码不为 baseline 报告需要真实 scene grouping 的 JADE/JFDE/Collision 指标。公平的 joint-metric baseline 需要另做 synchronized independent-goal evaluation protocol。
- structured modes 当前禁用 legacy pixel augmentation；若要重新启用，augmentation 必须同步更新 scene homography，保证 graph、TTC 和 goal losses 仍处于正确的 world coordinates。
- pseudo-likelihood 规避了 full partition function，但训练时 sparse-edge `[E,K,K]` tensor 在很大的 `K` 或稠密 `full` graph 下仍可能占用显存。
- radius/TTC builder 当前先检查 scene 内 canonical pairs，graph construction 最坏仍为 `O(N^2)`；稀疏性主要降低后续 message passing、energy 和 refinement 的开销。
- collision 和 goal-recall 阈值必须由 dataset protocol 决定；仓库不提供未经验证的默认值。
- visualization 需要调用方导出 `.npz` diagnostics，训练默认不会批量保存大体积中间 tensor。
- checkpoint 目前保存 model weights 和 epoch；恢复训练时 optimizer/scheduler state 会重新初始化。
- 恢复训练也不会恢复历史 `best_metrics`/`best_metrics_epochs`；继续训练可能以新的、但不如旧 checkpoint 的 validation 结果覆盖 `best_model.pt`，重要实验应先备份旧 best checkpoint。

## Acknowledgement

Part of the code is borrowed from [Goal-SAR](https://github.com/luigifilippochiara/Goal-SAR) and [MID](https://github.com/Gutianpei/MID). We thank the authors for releasing their code.

## Citation

If you find the original GDTS code useful for your research, please cite:

```bibtex
@inproceedings{sun2025gdts,
    title={\uppercase{GDTS}: Goal-Guided Diffusion Model with Tree Sampling for Multi-Modal Pedestrian Trajectory Prediction},
    author={Sun, Ge and Wang, Sheng and Zhu, Lei and Liu, Ming and Ma, Jun},
    booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    pages={14595--14602},
    year={2025},
    organization={IEEE}
}
```
