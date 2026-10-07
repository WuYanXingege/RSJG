# Conditional next design

## 当前决定

正式 E_joint 与机制结果未齐，**本轮不选定或执行新实验**。当前只有一个条件性推荐：

> 若最终 A1 不过门槛、A0 或 A1 的 full 明显优于 E_joint baseline、
> interaction-off 证明不可约 pair term 有贡献，并且 native probability 进一步确认
> `H_cond` 与 `H_agg` 同时低，则下一轮只做 **M=4 → M=1 的 A0 重训练**。

它检验“四个无监督关系类别是否是多余自由度”，保留实际 pair interaction，不与
learning rate、候选、sampler、epoch、精度或 loss 正则一起改变。若任一前提不成立，
不做 M=1：整体模块无收益时先处理 surrogate/推理错配；interaction-off 无差异时先
处理 additive cost；A1 过门槛则保留 A1。最多三个分支，没有把当前趋势写成已验证设计。

## Current architecture（运行源码 `664ad041…`）

| 模块 | 类型与层 | 输入→输出 | 训练角色 |
|---|---|---|---|
| frozen GDTS | goal CNN + diffusion/denoiser | obs8 → 21 endpoint candidates；selected goals → pred12 trajectories | 全冻结；只提供 bank 与轨迹 |
| social | `Linear(4,128)-ReLU-GRU(128)`；1 层 sparse message：`270→128→128`、update `256→128→128`，residual+LayerNorm，dropout0 | `[N,8,4]→[N,128]`，edge `[2,E]`/feat14 | train |
| base relation | shared bidirectional MLP `270→128→128→4`, ReLU | `[E,270]→[E,4]` | train；obs-only prior |
| dynamic relation | geometry4 `4→32→16`, SiLU；query `[4,16]`+bias4 | `[E,21,21,4]` probabilities | train；relation embedding64 在当前 Stage-A dormant |
| future teacher | MLP `6→64→64→4`, SiLU | future descriptor `[E,6]→q[E,4]` | train-only；部署不读取 future |
| unary | goal `4→64→64+LN`，agent `128→64+LN`，concat/product `192→64→1` | `[N,21]` score | train；加 frozen log prior |
| pair energy | candidate `4→64→64`；pair `270→128→64+LN`；relation embedding `[4,64]`；factor `64→8` | energy `[E,4,21,21]`，effective `−logsumexp(log p−E)` | train，rank8 |
| objective | `0.5 PL_post + 0.5 PL_prior + beta KL` | scene/window scalar | A0 mean-energy；A1 S=4 conditional expectation |
| sampler | weighted Gumbel Round0 + 2 exact lexicographic persistent-tie refinements | K=21 → P=20 joint worlds | 固定，不反传 |
| corrector | state21→64→64、time32→64→128、gate64→32→1、value64→64→64、output64→64→2 | diffusion epsilon residual | Stage-A 全冻结，末层严格零；eval literal bypass |

Stage-A 注册参数 331,845，active 331,781。没有 attention heads、Transformer、额外
dropout 或 scene latent；`strict_no_z` 的 scene mode 数为 0。

## Proposed M=1 variant（仅条件性提案）

| 改动 | 精确定义 |
|---|---|
| base relation | 删除 `270→128→128→4` MLP；令唯一关系概率恒为 1 |
| dynamic relation | 删除 geometry/query/bias/四类 embedding；不再生成 `[E,K,K,4]` prob |
| teacher/KL | 删除 `6→64→64→4` teacher 和 relation KL；future 不再进入 Stage-A objective |
| pair energy | relation embedding 从 `[4,64]` 改 `[1,64]`；其余 candidate/pair/factor 层和 rank8 不变；直接得到 `[E,21,21]` cost |
| objective | 使用 A0 的单一 prior pseudo-likelihood，系数 1.0；不添加 entropy、scale 或额外 loss |
| sampler/inference | Round0、P20、K21、两轮 refinement、persistent keys、坐标与 frozen GDTS 全不变 |

预计 active 参数由 331,781 降到约 **274,249**（减少 57,532，约 17.3%）；这是按删除
base relation 51,716、dynamic relation 820、teacher 4,868，并将 energy embedding
减少 192 计算的静态估计。实现前须由代码级参数表重新认证。推理减少 relation MLP、
geometry relation logits 和四模式 logsumexp；本轮未测 wall-clock，不能宣称速度倍数。

## Pipeline 与梯度边界

```python
# proposal only; names marked TODO do not yet exist
with no_grad():
    candidates, frozen_log_prior, frozen_trajectories = frozen_gdts(obs)  # K=21
    edge_index, edge_feat = fixed_graph(obs)

h = social_encoder(obs, edge_index, edge_feat)            # trainable
u = unary(h, candidates, frozen_log_prior)                 # trainable
C = single_relation_energy(h, candidates, edge_feat)       # trainable, [E,K,K]
loss = prior_pseudolikelihood(u, C, train_target_ids)       # gradients to h/u/C

# deployment only; discrete choices are intentionally outside the loss graph
with no_grad():
    round0 = fixed_weighted_gumbel(u, persistent_rng)
    worlds = fixed_two_round_solver(u, C, round0, persistent_keys)  # P=20
    prediction = frozen_gdts_trajectory_lookup(worlds)
```

没有声称通过不可微 solver 或 min-over-worlds 直接优化 JADE。teacher、GT future 仅用于
既有 candidate target construction 的训练角色，部署分支严格 obs-only；若实现中不能
满足该 role gate，实验不得启动。

## 最小公平协议

- 对照：同一正式 E_joint baseline、最终实际有效的 M=4 A0、M=1 A0；
- 初始化：按 train seed 3101–3103 从相同 parent/candidate cache 派生，共享 retained
  tensors；新增/缩减 embedding 的初始化规则先冻结；
- 固定：FP32 loss/BF16 policy、Adam `1e-4`、gamma .995、40 epochs、K21/P20、两轮、
  train/eval seeds、checkpoint 选择、预算；
- 评价：先在每个训练 seed 内平均 2035–2039，再做三对；同时报告 ADE/FDE、JADE/JFDE、
  CRmean、candidate IDs 与计算成本；
- go/no-go：只有上面的四个条件全部由当前正式结果满足才预注册启动。该设计尚未做
  文献新颖性检索，不能声称创新或 SOTA。
