# ETH Multiway Coupling V4 Results

> 本文件由训练产物自动生成。模型选择只使用内部验证，held-out test 仅在 best checkpoint 确定后评测。

## 1. 最佳模型与协议

- selection metric：`JADE`
- best epoch：**1**
- model selection split：`internal_train`
- upstream generator：`stage0_independent`
- upstream checkpoint：`output/eth/saved_models/epoch_100.pt`
- upstream frozen：是
- internal validation strategy：`source_block`
- internal validation source：`data/eth5/eth/train/uni_examples.txt`
- train / valid / final-test windows：3790 / 320 / 139
- final test split：`heldout_test`

## 2. Best internal-validation 指标

| 指标 | Raw | Aligned | Aligned - Raw |
|---|---:|---:|---:|
| minADE | 0.28100 | 0.28100 | 0.00000 |
| minFDE | 0.50082 | 0.50082 | 0.00000 |
| JADE | 0.39146 | 0.37008 | -0.02138 |
| JFDE | 0.76173 | 0.71593 | -0.04580 |

- Marginal preservation check：**通过**
- max minADE absolute gap：0.00000
- max minFDE absolute gap：0.00000
- alignment oracle：0.84790
- alignment headroom：0.33181
- recovery ratio：0.08515

## 3. Coupling 行为诊断

- changed fraction：0.10554
- keep confidence：0.70669
- mean pair gate：0.95708
- relation entropy：1.36874
- permutation entropy：0.34583
- Sinkhorn row / col error：0.01562 / 0.00000
- soft-hard gap：0.98550
- Hungarian / greedy JADE：0.37008 / 0.37184
- Hungarian - greedy JADE：-0.00176
- average N / edges / degree：1.94 / 1.37 / 1.35
- maximum component size：4

### 按 scene N 分桶

| 分桶 | count | Raw_JADE | JADE | Raw_JFDE | JFDE | recovery_ratio |
|---|---:|---:|---:|---:|---:|---:|
| N=1 | 132 | 0.30096 | 0.30096 | 0.57004 | 0.57004 | 0.00000 |
| N=2 | 114 | 0.41301 | 0.37563 | 0.81839 | 0.73694 | 0.16660 |
| N=3~5 | 74 | 0.51971 | 0.48483 | 1.01639 | 0.94382 | 0.11154 |

### 按 connected-component size 分桶

| 分桶 | count | Raw_JADE | JADE | Raw_JFDE | JFDE | recovery_ratio |
|---|---:|---:|---:|---:|---:|---:|
| comp_size=1 | 143 | 0.33957 | 0.33957 | 0.67798 | 0.67798 | 0.00000 |
| comp_size=2 | 118 | 0.41109 | 0.36940 | 0.81254 | 0.72347 | 0.17962 |
| comp_size=3~5 | 68 | 0.48969 | 0.45910 | 0.94178 | 0.87869 | 0.10219 |

## 4. 结论与下一步

- 大 component 是否仍显著失效：**当前未观察到**
- Stage1 upstream：建议保留 Stage1 upstream 作为消融，但主结果仍应先采用更干净的 Stage0 strong-marginal + V4 coupling；只有同协议 Stage1+V4 明显更优时再切换。
- 最值得做的下一步消融：优先在完全相同的 20 个 seeds 和内部验证划分上比较 Stage0/Stage1 upstream × no-coupler/V4；随后比较 lowrank/MLP energy 与 trajectory-conditioned relation on/off。

## 5. Held-out final test

- checkpoint：`best` (epoch 1)

- test_ADE: 7.11460
- test_ADE_world: 0.31386
- test_Aligned_minADE: 0.31386
- test_Aligned_minFDE: 0.44956
- test_FDE: 9.98288
- test_FDE_world: 0.44956
- test_JADE: 0.45973
- test_JFDE: 0.76780
- test_Raw_JADE: 0.50811
- test_Raw_JFDE: 0.89124
- test_Raw_minADE: 0.31386
- test_Raw_minFDE: 0.44956
- test_alignment_headroom: 0.60164
- test_alignment_oracle: 0.82322
- test_alignment_recovery_ratio: -64195.78261
- test_avg_degree: 1.81296
- test_avg_num_agents: 2.64748
- test_avg_num_edges: 3.10072
- test_changed_fraction: 0.12713
- test_delta_JADE: 0.04837
- test_delta_JFDE: 0.12344
- test_greedy_JADE: 0.46104
- test_hungarian_minus_greedy_JADE: -0.00130
- test_keep_confidence: 0.66197
- test_marginal_ADE_max_abs_error: 0.00000
- test_marginal_FDE_max_abs_error: 0.00000
- test_max_component_size: 3.53237
- test_mean_pair_gate: 0.83927
- test_minADE@K: 0.31386
- test_minFDE@K: 0.44956
- test_permutation_entropy: 0.32259
- test_relation_entropy: 1.19867
- test_sinkhorn_col_error: 0.00000
- test_sinkhorn_row_error: 0.01280
- test_soft_hard_gap: 1.07438
