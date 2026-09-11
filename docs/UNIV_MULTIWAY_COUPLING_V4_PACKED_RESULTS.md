# UNIV Multiway Coupling V4 Results

> 本文件由训练产物自动生成。模型选择只使用内部验证，held-out test 仅在 best checkpoint 确定后评测。

## 1. 最佳模型与协议

- selection metric：`JADE`
- best epoch：**12**
- model selection split：`internal_train`
- upstream generator：`stage0_independent`
- upstream checkpoint：`output/univ/saved_models/epoch_100.pt`
- upstream frozen：是
- internal validation strategy：`source_block`
- internal validation source：`data/eth5/univ/train/uni_examples.txt`
- train / valid / final-test windows：2982 / 320 / 947
- final test split：`heldout_test`

## 2. Best internal-validation 指标

| 指标 | Raw | Aligned | Aligned - Raw |
|---|---:|---:|---:|
| minADE | 0.26705 | 0.26705 | 0.00000 |
| minFDE | 0.45321 | 0.45321 | 0.00000 |
| JADE | 0.35434 | 0.34092 | -0.01342 |
| JFDE | 0.64951 | 0.62180 | -0.02771 |

- Marginal preservation check：**通过**
- max minADE absolute gap：0.00000
- max minFDE absolute gap：0.00000
- alignment oracle：0.75449
- alignment headroom：0.27224
- recovery ratio：0.02903

## 3. Coupling 行为诊断

- changed fraction：0.09659
- keep confidence：0.74609
- mean pair gate：0.67813
- relation entropy：1.33288
- permutation entropy：0.35985
- Sinkhorn row / col error：0.00405 / 0.00000
- soft-hard gap：0.87282
- Hungarian / greedy JADE：0.34092 / 0.34081
- Hungarian - greedy JADE：0.00011
- average N / edges / degree：1.94 / 1.37 / 1.32
- maximum component size：4

### 按 scene N 分桶

| 分桶 | count | Raw_JADE | JADE | Raw_JFDE | JFDE | recovery_ratio |
|---|---:|---:|---:|---:|---:|---:|
| N=1 | 132 | 0.26538 | 0.26538 | 0.44931 | 0.44931 | 0.00000 |
| N=2 | 114 | 0.38118 | 0.35849 | 0.71664 | 0.66322 | 0.03050 |
| N=3~5 | 74 | 0.47169 | 0.44861 | 0.90321 | 0.86567 | 0.07857 |

### 按 connected-component size 分桶

| 分桶 | count | Raw_JADE | JADE | Raw_JFDE | JFDE | recovery_ratio |
|---|---:|---:|---:|---:|---:|---:|
| comp_size=1 | 143 | 0.30324 | 0.30324 | 0.53848 | 0.53848 | 0.00000 |
| comp_size=2 | 118 | 0.37225 | 0.35057 | 0.69553 | 0.64167 | 0.02898 |
| comp_size=3~5 | 68 | 0.44857 | 0.42578 | 0.85106 | 0.81343 | 0.07422 |

## 4. 结论与下一步

- 大 component 是否仍显著失效：**当前未观察到**
- Stage1 upstream：建议保留 Stage1 upstream 作为消融，但主结果仍应先采用更干净的 Stage0 strong-marginal + V4 coupling；只有同协议 Stage1+V4 明显更优时再切换。
- 最值得做的下一步消融：优先在完全相同的 20 个 seeds 和内部验证划分上比较 Stage0/Stage1 upstream × no-coupler/V4；随后比较 lowrank/MLP energy 与 trajectory-conditioned relation on/off。

## 5. Held-out final test

- checkpoint：`best` (epoch 12)

- test_ADE: 12.88868
- test_ADE_world: 0.28714
- test_Aligned_minADE: 0.28714
- test_Aligned_minFDE: 0.52057
- test_FDE: 23.36664
- test_FDE_world: 0.52057
- test_JADE: 0.65467
- test_JFDE: 1.31784
- test_Raw_JADE: 0.65580
- test_Raw_JFDE: 1.32055
- test_Raw_minADE: 0.28714
- test_Raw_minFDE: 0.52057
- test_alignment_headroom: 1.14315
- test_alignment_oracle: 0.83761
- test_alignment_recovery_ratio: 0.00369
- test_avg_degree: 16.56683
- test_avg_num_agents: 25.69588
- test_avg_num_edges: 244.44245
- test_changed_fraction: 0.00292
- test_delta_JADE: 0.00113
- test_delta_JFDE: 0.00271
- test_greedy_JADE: 0.65462
- test_hungarian_minus_greedy_JADE: 0.00005
- test_keep_confidence: 0.99733
- test_marginal_ADE_max_abs_error: 0.00000
- test_marginal_FDE_max_abs_error: 0.00000
- test_max_component_size: 25.81204
- test_mean_pair_gate: 0.61742
- test_minADE@K: 0.28714
- test_minFDE@K: 0.52057
- test_permutation_entropy: 0.83884
- test_relation_entropy: 1.32489
- test_sinkhorn_col_error: 0.00000
- test_sinkhorn_row_error: 0.00007
- test_soft_hard_gap: 0.37497
