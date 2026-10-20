# NO_TOKEN 固定锚点短评估

## 结论

- 候选筛选：**PASS**。ρ=0 相对 Raw 四项均值均不退化，且有改善。
- 原合同：**FAIL**。联合指标优于 matched GDTS，但边缘 minADE/minFDE 仍退化。
- 本轮仅改变冻结模型输出；训练 seed=3101，optimizer updates=0，五个推理 seed 不是独立训练重复，`independent_test=false`。

## 五种子结果（m，均值 ± 样本标准差）

| 输出 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| NO_TOKEN Raw | 0.4330 ± 0.0026 | 0.5906 ± 0.0162 | 0.4882 ± 0.0014 | 0.7265 ± 0.0113 |
| NO_TOKEN Fixed-anchor | 0.4322 ± 0.0051 | 0.5789 ± 0.0092 | 0.4862 ± 0.0035 | 0.7077 ± 0.0081 |
| matched GDTS | 0.4158 ± 0.0052 | 0.5771 ± 0.0101 | 0.4927 ± 0.0048 | 0.7492 ± 0.0086 |

Fixed-anchor − Raw：minADE -0.000833, minFDE -0.011660, JADE -0.001960, JFDE -0.018817。逐 seed 改善数：minADE 3/5, minFDE 4/5, JADE 3/5, JFDE 5/5。

Fixed-anchor − matched GDTS：minADE +0.016463, minFDE +0.001809, JADE -0.006494, JFDE -0.041492。

## 身份、几何与成本

checkpoint 与 manifest SHA256、seeds 2035–2039、P20、20 DDIM steps、eta0、FP32 均匹配。Raw bank 回放最大误差 `1.643e-08`；固定终点误差 `2.384e-07`；线性修正误差 `9.537e-07`；桥残差误差 `2.861e-06`；实际锚点集合与 native 候选集合双向误差 `0.000e+00`。ρ=0 minFDE 与 native oracle 最大误差 `3.684e-10`。所有值有限，253 scenes / 364 agent occurrences 完整。

正式 GPU 推理墙钟 `406.43` 秒；离线 CPU 核验约 `29.56` 秒。一次缺少 ablation 身份的启动在建模前被拒绝，未发生推理或更新。完整 bank 和日志仅留本地。

## 命令与边界

使用既有 `python -m ebjd.evaluate --ablation v2_no_token --endpoint-projection-radius 0 --prediction-bank-root ...`，随后由 `scripts/finalize_no_token_fixed_anchor.py` 对 Raw/Fixed bank 做成对核验。ρ=0 不改写上一轮主结果，也不保证零容忍边缘合同。

唯一下一步：保留 Fixed-anchor 为候选输出变体；若继续研究，只定位 native anchor 相对 GDTS 约 0.00181 m 的 minFDE 缺口来源，不调半径、不续训、不启动 rollout。
