# Training dynamics

## Figure contract

- **结论**：surrogate loss 持续下降，但 validation JADE 不单调；关系分布同时高度集中。
- **问题**：优化进展是否与模型选择指标对齐，关系集中是否伴随该错位？
- **图型**：2×2 quantitative grid；每条线是一条训练臂，不汇总为独立重复。
- **panel 角色**：a，优化目标；b，JADE 与被选 epoch；c，candidate-conditioned
  relation 条件熵；d，日志聚合 mode mass 的最大占比。
- **数据完整性**：使用所有完整 epoch；A0_3103 明标 partial；A1_3103 无数据故不画。
- **尺寸/输出**：Python/matplotlib，180×122 mm；editable SVG、PDF、600-dpi
  LZW TIFF 和 PNG preview。

[SVG](figures/training_dynamics.svg) · [PDF](figures/training_dynamics.pdf) ·
[TIFF](figures/training_dynamics.tiff) · [source CSV](TRAINING_CURVES.csv)

## 数值摘要

| 臂 | epoch 范围 | loss start→end | dynamic entropy start→end | dominant fraction start→end | energy std start→end |
|---|---:|---:|---:|---:|---:|
| A0_3101 | 1–40 | 3.0310→2.8711 | 1.1344→0.0135 | 0.397→0.998 | 0.102→1.341 |
| A1_3101 | 1–40 | 3.0313→2.8729 | 1.1291→0.0194 | 0.401→0.997 | 0.101→1.405 |
| A1_3102 | 1–40 | 3.0304→2.8724 | 1.0281→0.0140 | 0.467→0.998 | 0.097→1.249 |
| A0_3102 | 1–40 | 3.0298→2.8711 | 1.0117→0.0102 | 0.480→0.998 | 0.100→1.236 |
| A0_3103 | 1–31 | 3.0326→2.8798 | 1.0874→0.0393 | 0.444→0.993 | 0.071→1.313 |

熵单位为 nat；图中除以 `log(4)`。dominant fraction 是四个日志 usage 值除以其
和后取最大值。它纠正 E=0 window 造成的总质量不足，但不是用保存的 edge-level
概率重新计算的严格 `H_agg`。

四个完成臂的 relation KL 到 epoch 40 降至 `0.000866–0.001027`。这只说明 prior
与 teacher 接近；二者共同训练时，不能由小 KL 推断 teacher 正确或 mode 有真实语义。

`Goal_minFDE=0.1793821013` 在所有可见 epoch 恒定，符合固定 candidate bank 的定义；
它不是最终 trajectory FDE，也不是所选 20 worlds 的独立 oracle 证明。

## QA

- source preflight：21 PASS、0 WARN、0 FAIL；
- panel alignment：strict PASS，见
  [alignment JSON](figures/training_dynamics.alignment.json)；
- PDF glyph floor：最小 6.8 pt，高于 5 pt；
- rendered collision audit：0 FAIL、0 WARN，见
  [collision JSON](figures/training_dynamics.collision-audit.json)；
- 原尺寸 PNG 已人工目视检查：panel 标签、轴、legend、selected markers 均可辨，
  无裁切或遮挡。

绘图代码为 [plot_training_dynamics.py](plot_training_dynamics.py)。
