# ZARA1 EBJD epoch 109 训练状态更新

快照时间：2026-10-24 17:24:53 +0800  
训练源码提交：`6f86d31f7987f2e8c2fff802e12d8eb04e8d82da`  
运行：`zara1_full_wide192_rollout_off_seed3101_e250_v1`

## 当前状态

- epoch 109 已完整训练、验证并保存；epoch 110 正在运行；
- 完成 109/250（43.6%），当前 global update 29464；
- 每轮近期约 12.77 分钟，日志 ETA 约 30.53 小时；
- 原始日志持续写入，loss、参数范数及已检查中间量均为有限值；
- rollout、soft marginal loss 和 paired projection 仍全部为 0，符合冻结合同；
- 控制台顶部的 startup mismatch traceback 属于旧启动记录，不是当前故障。

## 与 matched GDTS 的同协议比较

相同 ZARA1 validation manifest、推理 seed 2035、20 worlds、20 DDIM
steps；数值越低越好。

| 模型/checkpoint | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| matched GDTS | **0.188109** | **0.285034** | 0.319607 | 0.586878 |
| EBJD epoch 79（联合指标最好） | 0.207063 | 0.367765 | **0.283743** | **0.561985** |
| EBJD epoch 96（当前 selector） | 0.205477 | 0.359933 | **0.287166** | **0.574830** |
| EBJD epoch 109（最新完整） | 0.207832 | 0.376445 | **0.282994** | **0.570151** |

当前 selector 为 epoch 96。相对 matched GDTS：

- minADE 高 9.23%；
- minFDE 高 26.28%；
- JADE 低 10.15%；
- JFDE 低 2.05%。

`best_eligible` 仍为空，严格的边缘不退化合同尚未通过。epoch 79 同一
checkpoint 的 JADE/JFDE 分别比 GDTS 低 11.22% 和 4.24%，是当前最清晰
的联合预测收益证据，但其边缘指标仍退化。

单项最好值来自不同轮次，不能拼成一个模型：

| 指标 | 最好值 | epoch |
|---|---:|---:|
| minADE | 0.205456 | 85 |
| minFDE | 0.359933 | 96 |
| JADE | 0.282994 | 109 |
| JFDE | 0.561985 | 79 |

## 曲线趋势

| 区间均值 | loss | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|---:|
| epoch 1--10 | 1.833657 | 0.243953 | 0.432022 | 0.351500 | 0.695020 |
| epoch 71--80 | 1.317843 | 0.210574 | 0.373585 | 0.291620 | 0.581031 |
| epoch 81--90 | 1.292758 | 0.210571 | 0.376094 | 0.290544 | 0.582500 |
| epoch 91--100 | 1.277962 | 0.209073 | 0.370733 | 0.290152 | 0.580392 |
| epoch 100--109 | 1.269709 | 0.209398 | 0.372340 | 0.289301 | 0.578164 |

loss 仍在下降，训练稳定；联合指标继续保持优势，epoch 109 刷新 JADE。
但是 minADE/minFDE 在最近约 40 轮内主要围绕平台波动。从 epoch 67 到
epoch 96，29 轮仅把 minADE 从 0.206064 改善到 0.205477，将 minFDE 从
0.362873 改善到 0.359933，边缘改善速度仍不足以支持追平 GDTS 的判断。

截至 epoch 109，剩余累计学习率面积约为已用部分的 35.9%。基于最近
20--40 轮与剩余学习率面积的外推范围约为：

- minADE：0.204--0.208；
- minFDE：0.356--0.367；
- JADE：0.277--0.289；
- JFDE：0.540--0.571。

这不是置信区间，也不能替代完整训练。当前证据支持“联合指标稳定受益”，
但严格边缘不退化合同的通过前景依然偏低，主要瓶颈仍是 minFDE。

## 绑定与证据边界

- `last.pt`：epoch 109，SHA256
  `b83258a5cb4ea0a507f76e8a468727f2a031a55ff43d628df56c60e9892b453e`；
- `least_violation.pt`：epoch 96，SHA256
  `2b6765349e2f87dd86acdbf4431436aa931553378c64f982fbe57e1b6bc3e3cd`；
- matched GDTS checkpoint SHA256：
  `34af0f4335de77cf2b6ecd48b93262dfc4764e679bd52fd677392a85c22bf6d7`；
- validation manifest SHA256：
  `9c12e30ac17e91bbf8efa9e23a3d61a16c9e9c8dfc6414871fc7431b76379478`。

当前仍是一个训练 seed、一个固定验证推理 seed，且 validation/test 为镜像
官方包的内部设置；不能把本快照当作独立测试或五推理种子最终结论。

本次仅归档状态和分析，没有修改训练、配置、checkpoint 或 selector。
