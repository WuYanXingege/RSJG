# ZARA1 EBJD 训练曲线与达标前景分析

快照时间：2026-10-24 10:55:03 +0800  
训练源码提交：`6f86d31f7987f2e8c2fff802e12d8eb04e8d82da`  
运行：`zara1_full_wide192_rollout_off_seed3101_e250_v1`

## 结论

当前没有训练崩溃或指标计算异常。模型已经表现出明确的联合预测收益，
但按现有曲线，严格通过 matched-GDTS 的“边缘不退化”合同概率偏低，
主要瓶颈是 minFDE。更准确的表述是：**联合指标有效，边缘终点精度不足**，
而不是“网络完全无效”。

截至本快照，epoch 78 已完整保存，epoch 79 正在训练。训练完成进度为
78/250（31.2%），每轮约 12.7--13.1 分钟，日志估计剩余约 37.4 小时。
epoch 79 原始日志仍持续写入；所有已检查 loss、参数范数及中间量均为有限值。

## 同协议指标

以下均使用 ZARA1 validation manifest、seed 2035、20 worlds、20 DDIM
steps。数值越低越好。

| 模型/轮次 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| matched GDTS | **0.188109** | **0.285034** | 0.319607 | 0.586878 |
| EBJD epoch 67 | 0.206064 | 0.362873 | **0.295380** | **0.583195** |
| EBJD epoch 73 | 0.208737 | 0.364591 | **0.293987** | **0.575245** |
| EBJD epoch 78 | 0.211128 | 0.371612 | **0.292199** | **0.585946** |

epoch 67 是当前 selector 的 least-violation checkpoint。相对 GDTS：

- minADE 高 9.55%；
- minFDE 高 27.31%；
- JADE 低 7.58%；
- JFDE 低 0.63%。

因此同一 checkpoint 已经同时改善 JADE/JFDE，但没有满足 minADE/minFDE
不退化条件。当前 `best_eligible` 为空，不应将 `least_violation.pt` 解释成
合同通过。

当前各指标单项最好值来自不同 epoch，不能拼成一个模型：

| 指标 | 最好值 | epoch |
|---|---:|---:|
| minADE | 0.206064 | 67 |
| minFDE | 0.362873 | 67 |
| JADE | 0.287469 | 76 |
| JFDE | 0.575245 | 73 |

## 曲线趋势

| 区间均值 | loss | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|---:|
| epoch 1--10 | 1.833657 | 0.243953 | 0.432022 | 0.351500 | 0.695020 |
| epoch 41--50 | 1.359860 | 0.215437 | 0.381819 | 0.300536 | 0.598259 |
| epoch 61--70 | 1.329033 | 0.213279 | 0.380041 | 0.296856 | 0.592689 |
| epoch 69--78 | 1.321211 | 0.211528 | 0.375497 | 0.292941 | 0.584606 |

训练确实有效，但 epoch 40 后边缘指标进入缓慢改善区。epoch 41--50 到
epoch 69--78，minADE 仅改善约 1.8%，minFDE 仅改善约 1.7%。从目前最佳
值追平 GDTS，minADE 仍需降低约 8.7%，minFDE 仍需降低约 21.5%。

当前累计学习率面积约为 `0.007085`，剩余约 `0.005538`，即剩余优化预算
约为已用预算的 78.2%；尚有学习空间，但后半程学习率会继续余弦衰减。

按最近 20--40 个 epoch 与剩余学习率面积做保守外推，训练后期的合理量级
约为：

- minADE：0.196--0.202；
- minFDE：0.345--0.366；
- JADE：0.264--0.276；
- JFDE：0.536--0.551。

这不是置信区间，也不能替代完整训练和 selector；它仅用于判断当前曲线
是否有足够斜率追平基线。minADE 仍可能进一步接近 GDTS，但 minFDE 在不
出现新的训练阶段跃迁时追到 0.285 的可能性较低。

## 原因判断

当前正式合同全程关闭 rollout 与 paired actual-step projection。原始日志中：

- `rollout = 0`；
- `soft_marginal_ade = 0`；
- `soft_marginal_fde = 0`；
- `projection_active = 0`。

训练主要优化扩散去噪及几何/地图辅助目标，而评测是 20 步 DDIM、20 worlds
下的轨迹选择。训练 loss 下降不保证 minADE/minFDE 同比例下降。当前结果与
这一目标错位一致：联合场景质量持续改善，单人候选覆盖与终点精度仍弱于
matched GDTS。

## 决策边界

1. 不应因当前快照提前宣告失败；250 轮完整曲线仍有必要保留。
2. 若唯一目标是严格通过边缘不退化合同，当前前景不乐观，minFDE 是主要
   阻断项。
3. 可在 epoch 120--150 做一次不改变训练的曲线复核。如果届时历史最好
   minFDE 仍明显高于约 0.31--0.32，仅依靠当前 rollout-OFF 合同追平
   0.285 将缺乏曲线支持。
4. 当前验证使用单个固定推理 seed，且 validation/test 来自镜像官方包；
   不能把本快照当作最终独立测试或五推理种子结论。

## 绑定信息

- `last.pt`：epoch 78，SHA256
  `b41362c5b696253d47a291cc7399baa338dddcf2ef8466e6b664aee42c2c928b`。
- `least_violation.pt`：epoch 67，SHA256
  `84bc34c9da80020512e38566e7e3a00ee461dc2649c906097a2e7618d1803f8e`。
- matched GDTS checkpoint SHA256：
  `34af0f4335de77cf2b6ecd48b93262dfc4764e679bd52fd677392a85c22bf6d7`。
- validation manifest SHA256：
  `9c12e30ac17e91bbf8efa9e23a3d61a16c9e9c8dfc6414871fc7431b76379478`。

本归档只记录训练状态与前景分析，没有修改训练、配置、checkpoint 或
selector。
