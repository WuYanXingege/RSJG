# Mechanism intervention review

## 当前状态

三项机制结果均为 `PENDING`。本文件只核对固定源码中的实际干预位置，不将配置名
当成已执行证据，也不宣称 GPU observer neutrality。

## 静态入口

真实 local score 是：

```text
conditional_score = unary_score - accumulated_pair_cost
```

干预位于 `src/models/joint_dependency_v2/joint_sampler.py:450–522` 的 deployable
两轮 refinement 内，不是 legacy `energy_weight`。

| 干预 | 源码实际行为 | 静态结论 | 仍缺证据 |
|---|---|---|---|
| `off` | source/destination effective pair cost 均 `zeros_like` | pair cost 真实入口被关闭 | 正式 IDs、metrics、共享随机输入回读 |
| `physical` | 用两端最后观测到 endpoint 的归一时间直线，取 `[0,1]` 最近点，返回 `relu(1−d/0.4)^2` | 4.8 s 直线由 pred12×0.4 s 隐含；阈值和公式匹配 | 源码未出现独立 train-calibration scale；若预注册要求非单位尺度，则当前只认证实际 raw scale=1 路径 |
| `pure_interaction_off` | 对完整 K×K cost 做 uniform row/column mean，保留 `mean+row+column`，再 gather 邻居 candidate | 实际只去 double-centered interaction I，未删除整个 energy | 正式 full/干预 candidate IDs、metrics 与 native cost tensor |

`pure_interaction_off` 的实现中，`row=C.mean(l)−mu`、`column=C.mean(k)−mu`、
`additive=mu+row+column`，与保留 `a+b` 的 gauge 等价。之后仍使用原 unary、Round0、
persistent keys 和 solver。

## 因果边界

- 正式结果不存在，无法判断 full、pair-off、physical、interaction-off 的指标差。
- 没有 native candidate IDs/noise keys，无法证实本次实际运行的底噪、候选和
  Round0→Round2 身份完全共享。
- 没有部署 prior cost 矩阵，无法独立重算 I/a/b；静态代码正确不等于干预已在
  GPU run 中按预期观察。
- pair-off 有差异也只说明 pair score 入口重要；只有 interaction-off 与 full 的
  同协议差异才能支持不可约 interaction 的机制归因。
- physical 结果即使改善，也只是固定直线 surrogate 的受控替代，不代表真实轨迹避碰。

正式结果到达后，CPU 汇总器会把三份文件纳入 provenance；若 native evidence
仍缺，结论必须保持“受控配置结果”而不是“完整生产路径因果认证”。
