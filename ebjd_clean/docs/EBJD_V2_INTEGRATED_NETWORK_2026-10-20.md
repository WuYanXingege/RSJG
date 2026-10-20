# EBJD V2 集成网络：ETH 固定短程实验

## 状态

- 工程：COMPLETE；CPU/CUDA 接线、按名称迁移和控制复用均通过。
- 训练与五种子评估：COMPLETE；比较固定 epoch25，不按中间曲线选模。
- 原合同：**FAIL**。五个 inference seeds 共享同一训练 seed3101，`independent_test=false`。

## 真实数据流

```text
历史8帧 → GRU/social → 8历史+4运动摘要 ┐
                                       ├→ 12/16分组cross-attention → 3 coarse blocks
14通道crop → 可训练U-Net → 4×4地图16 ┘                         │
冻结GDTS/TTST20 → 每人20候选 → GT盲一一配对 → world锚点       │
高斯latent + 整组候选 + world锚点 → 锚点终点残差/桥变量 ←──────┘
粗物理轨迹 → 路径地图采样 + 物理关系 → Jacobian时间到latent汇总
          → 3 refinement blocks → 20步DDIM → P20联合场景
```

Provider 只读冻结；训练 P1 模式标签由训练 GT 最近候选产生，推理 P20 不读取未来 GT。

## 模块与维度

| 模块 | V2_FULL | NO_TOKEN | NO_ANCHOR |
|---|---:|---:|---:|
| motion/map memory | 12 + 16 | 8 + 64 | 12 + 16 |
| context attention | split + shared 384→64→2 gate | legacy joint | split + shared gate |
| goal latent | `(g-A)/s_g` | `(g-A)/s_g` | CV-centered legacy |
| joint pairing / assoc | ON / 0.05 | ON / 0.05 | OFF / 0 |
| keep loss | 0.01 | 0.01 | 0 |
| geometry map | decoder Jacobian | decoder Jacobian | decoder Jacobian |
| formal rollout / projection | OFF / OFF | OFF / OFF | OFF / OFF |

## 五种子结果（m，均值 ± 样本标准差）

| 模型 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| parent_epoch20 | 0.5105 ± 0.0050 | 0.8813 ± 0.0080 | 0.5766 ± 0.0071 | 1.0055 ± 0.0116 |
| C | 0.4745 ± 0.0061 | 0.7142 ± 0.0189 | 0.5483 ± 0.0090 | 0.8902 ± 0.0182 |
| V2_FULL | 0.4401 ± 0.0044 | 0.5977 ± 0.0164 | 0.4954 ± 0.0039 | 0.7345 ± 0.0142 |
| V2_NO_TOKEN | 0.4330 ± 0.0026 | 0.5906 ± 0.0162 | 0.4882 ± 0.0014 | 0.7265 ± 0.0113 |
| V2_NO_ANCHOR | 0.4786 ± 0.0055 | 0.7271 ± 0.0159 | 0.5519 ± 0.0080 | 0.9019 ± 0.0117 |
| matched_GDTS | 0.4158 ± 0.0052 | 0.5771 ± 0.0101 | 0.4927 ± 0.0048 | 0.7492 ± 0.0086 |

所有逐 seed 差值、人数分层、K 曲线、endpoint P90/P95、单候选 FDE、中心误差和中心化 RMS 均在 `RESULTS_AND_DIAGNOSTICS.json` 中。`best_*` 未跨 epoch 拼接。

## 执行速度审计

| 工作负载 | 优化前 s/update | 优化后 s/update | 提速 | 兼容级别 | 优化后峰值 |
|---|---:|---:|---:|---|---:|
| legacy_off | 0.572 | 0.522 | 1.076× | BITWISE | 2.97 GiB |
| legacy_on | 2.222 | 2.194 | 1.010× | FAIL | 9.79 GiB |
| v2_full_off | 0.615 | 0.581 | 1.039× | BITWISE | 2.96 GiB |

1.3× 是目标而不是结果门槛；未达到的工作负载不作提速宣称。启用项仅包括同 context 候选/锚点编码复用、DDIM 循环内诊断同步移除、OFF 无消费者零梯度树惰性化，以及显式 non-reentrant checkpoint 依赖。未通过或未证明有收益的 compile、worker、SDPA、QP 重写均未启用。

## 解释边界

- 一一配对只保持输入锚点集合；默认软残差输出不保证最终 minFDE 不退化。
- `rho=0` 与训练折最近锚点误差 P95 的投影是只读输出诊断，不是默认模型。
- Jacobian bias 是解码敏感度启发式，不是唯一正确的变量变换。
- C 是旧72-token、旧goal表示的已认证 rollout-OFF 控制；它不被描述成 V2 执行速度基准。
- 当前证据只覆盖 ETH，不与旧 HOTEL/UNIV 结构结果拼接。

完整小型收据目录：`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_v2_integrated/ebjd_clean/docs/validation/ebjd_v2_integrated_20261020`。
