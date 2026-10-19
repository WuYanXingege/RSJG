# EBJD ETH 跨折表现差异诊断（2026-10-19）

## 结论

ETH 折的失败不能用“ETH 平均人数少”单独解释。现有证据支持以下组合机制：

1. JMM 论文中的 `ETH (1.4)` 是 **left-out ETH 评测窗口**的平均人数，不是训练集平均人数。
2. ETH 留出集相对四场景训练集存在显著的 endpoint/bridge 表示尺度外推：goal residual RMS 为训练尺度的 `1.95x`，bridge residual RMS 为 `2.41x`。HOTEL 的对应比例只有 `0.55x/0.76x`，UNIV 为 `1.16x/1.01x`。
3. ETH 的 253 个评测窗口中有 183 个单人窗口，占 `72.33%`。这些窗口没有多人 joint interaction 的收益空间。
4. EBJD 使用 Gaussian-init DDIM，未保留 GDTS 的 heatmap、TTST endpoint proposal 和 tree sampling。固定预测库直接显示，ETH 上 EBJD 的 20 条候选逐渐欠分散；单候选平均误差可以改善，但 best-of-20 覆盖变差。

因此，当前最准确的判断是：**UNIV/HOTEL 中可见的 joint 收益，在 ETH 上不足以补偿跨场景尺度失配与 marginal candidate coverage 损失。** 这说明 joint 机制具有条件性价值，但当前 sampler/representation calibration 缺乏 ETH 所需的跨场景稳健性。

## JMM 协议中的 `ETH (1.4)`

JMM 在 Evaluation Setup 中规定 ETH/UCY 使用 leave-one-scene-out：在五个环境场景中的四个上训练，在剩余一个场景上评测。Table 1 caption 将括号数字定义为每个 20-frame sequence 的平均 agent 数。

因此 ETH 折为：

- 训练：HOTEL、UNIV、ZARA1、ZARA2；
- 评测：left-out ETH；
- `ETH (1.4)`：left-out ETH 评测窗口的平均人数。

当前本地 ETH manifest 与此数值一致：`364 / 253 = 1.438735`。论文未在主文中说明独立 inner validation 的精确构造。当前本地 validation/test 使用相同的 253 个 ETH `window_keys`，属于 mirrored evaluation population，不是独立测试。

## 三折人口与表示尺度

下表中的 residual RMS 使用 EBJD 实际 `EndpointBridgeRepresentation` 公式重新计算。训练尺度来自对应 leave-one-scene-out 四场景训练集，评测尺度来自 left-out 场景。

| 折 | 训练 scenes / agents | 训练 mean N | 评测 scenes / agents | 评测 mean N | 单人窗口比例 | goal RMS train → eval | bridge RMS train → eval |
|---|---:|---:|---:|---:|---:|---:|---:|
| HOTEL | 4249 / 37702 | 8.873 | 445 / 1197 | 2.690 | 32.36% | 1.0635 → 0.5854 (`0.55x`) | 0.2174 → 0.1656 (`0.76x`) |
| UNIV | 4067 / 15186 | 3.734 | 947 / 24334 | 25.696 | 0% | 0.9640 → 1.1135 (`1.16x`) | 0.2152 → 0.2178 (`1.01x`) |
| ETH | 4110 / 36906 | 8.980 | 253 / 364 | 1.439 | 72.33% | 1.0597 → 2.0715 (`1.95x`) | 0.2078 → 0.5000 (`2.41x`) |

三折的模型、diffusion、loss 和 representation 配置相同。UNIV 与 ETH 还使用相同的 `small_group_uncheckpoint_v1` 执行策略；HOTEL 使用已迁移的 `compact_grouped_v1`。因此没有发现一个只在 ETH 开启的损失或网络开关可直接解释差距。

## 相同 inference seed 2035 的 matched-GDTS 对比

数值均为 P20、20 DDIM steps、相同 fold manifest；负相对变化表示 EBJD 更好。

| 折 / checkpoint | minADE | minFDE | JADE | JFDE | 相对 GDTS |
|---|---:|---:|---:|---:|---|
| HOTEL GDTS | 0.135220 | 0.193382 | 0.211186 | 0.359854 | reference |
| HOTEL EBJD epoch 45 | 0.138630 | 0.204783 | 0.168829 | 0.278901 | `+2.52% / +5.90% / -20.06% / -22.50%` |
| UNIV GDTS | 0.287591 | 0.521361 | 0.654297 | 1.318582 | reference |
| UNIV EBJD epoch 73 | 0.274961 | 0.516973 | 0.532392 | 1.134318 | `-4.39% / -0.84% / -18.63% / -13.97%` |
| ETH GDTS | 0.417053 | 0.580032 | 0.489907 | 0.743412 | reference |
| ETH EBJD epoch 5 | 0.582031 | 1.015402 | 0.647612 | 1.121817 | `+39.56% / +75.06% / +32.19% / +50.90%` |

HOTEL 不能描述成“四项都更好”：它是 joint metrics 明显改善、marginal metrics 轻微退化。UNIV 才是四项均通过 matched baseline。ETH 则在所有四项上均显著落后。

此外：

- HOTEL epoch 45 是 least-violation checkpoint，不是零 marginal violation 的 best-eligible checkpoint；
- UNIV epoch 73 是 best-eligible checkpoint；
- ETH 100 epochs 内没有 best-eligible checkpoint，epoch 5 只是 least-violation checkpoint。

## ETH 固定预测库的直接证据

固定 ETH 253 scenes、364 agent occurrences、seed 2035、P20、20 steps，并保持 checkpoint 只读。所有 prediction-bank replay parity 为精确一致，optimizer updates 为零。

### 长期变化

| 模型 / checkpoint | 平均单候选 FDE | minFDE@20 | endpoint center offset | endpoint RMS radius | pairwise endpoint distance |
|---|---:|---:|---:|---:|---:|
| GDTS | 2.2425 | 0.5800 | 1.6227 | 1.6996 | 2.1924 |
| EBJD epoch 5 | 2.5134 | 1.0154 | 2.1268 | 1.4438 | 1.8056 |
| EBJD epoch 22 | 2.4661 | 1.0673 | 2.1476 | 1.3244 | 1.6211 |
| EBJD epoch 23 | 2.2614 | 1.2620 | 2.1176 | 0.8268 | 1.0098 |
| EBJD epoch 65 | 2.2399 | 1.1803 | 2.0566 | 0.9326 | 1.1247 |
| EBJD epoch 66 | 2.2036 | 1.3709 | 2.0757 | 0.7142 | 0.8603 |
| EBJD epoch 100 | 2.1881 | 1.2713 | 2.0339 | 0.7980 | 0.9605 |

epoch 5→100 时，平均单候选 FDE 改善 `12.9%`，但 minFDE@20 恶化 `25.2%`；endpoint RMS radius 缩小 `44.7%`，pairwise endpoint distance 缩小 `46.8%`。这说明网络不是简单地“每条轨迹都更差”，而是把 20 条预测压缩到了更窄的区域。

### 相邻轮次突降

- epoch 22→23：平均单候选 FDE 改善 `8.3%`，minFDE@20 恶化 `18.2%`；RMS radius 与 pairwise distance 分别收缩 `37.6%/37.7%`。
- epoch 65→66：平均单候选 FDE 改善 `1.6%`，minFDE@20 恶化 `16.2%`；RMS radius 与 pairwise distance 分别收缩 `23.4%/23.5%`。
- 两次变化中分别有约 `65.1%` 和 `62.1%` 的 agent minFDE 变差，不是少数离群点造成。
- N=1 与 N>=2 两组都退化，因此不能只归因于 social encoder 或 interaction layer。

这构成“candidate under-dispersion / coverage collapse”的直接数值证据。它尚不能把收缩归因到某一具体子模块。

## 训练目标为何没有阻止收缩

当前训练使用 4-world normalized soft-min rollout，每 4 次 optimizer update 执行一次，soft-min temperature 从 0.10 m 降到 0.05 m；正式评测使用 20-world hard min。marginal projection 保护的是当前训练 minibatch 上的 4-world soft loss，不是 held-out ETH 上的 P20 hard metric，也没有显式保护候选半径或 pairwise coverage。

训练日志出现了明确的 objective mismatch：

- epoch 22→23 diffusion loss 从 `1.3959` 降至 `1.3525`，但验证 minFDE 明显恶化；
- epoch 65→66 diffusion loss从 `1.2543` 降至 `1.2415`，训练 softADE/softFDE 从 `0.3707/0.7359` 改善到 `0.3611/0.7180`，但固定库 minFDE 从 `1.1803` 恶化到 `1.3709`。

因此优化器在改善当前训练目标，但当前目标不足以保证 20-world held-out coverage。

## 证据等级

### CONFIRMED

- JMM `ETH (1.4)` 指 left-out ETH evaluation sequences，而不是训练集。
- ETH 的 goal/bridge residual scale 相对训练分布为 `1.95x/2.41x`；HOTEL 与 UNIV 没有同等级尺度外推。
- ETH 中 72.33% 的评测窗口为 N=1。
- ETH 的固定 20-world prediction banks 出现强烈候选收缩，且该收缩直接伴随 minFDE@20 退化。
- 训练没有 OOM、NaN、resume discontinuity 或 manifest population change；相邻轮次突降不是这些工程故障造成。
- validation/test population 在当前本地协议中镜像相同，因此不是独立测试。

### STRONGLY SUPPORTED

- ETH 的表示尺度外推与 EBJD 的 Gaussian/DDIM candidate under-dispersion 共同造成主要 marginal gap。
- UNIV 的高密度、尺度匹配数据使 joint interaction 收益足以超过 sampler 损失；HOTEL 处于中间状态。
- GDTS 的 TTST/tree endpoint proposal 对 ETH 的宽 endpoint coverage 尤其重要。

### OPEN

- goal、bridge、denoiser、decoder 中哪个子模块首先降低 latent-to-output sensitivity。
- seed 3101 是否放大 ETH 失败；当前五个 inference seeds 不是五个独立 training seeds。
- JMM 原论文实现中的 inner validation/model-selection 文件级规则；主文没有明确说明。

## 建议的最小后续诊断

1. 对 epoch 5、22、23、65、66 做只读 latent/noise scale sweep，同时记录 K=1/4/8/12/16/20 曲线与 endpoint radius；先确认 inference calibration 能否恢复覆盖。
2. 从 epoch 22 做同 RNG 的单轮 paired resume：原配置与 rollout-off，各自生成固定 prediction bank，以定位 rollout 是否触发该次收缩。
3. 将每轮 validation 改为 scene-keyed RNG 或 batch-size-independent noise mapping，并与正式 batch=1 评测保持一致。
4. 若只读 scale sweep 有效，再讨论 fold-calibrated prior、coverage-preserving loss 或保留 GDTS endpoint proposal；在此之前不建议直接扩大网络。
5. 至少增加一个独立 ETH training seed。五个 inference seeds 只能测采样随机性，不能证明训练稳定性。

## 来源与完整性

- JMM PDF SHA256: `f7db01c348729f7d4e53d0ebb13cb215e764de07d7e73c3a1f283d39e8c39e90`
- HOTEL epoch45 evaluation SHA256: `5e0f13ab4cb272aebe9ea19e6f0307a9eb3213301256c2f6c826cf449bab800f`
- UNIV final comparison SHA256: `978b6a33d45921d11f53e1f4eeb77571e13819c8983e09550c1f1727308c36a9`
- ETH final comparison SHA256: `b4891cb9601431bd2e3b80b90d47afd68b224521b595a9eccdb9791aa6169020`
- ETH CPU diagnostic SHA256: `e323018a7ef457e3f5fd94b4c34384bb902abf9e2433e5112f425aef1636599b`
- ETH bank analysis SHA256: `3e06a676c686d20b690911ff84f63e8f01916f2241adbdf796b49524e84b8d04`
- 本报告生成前分支 HEAD: `ebbbe189a258ed161be97e26e29fecb53acf467f`

机器可读摘要见 [`validation/EBJD_ETH_CROSS_FOLD_DIAGNOSIS_2026-10-19.json`](validation/EBJD_ETH_CROSS_FOLD_DIAGNOSIS_2026-10-19.json)。
