# ZARA1 CTF decoder-energy 汇聚短试验

## 结论

本轮已完整结束，候选 **不通过筛选，不建议采用**。原 250 轮训练已按用户授权停止，且不会自动恢复。

`decoder_energy` 在五个推理种子上都使 minFDE 略低于同期 Control，但平均改善只有 `0.000205 m`，远低于预注册的 `0.005 m`。同时 JFDE 平均变差 `0.000450 m`，只在 1/5 个种子改善。更重要的是，Control 与 Candidate 相对父模型 epoch 96 的四项指标都退化，因此候选不是对父模型的真实提升。

## 停止位置与冻结资产

- 原训练源码：`6f86d31f7987f2e8c2fff802e12d8eb04e8d82da`。
- 停止时最后完整边界：epoch 119 / update 32130。
- epoch 120 已记录到 update 32246，其中 116 个更新未进入完整 checkpoint；未把它们伪装成可恢复状态。
- `best.pt` 不存在，故按冻结规则选用 epoch 96 `least_violation.pt` 为父模型。
- 父模型 SHA256：`2b6765349e2f87dd86acdbf4431436aa931553378c64f982fbe57e1b6bc3e3cd`。
- epoch 79 只有曲线指标，没有对应权重，状态为 `METRICS_ONLY_NOT_REEVALUABLE`。
- 冻结资产位于 `RSJG_EBJD_assets/experiments/zara1_ctf_decoder_energy_20261024/`。

## 唯一改动

候选只改变 CTF 的物理时间反馈到 EBD 潜变量槽的汇聚。固定矩阵 `W` 从已有 `bridge_cholesky` 的列能量及终点插值系数构造；每行归一化，桥槽不直接接收最终时刻，终点槽汇聚整个预测区间。该映射是解码敏感度启发式，不是严格坐标变换。

Control 直接执行原 legacy 路径。Candidate 在 epoch 97–98 的 540 个 optimizer updates 内将 eta 从 0 线性增加到 1，epoch 99–101 固定为 1。两臂保持 192 维、4 heads、FFN 768、6 个 FJAB、FULL/Raw、CTF、原 250 轮学习率时间轴；rollout 和 paired actual-step projection 始终为 OFF。

旧 checkpoint 使用 `strict=True` 加载，未新增可训练参数或 state-dict 键。迁移收据确认模型、优化器、RNG 和 update 均精确恢复。Candidate 首个 eta=0 update 与 Control 的 loss 及各分量逐值相同。CPU 定向与全量测试共 45 项通过。

## 五种子正式结果

以下为同一训练 checkpoint 的五个推理种子（2035–2039），不是五个训练种子。数值为均值 ± 样本标准差，单位为米，越低越好。

| checkpoint | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| Parent epoch 96 | 0.202813 ± 0.001194 | 0.361215 ± 0.001175 | 0.287126 ± 0.002231 | 0.573244 ± 0.003408 |
| Control epoch 101 | 0.206984 ± 0.001056 | 0.367236 ± 0.002289 | 0.293640 ± 0.002243 | 0.584369 ± 0.003448 |
| Candidate epoch 101 | 0.206755 ± 0.000976 | 0.367031 ± 0.002249 | 0.293470 ± 0.002264 | 0.584819 ± 0.003536 |

Candidate − Control：

| 指标 | 均值差 | 改善 seed 数 |
|---|---:|---:|
| minADE | -0.000229 | 5/5 |
| minFDE | -0.000205 | 5/5 |
| JADE | -0.000170 | 4/5 |
| JFDE | +0.000450 | 1/5 |

逐 seed 的 Candidate − Control minFDE 为：2035 `-0.000232`、2036 `-0.000106`、2037 `-0.000096`、2038 `-0.000061`、2039 `-0.000530`。方向一致但量级极小。

Candidate − Parent 为：minADE `+0.003942`、minFDE `+0.005816`、JADE `+0.006344`、JFDE `+0.011575`，四项均退化。

## 筛选与 GDTS 边界

预注册条件的判定：

- minFDE 相对 Control 至少改善 `0.005 m`：**未通过**；实际 `0.000205 m`。
- minADE、JADE、JFDE 均不差于 Control：**未通过**；JFDE 变差。
- minFDE 至少 4/5 seed 改善：**通过**；实际 5/5。
- 相对 Parent 的 minFDE 改善且其余三项不退化：**未通过**；四项均退化。

因此总判定为 **DO_NOT_ADOPT**。

matched GDTS 目前只有与本协议绑定的 seed 2035 结果，不能冒充五种子均值。对同一 seed，Candidate − GDTS 为：minADE `+0.019255`、minFDE `+0.085607`、JADE `-0.024357`、JFDE `-0.001043`。联合指标较好，但边缘指标明显更差，原 GDTS 边缘不退化合同仍未通过。

## 终点机制判断

父模型 seed 2035 的真实 20-step DDIM 链中，最后一次 denoiser 调用的 coarse goal 到最终 Raw goal：minFDE 从 `0.362364` 变为 `0.362285`，JFDE 从 `0.575004` 变为 `0.574901`。N=1 和 N≥2 子组也都是极小改善，而非恶化。

因此当前证据更支持 **终点 proposal/先验不足**，不支持“refinement 阶段明显损害终点”。decoder-energy CTF 汇聚没有解决主要边缘差距。

下一步唯一优先方向：在不继续扫描本汇聚矩阵的前提下，单独审查并改善冻结的终点 proposal/先验质量；不要把本候选扩展成长训练。

## 成本与限制

- GPU 硬预算：14400 秒；实际：11168.94 秒（3小时06分09秒），剩余 3231.06 秒。
- 训练时观察到的进程显存约 6642 MiB（`nvidia-smi`），不是 allocator 仪器化峰值，不能当作精确峰值。
- matched GDTS 的 N=1/N≥2 导出入口不存在，本轮按范围记录为 `NOT_VERIFIED_NO_EXISTING_EXPORT`，未另建第二套工程。
- 所有大权重与完整运行结果保留在持久资产目录，Git 仅归档代码、配置、测试和小型结果。
