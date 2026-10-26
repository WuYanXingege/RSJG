# ZARA1 原生终点先验条件桥短试验

## 结论

状态：**COMPLETE / PRIOR_TRANSFER_ONLY**。

可靠的 GDTS 原生终点 proposal 已成功转移到 EBJD 完整轨迹：Candidate 相对 Parent epoch96 的五种子均值同时改善 minADE、minFDE、JADE 和 JFDE，先验转移筛选通过。但当前自由终点残差不应保留：它相对固定终点 Control 稳定改善路径及联合指标，却在 5/5 个推理 seed 上损害 minFDE，并使 Recall@0.5m 平均下降 5.59 个百分点。当前变体不升级正式模型，也不启动长期训练。

下一步唯一优先方向是：**保持原生 proposal 的逐人候选覆盖不变，研究场景级候选耦合/联合 world 选择，而不是继续训练当前逐人自由终点残差。**

## 设置与配对

- Parent：ZARA1 epoch96，SHA256 `2b6765349e2f87dd86acdbf4431436aa931553378c64f982fbe57e1b6bc3e3cd`。
- matched GDTS：SHA256 `34af0f4335de77cf2b6ecd48b93262dfc4764e679bd52fd677392a85c22bf6d7`。
- 结构为 192 hidden、4 heads、FFN768、6 FJAB，FULL/Raw；rollout 与 paired projection 全程关闭。
- Stage A：eta=0 三轮；Control：从 Stage A 完整状态恢复、eta=0 三轮；Candidate：从同一完整状态恢复、eta=1、trust weight 0.01 三轮。
- 每轮 270 次 optimizer updates。实际执行 Stage A 810、Control 分叉后 810、Candidate 分叉后 810，共 2430 次。
- 两臂恢复相同模型、optimizer、loader 与 CPU/CUDA RNG 状态。两臂原始样本、增强、world 选择与噪声抽样对应；因编码定义不同，latent 数值按预注册预期不同。
- validation 为同一 705 场景、2356 行人 manifest；20 worlds、20 DDIM steps；一个训练 seed 3101、五个推理 seed 2035–2039。

Control 的 eta=0 终点逐元素等于 proposal，五次验证最大绝对误差均为 0。新增 proposal embedding 之外的 307 个 Parent tensor 逐值继承；旧权重通过显式转换后 strict-load。

## 五种子结果

均值 ± 样本标准差，越低越好；五个 seed 是同一训练 checkpoint 的推理重复。

| 臂 | minADE | minFDE | JADE | JFDE | Recall@0.5m |
|---|---:|---:|---:|---:|---:|
| Control eta=0 | 0.217728 ± 0.001033 | **0.274816 ± 0.000666** | 0.325724 ± 0.003328 | 0.572203 ± 0.003532 | **0.917912 ± 0.001458** |
| Candidate eta=1 | **0.192292 ± 0.000934** | 0.309826 ± 0.003587 | **0.275205 ± 0.000737** | **0.510408 ± 0.003630** | 0.862054 ± 0.006479 |

Candidate−Control 均值：minADE `-0.025436`、minFDE `+0.035010`、JADE `-0.050519`、JFDE `-0.061794`、Recall@0.5m `-0.055857`。Candidate 改善 seed 数为 minADE 5/5、minFDE 0/5、JADE 5/5、JFDE 5/5。

| seed | ΔminADE | ΔminFDE | ΔJADE | ΔJFDE |
|---:|---:|---:|---:|---:|
| 2035 | -0.026255 | +0.033249 | -0.052621 | -0.066100 |
| 2036 | -0.024787 | +0.037848 | -0.054273 | -0.056624 |
| 2037 | -0.026594 | +0.038497 | -0.049814 | -0.062828 |
| 2038 | -0.025123 | +0.035490 | -0.045016 | -0.059396 |
| 2039 | -0.024422 | +0.029965 | -0.050871 | -0.064023 |

## 分组、修正幅度与门槛

N≥2 的 JFDE 均值由 Control `0.630249` 降至 Candidate `0.552159`，改善 `0.078090 m`；seed2035 Candidate 的 N≥2 JFDE `0.548319` 也优于 Parent 的 `0.617446`。因此当前问题不是“多人联合终点仍弱于 Parent”。

问题在于残差并不小：Candidate 的终点修正 P50/P95 平均为 `0.838519/2.301594 m`。弱 trust loss 没有保住 proposal 的有效覆盖，导致 minFDE 与 Recall 稳定退化。

- 先验转移筛选：**PASS**。Candidate 相对 Parent 五种子均值的 minFDE 改善 `0.051389 m`，minADE 与 JADE 也分别改善 `0.010521/0.011921 m`。
- 残差保留筛选：**FAIL**。虽然 N≥2 JFDE 改善，但 minFDE 恶化 `0.035010 m`（门槛最多 0.005），Recall@0.5m 下降 `0.055857`（门槛最多 0.01）。
- matched-GDTS 原合同：seed2035 **FAIL**。Candidate 的 JADE/JFDE 优于 GDTS，但 minADE/minFDE 分别差 `0.004102/0.022677 m`；Control 也未四项同时改善。matched GDTS 只有 seed2035 绑定结果，因此五种子 GDTS 合同为 **NOT_EVALUABLE**。

## 成本与资产

- 可逐项复核的 GPU 计时合计 `12328.05 s`（`3.424 h`），低于 14400 s 硬上限；包括训练 proposal 导出、9 个短训 epoch、四个新增 proposal seed 和八次新增臂评测。两次短 preflight update 与进程初始化未单独计时，但剩余 2071.95 s 预算未被耗尽。
- 峰值 CUDA allocated memory `848858624 bytes`（约 809.5 MiB）；验证期间设备进程占用约 11 GiB，二者口径不同。
- Control checkpoint SHA256：`46622922e8d00f5305a4c2791107cee4af5090cd600bc6d9622920e978ce7964`。
- Candidate checkpoint SHA256：`4b08aa68fef82a0ee31663d4c860311aa5ff260afb4946072f05de75b668dfed`。
- train proposal bank SHA256：`4c5b38a409d3ee8939d24243d191b8b63d361d6431a1f90344bb057e228fb247`。
- 大型 checkpoint、bank 和逐 update 日志保存在持久资产目录，未进入 Git。

原 250 轮训练未恢复；本轮结束后没有训练或评测进程。该结论是一个训练 seed 上的短周期工程筛选，不是统计显著性或跨折验证。
