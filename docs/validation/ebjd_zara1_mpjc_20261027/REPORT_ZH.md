# ZARA1 MPJC 冻结端点与条件桥短试验

## 结论

状态：**COMPLETE / ENDPOINT_COUPLING_ONLY**。

MPJC 完整保留了原生 proposal 的逐人候选集合，并将 Candidate 的联合 world 参考稳定转化为终点 JFDE 收益：N≥2 五推理种子的平均 JFDE 由 `0.630249` 降至 `0.587097`，改善 `0.043152 m`，5/5 seed 方向一致；minFDE 与 Recall@0.5m 逐 seed 严格不变。因此冻结端点 gate 通过。

但 3 轮条件桥短训没有通过预注册轨迹 gate。seed2035 下 BRIDGE_COUPLED 相对 BRIDGE_NATIVE 的 JADE 仅改善 `0.002629 m`，低于 `0.005 m` 门槛；minADE 改善 `0.004281 m`，没有边缘路径退化。按规则不补 2036–2039 完整轨迹推理，也不替换主模型。

唯一下一步建议：**保留 MPJC 作为冻结终点耦合原型，下一轮只研究如何让条件桥在固定联合 world 下更充分利用终点条件；不要继续扫描匹配 cost，也不要直接启动长期训练。**

## 身份与实现

- 冻结 Candidate reference SHA256：`4b08aa68fef82a0ee31663d4c860311aa5ff260afb4946072f05de75b668dfed`。
- matched GDTS SHA256：`34af0f4335de77cf2b6ecd48b93262dfc4764e679bd52fd677392a85c22bf6d7`。
- validation manifest SHA256：`9c12e30ac17e91bbf8efa9e23a3d61a16c9e9c8dfc6414871fc7431b76379478`。
- K=20；每个有效行人使用 squared-Euclidean cost 与 SciPy `linear_sum_assignment` 做精确一对一分配。R 与匹配不读取 GT。
- H 通过 G 的索引 gather 产生；每个 valid agent 的 pi 都是 0–19 的完整排列。N=1 保持 identity。
- 五个 seed 的 602 个多人场景全部发生真实跨人配对变化；平均 relative-change `0.943932`，不是共同全局 world 重编号。
- 完整轨迹比较中，H 使用的初始 bridge noise 由 G 的同一基础噪声按 pi 同步 gather。

## 冻结端点结果

均值 ± 样本标准差；五个 seed 是同一冻结 checkpoint 的推理重复。

| 端点路径 | minFDE | JFDE | Recall@0.5m |
|---|---:|---:|---:|
| G：原生 proposal | 0.274816 ± 0.000666 | 0.572203 ± 0.003532 | 0.917912 ± 0.001458 |
| R：Candidate reference | 0.309826 ± 0.003587 | **0.510408 ± 0.003630** | 0.862054 ± 0.006479 |
| H：MPJC 耦合 proposal | **0.274816 ± 0.000666** | 0.535355 ± 0.005775 | **0.917912 ± 0.001458** |

N≥2 的 H−G JFDE 差值分别为 `-0.036644`、`-0.037901`、`-0.051237`、`-0.044760`、`-0.045219 m`。H 恢复了 G 的逐人覆盖，同时保留了 R 的一部分联合收益；R 仍有更低 JFDE，说明将连续 reference 投影回冻结候选集合会损失部分布局收益。

## 条件桥固定 E+3 比较

两臂都从同一原 Candidate 模型参数初始化，各自使用空且相同定义的 optimizer，分别执行 810 个 updates；主干/U-Net LR 为 `2e-5/2e-6`，eta=0，rollout/projection 关闭。比较对象固定为第 3 轮 checkpoint。

| seed2035 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| BRIDGE_NATIVE | 0.226309 | 0.274462 | 0.318334 | 0.574082 |
| BRIDGE_COUPLED | **0.222028** | 0.274462 | **0.315705** | **0.542792** |
| COUPLED−NATIVE | -0.004281 | 0.000000 | -0.002629 | -0.031290 |

N≥2 下，COUPLED 相对 NATIVE 的 minADE/JADE/JFDE 差值为 `-0.004814/-0.004338/-0.036644 m`。JFDE 差值属于冻结 H 的 MPJC 收益；桥训练能归因的主要观察是 minADE 与 JADE 的小幅改善。因 JADE 改善未达 0.005 m，轨迹 gate 为 FAIL。

相对旧 Candidate seed2035（`0.192211/0.307711/0.275192/0.507982`），BRIDGE_COUPLED 改善 minFDE，但 minADE、JADE、JFDE 均更差，是另一种精度权衡，不是四项同时改善。相对 matched GDTS seed2035（`0.188109/0.285034/0.319607/0.586878`），它改善 minFDE、JADE、JFDE，但 minADE 更差 `0.033920 m`，因此没有通过原四项性能合同。GDTS 只有一个绑定 seed，不给出五种子合同结论。

## 成本、资产与边界

- BRIDGE_NATIVE checkpoint：`fff17149a59bfb5c281ed01836792f0052e55d9f089376c889cda75f97362e7b`。
- BRIDGE_COUPLED checkpoint：`887b0a91813e98cd978a9103e3fed081345c385c3f72ff53e43d0c5417f2b574`。
- train MPJC bank SHA256：`aebb0adfa4925edaa8029f36cb90b6f29398d6b7f5d93464d5e52ec1a671664b`。
- 一次完整 validation 在线原型（冻结 Candidate 20 步＋匹配＋条件桥 20 步）耗时 `395.67 s`；其中匹配及 reference tensor 传输 `0.223 s`。此数字不包含另计的缓存 GDTS proposal provider。
- 成功调用可逐项计时合计 `8466.88 s`；加首次 padding-NaN 断言失败调用约 `184.3 s`，计入预算共 `8651.18 s`，低于 14400 s。另一次错误 train-manifest 哈希在正式推理前被拒绝，约 1 秒初始化未单独计时。
- 峰值 CUDA allocated memory `3593318400 bytes`（约 3.35 GiB）。
- 证据来自一个训练 seed 3101；端点为五推理 seed，完整轨迹因预注册 gate 失败只评 seed2035。没有统计显著性声明。
- 大型 bank、checkpoint 与逐 update 日志留在持久资产目录，未进入 Git。原 250 轮训练未恢复，本轮结束后不启动其他实验。
