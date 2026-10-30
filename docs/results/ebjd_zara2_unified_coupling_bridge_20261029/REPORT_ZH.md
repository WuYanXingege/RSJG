# ZARA2 统一配对条件桥最终训练报告

## 完成状态

实验已于 2026-10-29 完成：30 epochs、7,530 次 optimizer update，并对 selector 冻结的 checkpoint 完成推理种子 2035–2039 的正式评测。`RUN_MANIFEST.json` 标记 `COMPLETE`，性能合同 `performance_contract_pass=true`。

本实验只有一个训练 seed（3101）；2035–2039 是同一 checkpoint 的五次推理随机重复，不是五个训练 seed。

## 固定身份

- fold：ZARA2
- 源码提交：`5369f589a8b4ad89735b4028e237bede53d35771`
- 模型：可靠原生终点先验＋可学习联合配对＋未来交互条件桥
- 输出：G（原生候选）与 H（条件桥最终轨迹）
- worlds / DDIM steps：20 / 20
- rollout / paired actual-step projection：均关闭
- validation 场景/行人出现次数：998 / 5,910
- selected checkpoint：`best_eligible.pt`，epoch 23
- selected checkpoint SHA256：`1dcdf54a2ea8081ee21480f1bc7a0499b89dff5a8ccc2c99357f9dd01243c3be`
- matched GDTS checkpoint SHA256：`bba3e538775e57fac45fd693ae6308c84fd92d6bf8d5451ab337005b292b2d80`

## 五推理种子正式结果

均值后的 `±` 为样本标准差（ddof=1），单位为米。

| 方法 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| matched GDTS | 0.158421 ± 0.000305 | 0.257847 ± 0.001030 | 0.295602 ± 0.001261 | 0.582360 ± 0.002315 |
| 原生候选 G | 0.157844 ± 0.000532 | 0.249244 ± 0.000676 | 0.283911 ± 0.000874 | 0.570115 ± 0.002664 |
| 条件桥 H | **0.157652 ± 0.000473** | **0.249244 ± 0.000676** | **0.270794 ± 0.000967** | **0.541744 ± 0.001152** |

H 相对 matched GDTS 的均值变化：

- minADE：降低 `0.000769 m`（约 0.49%）；
- minFDE：降低 `0.008603 m`（约 3.34%）；
- JADE：降低 `0.024808 m`（约 8.39%）；
- JFDE：降低 `0.040616 m`（约 6.97%）。

H 相对 G：minADE 降低 `0.000192 m`，minFDE 保持不变，JADE 降低约 4.62%，JFDE 降低约 4.98%。这说明条件桥在保留冻结终点覆盖的同时，主要改善了联合轨迹协调。

## 逐 seed 的 H 指标

| 推理 seed | minADE | minFDE | JADE | JFDE |
|---:|---:|---:|---:|---:|
| 2035 | 0.158050 | 0.250403 | 0.271457 | 0.542692 |
| 2036 | 0.158237 | 0.249272 | 0.271472 | 0.539783 |
| 2037 | 0.157543 | 0.248944 | 0.271558 | 0.542085 |
| 2038 | 0.157241 | 0.248767 | 0.269888 | 0.541744 |
| 2039 | 0.157188 | 0.248836 | 0.269594 | 0.542417 |

## 终点提供器依赖结论

本实验使用的原生终点 G 不能由 Goal-pretrain 权重原样替代：ZARA1/ZARA2 的 Goal U-Net 在 matched GDTS 阶段均有 46/46 参数张量发生变化，Goal-pretrain-only 回放与正式 G 不一致。

但完整 GDTS 轨迹网络不是运行时依赖。仅抽取 matched checkpoint 中的 Goal U-Net（613,772 参数），配合原预处理、TTST 和坐标转换，即可在 ZARA1/ZARA2 的固定 CPU 回放中逐值复现正式 G。因此当前依赖应描述为：**matched GDTS 是权重来源/训练历史依赖，而非部署时的完整轨迹去噪网络依赖。**

详细逐参数与回放证据见同一归档分支的 `docs/audits/goal_provider_dependency_20261029/`。

## 成本与证据边界

- 统一训练与评测 GPU 墙钟：43,782.74 秒（约 12.16 小时）；
- 完整串行流水线 GPU 预算消耗：47,752.64 秒（约 13.26 小时）；
- 训练 seed：1；推理 seed：5；
- validation/test 为当前官方数据包的镜像成员设置，只支持该协议内的本地配对比较；
- 结果不能表述为五个独立训练 seed 的统计验证。
