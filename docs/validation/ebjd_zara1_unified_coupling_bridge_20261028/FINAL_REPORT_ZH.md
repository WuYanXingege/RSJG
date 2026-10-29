# ZARA1 统一配对条件桥最终报告

## 结论

状态：`COMPLETE`。固定 30 epochs 全部完成，共 8,100 次 optimizer update；五个推理种子 2035–2039 的 G/H 冻结评测完成。按预注册规则，最终判断为 **`READY_FOR_ZARA2`**。

这表示 ZARA1 上“冻结可靠原生终点先验＋可学习联合配对＋未来交互条件桥”的单阶段方案同时满足：

1. 直接条件桥 G 的四项五种子均值均不差于 matched GDTS；
2. 学习配对 H 严格保持终点候选集合和 minFDE，并对 JADE/JFDE 产生五个种子一致的联合收益；
3. H 的四项五种子均值均优于 matched GDTS。

这仍是一个训练 seed（3101）与五个随机推理 seed，不能表述为五个独立训练重复或统计显著性检验。

## 固定身份

- 训练源码提交：`9bf9e71b0da74e69f3bca9add6833c39b51f1642`
- 训练 fold：ZARA1
- 模型：192 hidden、4 heads、FFN 768、6 FJAB、FULL、CTF、Raw
- 训练：30 epochs，rollout OFF，paired actual-step projection OFF
- 推理：20 worlds、20 DDIM steps
- 选中 checkpoint：epoch 30 `best_eligible.pt`
- checkpoint SHA256：`062a497c7baa28bfe88cf7e559784ad7029e979ddf93f3e0f8b08baa4713214b`
- matched GDTS checkpoint SHA256：`34af0f4335de77cf2b6ecd48b93262dfc4764e679bd52fd677392a85c22bf6d7`
- validation manifest SHA256：`9c12e30ac17e91bbf8efa9e23a3d61a16c9e9c8dfc6414871fc7431b76379478`

## 五推理种子正式结果

均值 ± 样本标准差，单位为米，越低越好。

| 路径 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| G：原生终点＋直接条件桥 | 0.184812 ± 0.000424 | 0.274816 ± 0.000666 | 0.296446 ± 0.001098 | 0.572203 ± 0.003532 |
| H：学习配对终点＋同一条件桥 | **0.184314 ± 0.000317** | **0.274816 ± 0.000666** | **0.279281 ± 0.001438** | **0.528041 ± 0.002193** |
| matched GDTS | 0.188556 ± 0.001004 | 0.285608 ± 0.001405 | 0.318124 ± 0.002240 | 0.585493 ± 0.004152 |

H 相对 G 的均值差（H−G）：

- minADE：−0.000498 m；
- minFDE：0.000000 m；
- JADE：−0.017166 m（改善 5.79%）；
- JFDE：−0.044161 m（改善 7.72%）。

H 相对 matched GDTS 的均值差（H−GDTS）：

- minADE：−0.004242 m；
- minFDE：−0.010792 m；
- JADE：−0.038844 m；
- JFDE：−0.057452 m。

## 配对方向一致性

H 相对 G：minADE、JADE、JFDE 均为 5/5 个推理 seed 改善；minFDE 为 5/5 完全相同。这与一对一排列严格保持每个行人的终点候选集合一致。

逐 seed 的 JADE/JFDE 差值（H−G）：

| seed | ΔJADE | ΔJFDE |
|---:|---:|---:|
| 2035 | −0.018329 | −0.048585 |
| 2036 | −0.016624 | −0.042550 |
| 2037 | −0.017038 | −0.047771 |
| 2038 | −0.017308 | −0.040395 |
| 2039 | −0.016528 | −0.041504 |

## 三项固定判断

### A. 直接条件桥是否训练成功

是。G 的四项五种子均值均优于 matched GDTS。说明不依赖旧 Base→Stage A→Candidate 链条，从随机初始化训练共享上下文与条件桥（仅复用同折地图 U-Net 初始化）能够在 30 epochs 内形成有效路径模型。

### B. 学习配对是否有效

是。H 不改变终点候选集合及 minFDE，却在全部五个推理 seed 上同时降低 JADE 和 JFDE；路径 minADE 也小幅改善。收益表现为联合 world 组织改善，而不是生成了新的终点或牺牲边缘终点覆盖。

### C. 是否适合替换旧流程

在本次 ZARA1 证据范围内满足替换闸门，标记 `READY_FOR_ZARA2`。相较旧 GT-conditioned E3 五种子结果，H 的 JADE/JFDE 还分别降低约 0.00870/0.00731 m；minFDE 相同，minADE 高约 0.00056 m。该比较支持简化流程，但跨折结论仍需新的 ZARA2 正式复验，不能由 ZARA1 自动外推。

## 计算成本与运行状态

- GPU 墙钟：32,608.15 s（约 9.06 h），包含训练、每轮完整 G/H 验证和最终五种子评测；低于 12 h 上限。
- 每轮总耗时约 16.8–17.3 min，包含训练和完整验证。
- 后台服务以退出码 0 正常结束，当前没有该训练 GPU 进程。
- ZARA2 旧流水线仍保持 `PAUSED_FOR_UNIFIED_MODEL_VALIDATION`，本轮没有自动恢复或启动 ZARA2。

## 证据边界

`READY_FOR_ZARA2` 仅授权按同一冻结合同开展 ZARA2 跨折复验，不代表已经获得跨数据集统一验证，也不授权温度、损失权重或网络宽度扫描。大 checkpoint、proposal bank 和原始逐 update 日志保留在持久资产目录，不进入 Git。
