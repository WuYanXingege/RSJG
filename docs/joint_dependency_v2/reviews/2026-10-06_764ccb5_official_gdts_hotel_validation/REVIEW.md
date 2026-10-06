# GDTS 官方代码 HOTEL baseline 完成与独立验证

结论：固定上游 `Winderting/GDTS@297d508558c10831983ea4b19c2b3e657459a449` 的 HOTEL 训练已完整完成 250 epochs。epoch 110 的 best checkpoint 经独立 strict-load 和 5 次 K=20、TTST 推理验证，得到 **ADE20/FDE20 = 0.13394/0.19140 m**。状态为 **OFFICIAL_CODE_BASELINE_VALIDATED_PAPER_ADE_MATCH_FDE_NOT_MATCH**。

这认证了本地“官方公开代码路径”的 HOTEL baseline，并可作为后续共享同一协议的 A0/A1 对照基座；它不等于论文表格数值的精确复现，也不是独立 held-out test 证据。

## 1. 固定协议与产物

| 项目 | 实际值 |
|---|---|
| 审查基点 | `764ccb50fec3c90cb8a9f0b0ee1657f47d38986d` |
| 上游源码 | `297d508558c10831983ea4b19c2b3e657459a449` |
| 测试折 | HOTEL |
| 模型选择 | 每10轮完整 valid；官方 `best_valid_metric()` 返回像素 ADE |
| 训练 | seed 2025、FP32、batch 64、Adam lr `1e-3`、250 epochs |
| 推理 | 观察8帧、预测12帧、TTST、K=20、同一进程5次随机推理 |
| loader | train 591 batches、valid 19 batches |
| best checkpoint | epoch 110，112个 state tensors，strict-load 通过 |
| checkpoint SHA256 | `5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b` |

训练完整运行 5:30:20，最后保存到 epoch 250；best checkpoint 仍为 epoch 110。训练配置、曲线和日志的 SHA256 见 [VALIDATION_RECEIPT.json](VALIDATION_RECEIPT.json)。

## 2. 独立验证结果

验证使用训练结束后保存的 best checkpoint，重新启动官方 `main.py` test-only 路径；没有 optimizer update，也没有写入新 checkpoint。

| run | ADE20 (m) | FDE20 (m) |
|---:|---:|---:|
| 0 | 0.13496 | 0.18940 |
| 1 | 0.13489 | 0.19518 |
| 2 | 0.13447 | 0.19296 |
| 3 | 0.13292 | 0.19060 |
| 4 | 0.13248 | 0.18887 |
| **官方程序输出均值** | **0.13394** | **0.19140** |
| 由打印值重算的总体标准差 | 0.001039 | 0.002356 |

独立验证耗时 2:33，checkpoint epoch 110 strict-load 通过，19/19 batches 全部完成。完整本地日志位于 `outputs/joint_dependency_v2/standard_gdts_hotel_297d508_20261006/formal_validation/official_gdts_hotel_validation.log`，SHA256 为 `49d7d1bdaad6cef2f3560b044f20479e2b380ee5872f6a9025801bf9773d32e6`；运行日志不提交 Git。

## 3. 重复稳定性

训练进程结束时自动执行的另外5次推理为：

| run | ADE20 (m) | FDE20 (m) |
|---:|---:|---:|
| 0 | 0.13462 | 0.19230 |
| 1 | 0.13374 | 0.19018 |
| 2 | 0.13259 | 0.18955 |
| 3 | 0.13443 | 0.19107 |
| 4 | 0.13386 | 0.19214 |
| **官方程序输出均值** | **0.13385** | **0.19105** |

独立验证与训练末自动评测的均值差为 ADE `+0.00009 m`、FDE `+0.00035 m`。这支持当前 checkpoint 在官方随机推理口径下的重复稳定性；这10次推理不是10个独立训练模型，也不能替代多训练种子统计。

## 4. 与论文 Table I 的关系

用户提供的 IROS 论文 PDF Table I 报告 HOTEL 为 `0.13/0.18 m`。本地独立验证为 `0.13394/0.19140 m`：

- ADE 四舍五入到两位小数为 `0.13`，与论文表格一致；
- FDE 四舍五入到两位小数为 `0.19`，不等于论文的 `0.18`；
- 相对论文已舍入值，原始差值为 ADE `+0.00394 m`、FDE `+0.01140 m`。由于论文只给两位小数，不能据此声称精确相对误差或逐位复现。

因此准确表述是：**官方公开代码 HOTEL baseline 已在本机验证，ADE 达到论文报告的两位小数，FDE 未达到；不可标记为论文指标完整复现。**

## 5. 必须保留的边界

1. 官方 test-only 入口仍执行预处理并从相同原始文件重新物化 cache；本次看到的数量为 train 591、valid 19、test 0。它没有改变 checkpoint、模型、采样或指标，但不是只读 cache 回放。
2. 官方 tester 调用 `mode='valid'`；作者数据包中 HOTEL valid/test 的 `biwi_hotel.txt` 字节完全相同，SHA256 均为 `9caa771bb9153d6b809dd0916b6f86761b641e6bbb15e766c1de3133fbbb7fcf`。因此不能把该结果描述为 leakage-controlled 的独立 held-out test。
3. 公开代码没有实际加载论文文字所述的 goal-pretrain 权重；本实验是官方公开代码的 fresh joint 路径，不冒充论文意图的两阶段训练。
4. 现代依赖兼容补丁仅替换 Albumentations 公共入口，并对高通道热图 replay 同一几何增强；补丁后 `src/data_loader.py` SHA256 为 `c6c66e761455c126dda173d107033dd30e60d4777acafeb285bd671632376952`。模型、loss、采样和 metric 未改。
5. 本次验证没有启动其他数据折、JDV2 cache 或 A0/A1 训练。

## 6. 决策

- **通过**：作为本地官方代码 HOTEL baseline，并用于严格共享 checkpoint、cache、评测器和预算的后续 A0/A1 内部比较。
- **未通过**：声称 GDTS 论文 HOTEL `0.13/0.18` 已被完整复现。
- **未通过**：作为独立 held-out、无泄漏测试的证据。
- 标准五折主结果仍需按相同方式分别完成 ETH、UNIV、ZARA1、ZARA2，并明确区分本地重训结果与论文公开数字。
