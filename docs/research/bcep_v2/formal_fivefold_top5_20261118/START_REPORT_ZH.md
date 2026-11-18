# BCEP–EBJD v2 五折正式训练启动报告

北京时间 2026-11-18 14:19:01，**ZARA2 正在执行 Stage 1A 的 density learning，第 4 轮，累计 2170 次真实 optimizer 更新**。每轮 671 次。head 和 joint 各为 0 次，五折测试等待统一冻结。旧 v1 队列保持停止，旧 ZARA1 可恢复断点为 epoch 17、update 2926。

| 折 | 训练窗口 | dev 窗口 | 测试窗口 | 本次快照 |
|---|---:|---:|---:|---|
| ZARA2 | 2681 | 590 | 998 | Stage 1A · density 运行中 |
| ZARA1 | 2889 | 671 | 705 | 等待前折 |
| HOTEL | 3118 | 688 | 445 | 等待前折 |
| UNIV | 2719 | 622 | 947 | 等待前折 |
| ETH | 3283 | 733 | 253 | 等待前折 |

## 启动依据

| 项目 | 实测记录 |
|---|---|
| 源码提交 | 468c5eddde48bdeec47c4bb3c39418352be16bb9 |
| 正式授权 YAML 的 SHA256 | 7812151e68144b28461c8e99e8b5923caa4343a91e4f7f0a9af7fc9c4acaa362 |
| 解析后的配置身份 | edddd578f60a376395a3778ac01acbad4e891851e511ac7128db8353a71bf787 |
| 源码内容摘要 | 6113fba5c13b9cf492c9f82cefeb243c37b76ebc292a68374f92968518aabf88 |
| 首个已提交更新 | 2026-11-18 14:10:59，4 个窗口，63 次有效行人出现 |
| 首步 loss / 裁剪前梯度范数 | 3.70623422 / 1.21289170 |
| 首步完整断点 SHA256 | f615ea5adf35e152383a07867627374fb777e5467230227ad0d1edaa679e69fa |
| 完整状态读回 | 72 个 AdamW 状态均为 step 1，含 scheduler、CPU/CUDA、Python、NumPy、顺序及视图 RNG，epoch 内 cursor=4 |
| 后台控制器 / GPU 子进程 | 1922379 / 1922386，控制器独立会话，单一进程锁 |
| 硬件与计算 | RTX 5070 Ti，PyTorch 2.8.0+cu128，CUDA 12.8，FP32，严格确定性，math SDPA，TF32 关闭 |
| 磁盘预检 | 可用 87.42 GiB，按全部完整 epoch 和测试矩阵估算需 60.83 GiB |

首步权重和完整优化器状态保存在私有实验目录。公开收据列出了各折 split、成员、快照时间和配置哈希。[机器可读启动收据](START_RECEIPT.json)

## 已完成轮次的 density 指标

| epoch | 累计更新 | 每更新平均训练 loss | dev 终点 NLL | 训练秒数 | 验证秒数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 671 | 2.755148 | 2.355964 | 140.4 | 11.1 |
| 2 | 1342 | 2.437686 | 2.453349 | 137.5 | 11.1 |
| 3 | 2013 | 2.336480 | 2.378348 | 139.0 | 11.3 |

当前 density 的 dev 最优是第 1 轮。表中时间分别覆盖训练和验证。五 seed 的 dev minFDE 要等候选头开始验证后才有数据，H-JADE 要等 Stage 2。

## 留存、恢复与后续测试

Stage 1 依次学习密度、生成基础候选、冻结密度并学习校准头、核验精修 train/dev bank。Stage 2 使用冻结的 Provider，联合训练配对、反馈和六层条件桥。三段优化分别记录 epoch 和更新数。五个 CPU 检查通过，覆盖退出 75、控制器收到停止请求、异常退出、dev 排名同分规则和非首位权重只读加载。此次控制器检查的合成 optimizer 更新数为 0。

Provider Top5 固定本折 density dev rank1，按 head 五个推理 seed 的平均 dev minFDE 选取五个不同完整 epoch。Joint Top5 固定 Provider dev rank1，按 dev H-JADE 选取；同分取较早 epoch。保留内部 density Top5、全部完整 epoch、best 和可恢复 last。退出 75 时整个控制器暂停。恢复用原运行根目录的同一控制器入口，续接实际子流程。[运行与恢复说明](RUN_SPEC_ZH.md)

五折全部冻结后，先完成所有 dev rank1 的正式 G/H，再执行预登记的诊断。总计 125 组 H、25 组正式 G 及 125 组 Provider 终点评估。测试后择优的 checkpoint 单列为探索结果。当前尚无本轮测试成绩。
