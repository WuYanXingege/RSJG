# 剩余四折确认实验：后台执行配置

当前为启动归档，尚无新四折的真实测试成绩。后台完成后，本文件将由原生逐窗统计自动生成的中文结果报告替换。

从固定提交 `171bcff3b6b2dbb6aad0becb04e8d7c086c8ffe3` 建立独立分支，按 ZARA1、HOTEL、UNIV、ETH 执行。旧队列及 SSEC 保持现有状态。

| 折 | train 窗数 | selection_dev 窗数 | 初始上游资产 |
|---|---:|---:|---|
| ZARA1 | 2889 | 671 | density 已完成；7 个 base cache 可复用；head 第 10 轮部分断点复制后续跑 |
| HOTEL | 3118 | 688 | 按原合同补训 density、head、coordinate |
| UNIV | 2719 | 622 | 按原合同补训 density、head、coordinate |
| ETH | 3283 | 733 | 按原合同补训 density、head、coordinate |

上游保留原生身份、配方及选择器：density 按开发集终点 NLL，head 按五个推理 seed 的开发集边缘 minFDE 均值，coordinate 按开发集 H-JADE，均同分取较早完整轮。前两阶段 seed=3102，coordinate seed=3201。ZARA1 仅在新目录恢复复制的 head；旧权重和候选库只读复用。

每折 B 只训练一次 16,964 参数依赖网络。冻结同折 A 的 Flow、修正头、编码器及六层桥，使用 seed=3201、FP32、AdamW、lr/weight decay=1e-4、clip=5、逻辑批 16 窗、warmup 5、余弦周期 100、minimum 30/patience 15/max 100。每轮完整 dev 以 seed=2035 运行实际 K20/20 步 DDIM，仅按全体 JADE 选 rank1。保留全部完整 epoch、best、last、Top5 与完整恢复状态。

四个新折 A/B 全部选定并记录 SHA256 后才进入测试。每折先用观测与地图生成五个 seed（2035–2039）的 A/B K20 完整轨迹并封存全部预测，之后读取测试真值，使用原生 hard metric 和累加器评分。各 seed 独立计分；联合指标按窗等权，边缘指标按有效行人人次加权。新四折和五折汇总均先在同一 seed 内折等权平均，再计算五 seed 均值和样本标准差。无 N≥9 窗口的折标为 N/A，并记录参与汇总的折数。

ZARA2 直接复用固定提交中的逐 seed 测试 CSV，不重新推理。历史 A=epoch5 为旧模型 dev rank2，B=epoch35；新四折 A 均为本折 dev rank1。五 seed 衡量固定权重的推理波动，不代表五次独立训练。

运行入口：`python -m joint_flow_fourfold.controller --push-on-complete`。该入口持有本实验独占锁，GPU 忙时等待，故障从同一目录 last 恢复，完成后自动核验、归档并推送本分支。原始数据、checkpoint、候选库与预测留在训练主机，Git 仅保存代码、配置、报告和匿名指标。

历史耗时推算约 3–4 天；多数阶段达到训练上限时约 7–8 天，外部资源等待另计。这是启动估计，不是完成承诺。
