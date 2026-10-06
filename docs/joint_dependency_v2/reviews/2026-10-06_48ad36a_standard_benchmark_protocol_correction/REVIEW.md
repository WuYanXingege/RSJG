# 标准 benchmark 协议更正与当前运行重新分类

结论：**接受协议更正。** 当前 `p2_grouped_univ_hotel_official_prior_v1` 训练继续保留为 grouped-UNIV 辅助实验，但不再作为 GDTS 标准 benchmark 主线，也不能与 GDTS、MID、LED 等论文公开数字作严格排名。论文主结果应恢复 ETH/UCY 标准 leave-one-scene-out 五折，并让本地重训 GDTS 与新方法共享逐折完全相同的训练、选模、测试和指标实现。

本轮只做只读核对和实验定位更正：**不停止当前训练、不启动新训练、不访问 outer target、不生成标准折 checkpoint。**

## 直接证据

1. [GDTS arXiv v3](https://arxiv.org/html/2311.14922v3) 明确写明 ETH/UCY 使用五个场景的 leave-one-scene-out：四个场景训练、剩余一个测试；观察 8 帧、预测 12 帧，并报告 best-of-20 ADE/FDE。论文还给出 batch64、Adam 初始学习率 `1e-3`、指数退火、goal 150 epochs、joint 250 epochs、最终预测数20。
2. [GDTS 官方 train.sh](https://github.com/Winderting/GDTS/blob/297d508558c10831983ea4b19c2b3e657459a449/train.sh) 在固定 commit `297d508558c10831983ea4b19c2b3e657459a449` 上设置 joint `num_epochs=250`、`batch_size=64`、`start_validation=1`、`validate_every=10`、`learning_rate=0.001`，并列出 ETH/HOTEL/UNIV/ZARA1/ZARA2 五个 test-set 入口。
3. 当前冻结配置 [FRESH_JOINT_CONFIG.yaml](../2026-10-06_6d04a87_official_prior_fresh_joint/FRESH_JOINT_CONFIG.yaml) 是 HOTEL test-set 名义下的自定义 grouped 协议：UNIV 整组作为 inner、UNIV 不参与参数更新，joint lr `1e-4`，每轮运行完整 inner，另有 48 小时上限。

因此，“保留 GDTS 网络与损失”只能称为 **同一架构下的新协议基线**，不能称为标准 GDTS 复现。最关键的混杂来自训练成员和选模数据改变；学习率、验证频率与预算差异进一步扩大了不可比性。

## 当前运行的正确定位

当前 worker PID/SID `1115621` 仍正常运行。核对时已完成 epoch 1：175 次更新、11,122 exposures、完整 inner 391/391，随后进入 epoch 2。epoch 1 inner ADE 为 37.9624 原生像素，仅是该自定义协议的选模量，不是标准 ETH/UCY 米制 ADE，也不能与论文表格数字比较。

该运行仍可支持：

- grouped-UNIV 协议内的复现实验；
- 未来共享同一 base/cache/初始化/预算的 A0 与 A1 配对目标比较；
- 整场景 inner 选模下的稳定性与机制分析。

它不能支持：

- 与 GDTS、MID、LED 公开主表的严格优劣结论；
- 标准 HOTEL 折复现；
- 将差异全部归因于模型或目标创新；
- 作为标准 UNIV test fold 的父权重，因为 goal 与 joint 均使用 UNIV 做过选模。

## 主实验恢复顺序

1. **固定标准协议来源。** 锁定 GDTS 论文版本、官方代码 commit `297d508...`、作者数据包与预处理文件哈希；逐项审查实际 loader，而不是只从场景名推断成员。
2. **先认证标准 HOTEL 折。** 精确导出 train/validation/test 窗口和行人成员，确认四个非 HOTEL 场景如何进入训练、官方 validation 的真实来源、8→12 窗口、坐标换算、有效样本掩码及 ADE/FDE 聚合。
3. **独立训练标准父权重。** 标准 HOTEL 的 goal 与 joint 都须重新训练；不得复用已由 UNIV inner 选模的当前 epoch 96 goal 或当前 joint。
4. **先复现 GDTS，再接入方法。** 使用同一标准 fold、seed 计划、初始化策略、预算和评测代码分别训练本地 GDTS 与新方法；公开论文数字和本地重训数字分栏报告。
5. **扩展五折并统一评测。** HOTEL 通过后扩展 ETH、UNIV、ZARA1、ZARA2，每折拥有独立 goal/joint 父权重，报告逐折和五折平均 ADE/FDE@20。
6. **联合指标作为附加主张。** 在相同测试窗口、相同采样数和相同坐标单位下，对各方法实际预测统一计算 JADE/JFDE；不得从公开 ADE/FDE 反推。

## Go / no-go

- **GO：** 当前训练继续作为明确标注的辅助协议运行；其历史结果全部保留。
- **GO：** 下一步先做标准 HOTEL 的只读数据/loader/metric 认证和独立配置设计。
- **NO-GO：** 当前 checkpoint 进入标准五折主表、与论文公开数字直接排名、自动进入 cache/A0/A1。
- **NO-GO：** 在没有逐成员与指标身份认证前直接启动所谓“标准 HOTEL”训练。

机器可读结论见 [RESULTS.json](RESULTS.json)。
