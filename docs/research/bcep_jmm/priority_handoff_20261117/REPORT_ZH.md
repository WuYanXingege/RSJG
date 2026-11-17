# 正式五折优先：旧评测收尾、GPU验收及启动收据

快照（北京时间）：2026-11-17T16:57:21.063702+08:00。

**旧有界试验已按新授权部分收尾；新JMM五折正式流程已实际启动。** 当前为ZARA2从零训练BCEP Provider，训练seed3102；后续为该折冻结Provider及候选导出、随机初始化六层EBJD3201，再依次ZARA1、HOTEL、UNIV、ETH。不是五折训练已完成。

## 旧任务收尾

现场通过命令行、工作目录、进程启动时钟核对PID1878407；它是单一GPU协调/评测进程，父为systemd用户管理器、处于IDE scope，没有额外GPU worker或重启supervisor。

现场已经进入Flow3103 seed2036，因此没有回滚到2035。外部inotify监视器等待2036最终文件原子rename，随后SIGSTOP固定边界，校验ZIP完整性及SHA256，再对这个精确PID发送SIGTERM/SIGCONT，确认进程退出和GPU清空；没有热修改源码，没有全局pkill。

- Flow3102：2035–2039，共5份完整G/H轨迹。
- Flow3103：2035、2036，共2份完整G/H轨迹；2037–2039取消。
- 每份998窗口、5910有效行人出现、K20/T12、米制坐标。完整文件数7，不能称为双组完整五seed评测。
- 保留112个已存在文件及外部绑定资产，含两组best/last完整状态、Provider、候选bank、轨迹与日志；SHA清单见PRESERVED_ASSET_MANIFEST.json，删除数0。训练状态文件及best/last与停止前一致。
- 原PIPELINE_STATE中的RUNNING仅为历史快照；独立EVALUATION_STOP_RECEIPT.json和本次范围变更收据登记真实停止。没有改写原训练状态。
- CPU评分后置，避免占用正式GPU验收的CPU/I/O；本报告不混合5seed与2seed生成不平衡的双组均值。

## GPU验收

临时模型使用真实官方输入，覆盖有效人数[1, 3, 57]，256×256地图。没有读取旧学习权重作为正式初始化。临时候选仅用于接口/计算量核验，不是正式proposal bank或研究结果。

- Provider数值/梯度：{'loss_reference': 2.7778615951538086, 'loss_candidate': 2.7778615951538086, 'gradient_relative': 7.00197054431378e-08, 'passed': True}。
- 分组更新数值/梯度：{'loss_reference': 12.50845193862915, 'loss_candidate': 12.508452415466309, 'gradient_relative': 2.1910135661764228e-07, 'passed': True}。
- H-only与原G/H的H输出：{'passed': True, 'max_abs': 0.0}。
- 完整更新和最高人数容量检查未出现OOM；GPU验收墙钟1020.1秒，计入临时更新/验证/保存，不计入正式训练进度。

| 阶段 | 配对轮次 | 原路径秒/update | 候选秒/update | 比值 | 峰值allocated GiB 原/候选 |
|---|---:|---:|---:|---:|---:|
| provider | 1 | 0.127794 | 0.063726 | 2.005× | 0.603 / 0.603 |
| provider | 2 | 0.124918 | 0.064981 | 1.922× | 0.603 / 0.603 |
| joint | 1 | 8.014599 | 6.776501 | 1.183× | 2.131 / 2.131 |
| joint | 2 | 8.031352 | 6.981996 | 1.150× | 2.131 / 2.131 |

这是既有短验收窗口：每路径1次冷启动、1次预热、3次稳态完整update，重复两轮；EBJD每update仍为4场景microbatch×4累积。数量有限，不能称为长期epoch提速。运行时采纳要求数值通过且两次均≥1.03×；不同场景规模、正式epoch收益尚待实际训练测量。

运行时选择：{'provider': {'speedups': [2.005377476927856, 1.9223890445890914], 'adopt': True}, 'joint': {'speedups': [1.1827044811577463, 1.1502944632213885], 'adopt': True}}。未获采纳的可选优化保持原FP32路径。H-only用于每epoch既定H选模验证；最终G/H评测不缩减。

完整590窗口验证工作量计时：[{'mode': 'provider', 'windows': 590, 'agents': 4262, 'seconds': 26.456117389025167, 'temporary_candidate_interface': True, 'not_validation_quality': True}, {'mode': 'GH', 'windows': 590, 'agents': 4262, 'seconds': 558.280917316908, 'temporary_candidate_interface': True, 'not_validation_quality': True}, {'mode': 'H', 'windows': 590, 'agents': 4262, 'seconds': 287.43567122588865, 'temporary_candidate_interface': True, 'not_validation_quality': True}]。这些来自临时随机模型，只反映工作量，不能作为验证质量。

## 正式启动

- 执行源码固定提交：`bba1a349b8865963a0080a03b39b38e96d4918ad`。
- 源码哈希：`77b2e3c489cc1648ccf94095d975275369b57d8053cc629e043536a0aff6993c`；运行配置/划分/计划哈希见EXECUTION_IDENTITY和RUNTIME_FROZEN。
- 实际进程：1892323；阶段：PROVIDER，fold=zara2。
- 首个真实更新：epoch1/update1，loss=4.57885599；loss、梯度、参数有限性检查通过。
- 当前进度快照：epoch=1，update=2200。
- 完整模型/optimizer/scheduler/RNG与顺序/选择历史由既定原子checkpoint保存；本次没有继承旧试验的optimizer或权重。
- Provider最多100轮/最少20轮/patience15，原Adam与ExponentialLR gamma1.0；EBJD最多100轮/最少30轮/patience15、5轮warmup与100轮余弦合同不变。
- `nohup+setsid`独立后台运行，原评测无存活进程，只有一个正式GPU任务。
- 日志：`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/bcep_jmm_full_twostage_20261117_144947/logs/pipeline.log`。
- 首更新：`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/bcep_jmm_full_twostage_20261117_144947/folds/zara2/provider/FIRST_UPDATE.json`。
- 状态：`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/bcep_jmm_full_twostage_20261117_144947/PIPELINE_STATE.json`。

训练尚处于开始阶段，不报告最终指标，不声称五折完成。此报告独立结果分支发布，不移动正在运行工作区的源码HEAD。
