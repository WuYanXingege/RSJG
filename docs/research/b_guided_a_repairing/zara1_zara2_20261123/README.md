# B指导A完整轨迹重配：固定两折开发集实验

当前为回放修复状态，尚未形成可交付的最终评分。CPU版本已完成两折1261窗预测和重排；ZARA1评分时A与原CUDA归档的JFDE相差−0.006301993418 m，触发原定1e−4 m一致性检查。24个观测窗的CPU/GPU核对显示：11个多人窗的H0排列不同，13个单人窗一致。

恢复时采用原确定性CUDA配置重新生成A，继续复用已封存B。固定模型、种子、匹配规则和所有判据不变。原CPU预测及已完成的671窗ZARA1评分保留在主机供审计，不用于最终决策。恢复输出位于独立canonical_cuda_recovery目录；两折新版本全部封存后评分。完成后本页自动更新为中文结果报告。

固定起点为0165436a36440d457186e3f62da2e6f4afa8ff70。ZARA1用A epoch15、B epoch6，完整671 dev窗；ZARA2用A epoch5、B epoch35，完整590 dev窗。seed=2035，K=20，12帧；训练更新0、测试访问0，关闭整组交换与1 cm校准。四折控制器保持原计划。

复用上轮封存NPZ中的原B完整轨迹。仅为缺失窗口补齐原A native mode=H、DDIM20、eta=0的完整轨迹。每人独立计算B终点行、A终点列的20×20 float64欧氏距离，固定版本SciPy匈牙利一对一匹配。搬移整条A轨迹，R不再次解码。每人逆置换按字节核验。

两折全部A/B/R预测封存和SHA核验后才读取dev真值。原生hard_metric_statistics与MetricAccumulator评分；逐人minADE/minFDE必须与A精确相等，单人窗四项指标必须与A相同。每折检查全体JADE、全体JFDE、N≥9 JFDE。A>B时要求保留B收益至少90%，否则比例N/A并要求R≤A。按未舍入值判断六项，全部通过才判为“值得后续独立验证”，否则明确失败项并结束此方案。

恢复入口：`CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m b_guided_a_repairing.run --recover-canonical-cuda`。输出会包含两折中文报告、A/B/R绝对值及配对差CSV、匿名逐窗CSV、边缘恒等性、缓存复用数和匹配耗时。权重、原始轨迹、原始置换和行人ID留在训练主机。仅这一规则、这一个推理种子，无后续参数搜索。
