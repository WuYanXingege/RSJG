# ETH新空间密度模型：单臂有界训练

用户已授权启动，随后明确取消旧模型训练对照。本次只启动新模型，不运行GMM control，也不恢复旧epoch44的100轮任务。

- ETH、training seed3101，所有预测网络参数随机初始化。
- 192/4/768，8层（4＋4），CTF/Raw，rollout与projection OFF。
- 仅训练epoch1–5，总计5个完整epoch；保留100轮学习率时间轴（前5轮warmup），不把scheduler缩为5轮。
- 逻辑microbatch4场景，累积4次；保留原数据、增强及精度设置。
- 每epoch原协议验证一次，seed2035、20world/20step；记录G/H四指标及终点分布诊断。
- 保存完整last、已有selector的best_jade/best_eligible（若存在），完成后额外冻结固定epoch5完整状态。
- 本轮完成状态为BOUNDED_TRAINING_COMPLETE，不代表完成100轮，不自动扩展或再开其他实验。
- 结束报告只使用真实epoch5验证结果，明确单推理seed和validation/test镜像身份；不冒充五种子结果或与旧模型的归因对照。

使用现有nohup＋独立session后台supervisor，复用全局GPU锁和run锁，任一失败保存状态并停止。再次调用恢复脚本时，仅从同身份完整epoch断点恢复，不静默从头重开；已完成时直接退出。

训练期间冻结源码HEAD、配置hash与启动命令。控制台显示Epoch xxx/005，另明确LR timeline=100。完整epoch耗时包含训练、验证、checkpoint保存；ETA针对本次5轮预算。

5轮主要观察可训练性和早期验证趋势，尚不足以判定收敛上限。没有旧模型同条件对照，不能把指标变化严格归因于本次结构修改；是否追加训练留待本次结果后决定。
