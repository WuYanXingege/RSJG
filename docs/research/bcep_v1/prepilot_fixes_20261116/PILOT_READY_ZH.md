# Pilot 就绪结论

**READY_FOR_SEPARATE_AUTHORIZATION**，不是 TRAINING_STARTED，也不是预测精度达标。

已完成：旧问题实证复现、相对几何 bias+value、训练目标不变的监督拆分、agent-micro final NLL选模、身份锁定的成对导出、步边界预算与完整恢复。原8项+新增17项CPU测试通过；真实train/dev各一窗口完成五seed无梯度接口检查。未启动队列。

冻结配置：`configs/research/bcep_prepilot_v2.yaml`，allow_real_training=false；3790/320不变，10epochs、37900updates、3600秒累计流程墙钟、patience3、seed3101、lr1e-4、lambda_aux0.25、M256/K20。训练入口已实测拒绝本配置启动真实训练。

尚未覆盖：GPU数值与恢复、真实训练收敛、320窗口完整dev预测质量、外层泛化、其他密度对照、多训练seed。CPU合成验收不能替代这些证据。旧八层或GDTS权重不作为BCEP初始化，旧未训练BCEP原型的state_dict不直接迁移到新增edge参数版本。

后续必须先取得单独训练授权，再冻结允许运行的新配置、输出目录及执行设备；不在当前提交中自行开启开关。未来按最终world NLL agent-micro选择密度checkpoint，只在同checkpoint内部比较IID20与coverage20。独立外层测试不能用于选择或调参。
