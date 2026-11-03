# 清理与 SDD 就绪摘要（2026-11-03）

## 清理结果

按精确白名单删除 150 个已完成五折统一实验的冗余中间 epoch checkpoint 和 5 个可重建缓存，失败 0 项。收据计 13,724,430,336 bytes，文件系统实测可用空间增加 13,724,450,816 bytes。正式选中权重、best/last、数据、日志证据、proposal/prediction bank、Goal/GDTS 权重和独立种子任务均保留。

五折选中权重 SHA256 与正式来源清单完全一致，均能读取 328 个有限状态项。UNIV CR 后台评测已正常完成。

## SDD 当前状态

目标固定为 JMM TrajNet-SDD 30/17 协议、8 帧观察/12 帧预测、`dt=0.4 s`。原始像素坐标经 OpenTraj 逐场景近似尺度转换到模型内部米制，论文指标再逆变换回像素。

- validation/test：1999 个窗口、2829 次行人出现、平均人数 1.415208；两者成员镜像，已正式导出。
- train 源轨迹：5720 个窗口、8940 次行人出现、平均人数 1.562937；但 `nexus_2` 缺少经验证的六类语义地图，故正式 train manifest 未导出。
- CPU 预检：导出测试 2/2、padding、增强、EBD 往返、192/4/768 forward/backward 均通过；optimizer updates=0，GPU 使用=0。
- 本地旧 SDD Goal/GDTS 对应 2440/814 窗口的不同协议，未绑定为新协议资产。

因此结论为 **NOT_DATA_READY / ASSETS_PENDING**，不是 TRAIN_READY。启动脚本带硬闸门，缺项时会拒绝运行。必须先用现有六类地图的同一已验证流程补齐 `nexus_2/pred_mask.png`，再导出 train 并制备匹配新 manifest 的 GDTS-trained Goal U-Net、baseline 和 proposal banks。

完整机器可读收据位于：

`RSJG_EBJD_assets/maintenance/cleanup_and_sdd_readiness_20261103_105121/`

其中包含精确删除白名单/收据、空间变化、数据 manifest/statistics、CPU 预检、资产状态和后续唯一命令。
