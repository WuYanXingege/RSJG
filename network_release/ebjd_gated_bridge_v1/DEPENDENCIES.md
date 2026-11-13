# 独立性与外部资产

源码不需要旧工作区 PYTHONPATH、完整 GDTS 轨迹网络或历史 pipeline。
主模型依赖 PyTorch、NumPy、SciPy、PyYAML；SciPy 用于真实 Hungarian 双射。
在线原生终点提供器另需匹配的 torchvision。沿用主机已验证环境，本次不安装依赖。
`requirements.txt` 是依赖范围，不代表这些范围内所有版本均做过复现认证；
本次实际环境版本见交付检查收据。

正式运行需要外部资产：

1. 同折 train/validation/test manifest 及其引用的真实场景 shard。
2. 六通道语义 crop、8 帧历史、12 帧未来、有效行人及 scene/agent/frame 身份。
3. 已验证的同折 matched-GDTS Goal U-Net 初始化、SHA256。
4. 训练 bank 与五个验证 seed bank：checkpoint/manifest/scene/agent/单位都需匹配。
5. 同折 matched baseline 的真实结果、checkpoint 与 manifest 绑定。
6. SDD 逐场景像素/米转换；网络内部米制，论文输出按 paper_metric_scale 返回像素。

外部 provider 与主网络的 map U-Net 是两个不同角色：前者冻结，产生 G；
后者可训练，提供上下文和 CTF 地图特征。能够只部署 provider 不等于其权重可用
Goal-pretrain 原样替代；已有 ZARA1/ZARA2 参数与回放核查不支持这种替换。
其他 fold 必须依据各自资产收据，不能外推。

在线 `ebjd/goal_provider.py` 包含最小 Goal U-Net 与原生采样；输入必须是该 fold
原生预处理产生的图像/历史热图和显式坐标回调。本目录不重新实现数据导出或伪造
原生全场景预处理。已有身份绑定 proposal bank 是统一训练入口支持的路径。

旧 optimizer projection、原无条件 trainer、GDTS 训练/数据下载/大缓存转换脚本
没有进入快照。`ebjd/config.py` 删除未被统一入口使用的 `trainer_options`，
将 optimizer 标签规范化为实际的 `adamw`。训练本体没有替换优化器算法。
配置中仍存在部分归档 schema 的非活动字段，由 rollout/projection=false 明确禁用。
JSON配置使用JSON解析器，避免YAML 1.1把JSON科学计数法（例如1e-05）当作字符串；
YAML配置仍沿用原安全解析器。该修正不改变数值或模型计算。

MIT 许可证与本地 U-Net NOTICE 一并保留。数据和权重的许可继续遵循其原始来源；
本目录没有重新分发这些资产。
