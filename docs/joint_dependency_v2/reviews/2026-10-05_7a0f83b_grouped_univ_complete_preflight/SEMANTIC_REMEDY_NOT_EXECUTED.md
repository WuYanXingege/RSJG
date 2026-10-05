# 唯一输入来源阻断的具体补件 / 替代方案

优先补件：作者提供 `pred_mask.png SHA → segmentation checkpoint SHA → 训练 scene / label 清单 →
对应生成命令与版本` 的可核对映射。关键是证明该分割模型未使用本新协议 UNIV inner / HOTEL outer 的训练标签、
未来轨迹或禁止的统计，不是重新证明作者 ZIP 存在。证明满足后更新新资格 manifest，重跑真实生产 loader/inner
selection 与实际入口 check；不能只改 status 字符串。

若补件不可得，唯一具体替代建议是单独批准新的输入版本：仅用 train 场景 ETH/ZARA1/ZARA2 的
RGB 与合法 semantic 训练标注训练一个分割预处理器，保持现有六类映射；冻结后对 inner/outer 场景 RGB 做推理。
其训练输入、checkpoint、标签集、输出 raster 全部 SHA 绑定，验证 holdout 标签/轨迹从未进入分割训练。
推理时不得把 UNIV/HOTEL 的 GT semantic 当输入。需要另行核查训练标注许可、训练预算和模型生成来源。

这不是本轮自动实施项目，也不是把原主线计为 READY。它属于新的输入来源/产物版本，必须独立命名和重建
observation wrapper 的资产绑定；不覆盖本轮旧 raster 或历史结果。现有 Gaussian/目标可以在数学内容不变且
新资产尺寸/坐标合同逐项验证后复用，但不得只改 manifest 指向新 raster 而留下旧 payload 的 tensor_image。

本轮不把六通道置零、不换 RGB、不训练分割网络、不使用 oracle semantic。
