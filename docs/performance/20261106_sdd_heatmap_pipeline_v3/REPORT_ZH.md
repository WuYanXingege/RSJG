# SDD matched-GDTS 紧凑热图输入管线 v3

## 结论

采用 `compact_gpu_heatmaps_v1`。正式模型、损失、数据身份、batch 顺序规则、增强分布和优化器均未改变。训练从完整 epoch 23 状态恢复；用户要求立即停止时，epoch 24 的 282/2544 个未保存 update 已明确作废。

## 实际修改

- 将 train/valid/test 共 4,172 个缓存流式转换为紧凑缓存，只保留坐标、身份、地图和必要元数据，不再逐 batch 保存 `[B,T,H,W]` 密集高斯热图。
- 压缩缓存由 54,506,525,205 bytes 降为 750,449,055 bytes，约 72.63 倍缩小；原缓存保持不变。
- 无重采样的翻转、转置和 90° 旋转路径，在主进程 GPU 按旧公式逐值重建高斯热图并应用相同 replay。
- 透视、仿射、平移和任意角旋转会触发精确 CPU/OpenCV 回退，从旧缓存读取原热图，避免以近似 GPU 插值改变监督语义。该回退是当前正确性与速度之间的正式选择。
- validation/test 不做增强，均由 GPU 从紧凑坐标逐值重建原热图。

## 正确性

- 原始高斯重建与旧缓存逐值一致（`torch.equal=True`）。
- 固定相同 batch 和 augmentation replay 的 8-update 对照：
  - total loss 相对差 `3.45e-5`；
  - diffusion loss 相对差 `0`；
  - goal BCE 相对差 `1.36e-4`；
  - 参数 L2 相对差 `1.16e-7`；
  - 参数、loss、梯度均有限并正常更新。
- 某一步梯度范数相对差为 `4.18%`；但 legacy 对 legacy 的同位置重复波动为 `10.92%`，因此该单点门槛在当前 CUDA/增强执行下不能区分实现偏差。其余候选步骤最大差约 `0.115%`，总 loss 与参数对照通过。

## 性能

100 个真实 optimizer update，10 次预热，正式增强开启：

| 复测 | legacy s/update | compact s/update | 提速 |
|---|---:|---:|---:|
| 1 | 1.054425 | 0.837849 | 1.2585× |
| 2 | 1.053956 | 0.795971 | 1.3241× |

两次中位提速约 `1.291×`，单 update 墙钟下降约 `22.6%`。DataLoader 阻塞由 `0.750–0.768 s/update` 降至 `0.533–0.575 s/update`。峰值 CUDA allocated 基本不变（约 `6.51 GiB`），reserved 为 `13.38–13.41 GiB`；主进程与 worker RSS 总量没有增加。

按 2,544 update/epoch 外推，纯训练由约 44.7 分钟降至约 34.6 分钟；这是短窗口估计，首个恢复后完整 epoch 的真实数据由后台收据记录。

## 运行边界

- checkpoint 仍 strict 兼容，恢复模型、optimizer、scheduler、RNG 和早停状态。
- patience 仍为 3 次验证；验证每 10 epoch，minimum epoch 50，未改变。
- Goal-pretrain 没有重跑。
- 大缓存、checkpoint 和 benchmark 原始 JSON 位于持久资产目录，不进入 Git。
