# SDD matched-GDTS 输入管线 v4

## 结论

正式采用 `compact_gpu_heatmaps_v1 + 4 workers`，不启用 GPU affine
候选。模型、损失、FP32、batch、增强分布、优化器、scheduler 和早停
合同均未改变。

本轮按用户明确要求立即暂停。恢复点为完整 epoch 30，checkpoint SHA256
为 `1bdc3f5518ba963ee6c442713ccc7fcd4c923b41ba0bc2799c95fb80f7ed2deb`；
epoch 31 已执行的 376/2544 个未保存 update 作废，不被描述为可恢复进度。

## 正式采用的改动

- 在紧凑缓存路径重新比较2/4-worker，正式设为4-worker。
- 每个worker限制 PyTorch/OpenCV 内部线程为1，避免嵌套线程竞争。
- `prefetch_factor=2`；不使用 persistent workers，保留旧训练每轮重建
  worker及其RNG生命周期。
- 新增低开销逐epoch路由统计：GPU快路径数、CPU回退数、操作类型、
  地图尺寸和worker准备耗时。统计不触发CUDA同步。
- 新增按执行配置命名的首个完整epoch耗时收据。

## 2/4-worker性能

同一 epoch-30 checkpoint、相同100个batch及固定augmentation replay；每条
路径先预热10个update。单位为秒/update。

| 复测 | 2 workers | 4 workers | 提速 |
|---|---:|---:|---:|
| 1 | 0.812890 | 0.662604 | 1.2268× |
| 2 | 0.759428 | 0.664093 | 1.1436× |

中位提速 `1.1852×`；DataLoader等待分别从
`0.5766/0.5212` 降至 `0.4065/0.4102 s/update`。峰值 CUDA allocated
均为 `6,992,979,968 bytes`，未增加；reserved 均约 `14.92 GB`。

主进程峰值RSS约 `2.37 GB`，单个child的峰值约 `2.43–2.44 GB`。
4-worker会增加潜在并发主存占用，但短测期间系统可用内存约26 GiB，未发生
交换抖动或OOM。按短窗口直接外推，纯训练约28.1分钟/epoch；按epoch 30
真实训练35.35分钟除以短测提速，约29.8分钟/epoch。两者均为预计值，正式
恢复后的首个完整epoch收据才是实际值。

## GPU affine候选及回退比例

候选 `compact_gpu_heatmaps_v2_affine` 实现了：

- `Affine`（包含shear/translate）；
- `ShiftScaleRotate`平移；
- 任意角 `Rotate`；
- 多次重采样按原replay顺序执行，不合并插值；
- 共享逆向采样计划并按128通道分块，控制临时显存；
- `Perspective`继续精确CPU/OpenCV回退。

100-update固定replay中，CPU回退由 `39/100` 降至 `13/100`，相对下降
`66.7%`。4-worker下，v3为 `0.6626/0.6641 s/update`，候选为
`0.4069/0.4050 s/update`，候选速度为 `1.6283×/1.6399×`。

但该候选未进入正式配置。12-update对照中：diffusion loss差为0，total
loss相对差 `1.08e-5`，goal BCE相对差 `4.01e-5`，梯度范数最大相对差
`3.08e-4`；这些均通过。参数L2相对差为 `1.99e-4`，超过预设参考门槛
`1e-5`。重复性校准中v3-v3自身为 `7.85e-5`，说明绝对门槛在当前执行环境
下不能严格区分；但v3-v4校准仍为 `1.65e-4`，高于基线路径波动。因此按
保守原则，Affine/Rotate GPU实现保持benchmark-only，正式CPU回退比例不因
该候选改变。

## 验证

- 8项定向单元测试通过，覆盖非方形输入、三类affine、顺序重采样、
  exact-index变换和Perspective回退。
- v3正式路径仍使用原密集缓存和CPU/OpenCV处理所有复杂重采样，输出语义
  不变。
- GPU affine模式被入口显式限制为benchmark-only，不能误用于正式训练。
- 临时正确性状态文件已删除；checkpoint和大缓存未提交Git。

机器可读结果位于持久资产：

`RSJG_EBJD_assets/experiments/sdd_heatmap_pipeline_v4_20261106_145000/receipts/BENCHMARK_SUMMARY.json`

正式恢复后由以下文件记录逐epoch路由与首轮真实耗时：

- `receipts/INPUT_PIPELINE_EPOCH_STATISTICS.json`
- `receipts/INPUT_PIPELINE_FIRST_FULL_EPOCH_compact_gpu_heatmaps_v1_w4.json`
