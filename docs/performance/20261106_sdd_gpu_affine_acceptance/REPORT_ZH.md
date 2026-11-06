# SDD GPU affine 正式采用验收

## 结论

`compact_gpu_heatmaps_v2_affine + 4 workers` **正式采用**。候选从共同的
epoch 30 完整状态续训一个 epoch 后，seed 2035 验证的 ADE、FDE 均未退化，
训练期 goal BCE 相对增加 0.316%，低于预注册的 1% 门槛；完整训练 epoch
提速 2.704×。因此不触发 seed 2036 补测。

模型、损失、FP32、有效 batch、增强分布、Adam、ExponentialLR 和早停合同
均未改变。Perspective 继续走原 CPU/OpenCV 回退，旧密集缓存继续保留。

## 共同父状态和停止情况

用户授权立即停止时，进程恰好正在直接重写 `last_model.pt`，该文件被截断；
epoch 31 的 `last` 因而不可恢复，epoch 32 的未保存更新也被放弃。最近仍完整
的 `epoch_030.pt` 包含模型、optimizer、scheduler、RNG 和早停状态，恢复后
仅将其角色元数据设为 `resume_last`，作为两臂共同父状态。恢复文件 SHA256：
`0461bad3676e41942b0d0f96ab7c9eacc0501afaa17ba038d9f4f29ebbb878f4`。

## 固定条件短对照

同父状态、相同 12 个 batch、固定 augmentation replay，预热 2 updates 后
执行 12 updates。diffusion 主干的参数差异为 0；数值差异全部位于 Goal
U-Net。

| 对照 | total loss相对差 | goal BCE相对差 | Goal U-Net参数L2相对差 | 主干参数L2相对差 |
|---|---:|---:|---:|---:|
| v3重复 | 1.84e-5 | 6.81e-5 | 2.44e-4 | 0 |
| v3 vs affine | 1.39e-5 | 5.14e-5 | 5.56e-4 | 0 |

候选 loss 差小于本次 v3 自身重复差。8项定向测试继续通过，覆盖非方形输入、
平移、任意角旋转、affine、变换顺序、像素中心/边界和Perspective回退。

## 配对单 epoch 结果

两臂均由 epoch 30 完整状态训练 epoch 31，workers=4；验证沿用现有入口，
inference seed=2035，单位为像素。goal BCE 是本 epoch 的训练均值，不冒充
验证 BCE。

| 指标 | Control v1 | Candidate v2 affine | Candidate-Control |
|---|---:|---:|---:|
| valid ADE | 8.120143 | 8.099901 | -0.020242 (-0.249%) |
| valid FDE | 13.575710 | 13.496391 | -0.079320 (-0.584%) |
| train goal BCE | 0.002528668 | 0.002536663 | +0.000007994 (+0.316%) |
| train total loss | 0.183010942 | 0.183430240 | +0.000419298 |

4-worker完整 epoch 的增强操作计数存在轻微差异，说明多worker调度下不能把
整轮结果解释为逐replay严格位级配对。采用判断因此同时依赖固定replay短对照、
算子空间对齐测试及整轮验证门槛，而不是单纯放宽参数阈值。

## 性能与资源

| 项目 | Control | Candidate | 比值/变化 |
|---|---:|---:|---:|
| 完整训练epoch | 1771.83 s | 655.32 s | 2.704× |
| 固定验证 | 1484.46 s | 1438.37 s | 1.032× |
| 训练+验证 | 3256.29 s | 2093.69 s | 1.555× |
| CPU回退 | 1550/2544 | 259/2544 | 60.93%→10.18% |
| peak CUDA allocated | 6,992,352,768 B | 6,991,330,816 B | -1,021,952 B |
| peak CUDA reserved | 14,881,390,592 B | 14,688,452,608 B | -192,937,984 B |

主要收益来自将 Affine、ShiftScaleRotate 和 Rotate 的热图变换迁移到主进程
GPU；Perspective仍回退。没有新增显存压力。

## 正式衔接

正式训练从 Candidate epoch 31 的完整checkpoint继续，保留其模型、optimizer、
scheduler、RNG和早停状态。旁路 seed 2035 验证没有写回selector或早停计数。
后续仍为：matched-GDTS训练 → baseline评测 → proposal banks → EBJD训练及评测。

机器可读验收结果位于持久资产：
`RSJG_EBJD_assets/experiments/sdd_gpu_affine_acceptance_20261106_160700/evaluation/ACCEPTANCE_SUMMARY.json`。
