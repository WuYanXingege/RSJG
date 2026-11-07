# SDD matched-GDTS 训练验证与冻结评测一致性核查

## 结论

同一份 epoch70 模型在两种实际评测成员上得到不同数值，主差距不是
推理 seed 造成的。正式 baseline 冻结为 **JMM 同步 scene-window、五推理
seed** 口径：minADE `9.075967 ± 0.018871 px`，minFDE
`15.528292 ± 0.046028 px`。旧训练曲线 `8.015944/13.343094 px` 仅作为
legacy packed 训练诊断，不再与 EBJD 正式结果混用。

冻结权重为 `best_model.pt`，内部 epoch=70，SHA256：
`0a90d3db339ec38e62034dcef458a355d902af765241c8db3f95eaa567389dbf`。
`epoch_070.pt` 的文件哈希不同是因为它额外保存完整恢复元数据；两者的
112 个 `model_state_dict` 张量逐值完全一致。

## 同权重 2×2 回放

| layout | seed | physical items | 有效行人出现 | minADE (px) | minFDE (px) |
|---|---:|---:|---:|---:|---:|
| packed | 3101 | 814 | 51,513 | 8.011317 | 13.344287 |
| packed | 2035 | 814 | 51,513 | 8.010141 | 13.349923 |
| scene | 3101 | 1,999 | 2,829 | 9.091520 | 15.614754 |
| scene | 2035 | 1,999 | 2,829 | 9.102441 | 15.544203 |

同 layout 的 seed 效应：

- packed，2035−3101：ADE `-0.001176 px`，FDE `+0.005635 px`；
- scene，2035−3101：ADE `+0.010921 px`，FDE `-0.070551 px`。

同 seed 的成员/layout 效应：

- seed3101，scene−packed：ADE `+1.080203 px`，FDE `+2.270467 px`；
- seed2035，scene−packed：ADE `+1.092300 px`，FDE `+2.194280 px`。

因此 seed 波动不足以解释约 13%–16% 的稳定差距，主要差异来自两套
评测成员与场景组织根本不同。

## 身份与指标核查

- packed cache 实际含 51,513 个 `(sequence,start_frame,agent_id)` 唯一出现，
  与 JMM validation manifest 的 2,829 次出现不匹配；它是 independent-GDTS
  的单轨迹滑窗集合，而非 1,999 个同步窗口的物理重排。
- scene cache 含 2,829 次出现、1,999 个同步 window ID，与 manifest 多重集合
  完全一致，无缺失、额外或重复。
- 三次新增回放中，原 minADE/minFDE 函数与独立像素公式的逐行人最大绝对差
  均为 `0`。ADE/FDE 分别独立 best-of-K，随后按所有有效行人全局平均，未做
  batch 均值的二次平均。
- 实际采样为 K=20、DDPM=100、DDIM=20、trunk stage=30，`net.eval()`、
  `if_test=True`，单位为像素。

packed/3101 的当前 legacy 回放相对旧训练日志仍有 ADE `0.004627 px`、FDE
`0.001193 px` 的微小差异。训练验证使用
`compact_gpu_heatmaps_v2_affine`，冻结入口使用 legacy 密集输入；执行路径差异
和 CUDA 数值波动没有进一步逐项拆分，因此不宣称旧日志被逐值复现。该差异
仅约 0.058%/0.009%，远小于 scene/packed 主差距，不影响冻结口径判断。

## 冻结的统一协议

- checkpoint：上述 epoch70 权重；
- manifest SHA256：
  `7f2501aec2bc66d97e3511ef9aad94c8eb5afe385cd7f8fec982598ffdaa3c28`；
- 每个物理评测 item 对应一个同步 manifest window，场景隔离；
- K=20，DDPM=100，DDIM=20，trunk=30；
- inference seeds：2035–2039；
- minADE：逐行人先对未来12帧求均值，再独立对 K 取最小；
- minFDE：逐行人在最终帧独立对 K 取最小；
- 数据集归约：所有有效行人全局平均；单位：pixel。

完整逐行人误差、误差 bank、身份审计和机器可读冻结协议保存在持久资产：
`RSJG_EBJD_assets/analyses/sdd_gdts_eval_consistency_20261107_103701/`。

## 已确认与未排除

已确认：同模型张量、packed 与 JMM scene 不是相同成员、seed 影响很小、指标
归约正确、scene 身份完全匹配 manifest。

未进一步拆分：训练期 compact 输入与冻结 legacy 输入之间的千分之一级差异；
在样本集合已明确不同后，不再把剩余差值无证据归因于 TTST、扩散随机流或
batch 相关网络行为。
