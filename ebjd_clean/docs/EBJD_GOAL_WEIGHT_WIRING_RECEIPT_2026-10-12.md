---
experiment_id: EBJD-GW-B-261012-001
date: 2026-10-12
system: EBJD
type: training_objective_wiring_repair
dataset: HOTEL
training_seed: 3101
checkpoint: epoch_045.pt
checkpoint_sha256: 425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1
source: 87e99666ab71b17cecc87fade85d4898deda62e8
execution_commit: fb7fb8c319db8c3fdfe449e6293a4436e22514c1
status: GOAL_WEIGHT_WIRING_READY
device: NVIDIA GeForce RTX 5070 Ti
tags: [goal-weight, wiring, exact-resume, CUDA-preflight, paired-pilot]
---

# EBJD 终点权重接线修复收据

## 结论

`objectives.goal_v_weight` 已真实接入配置、`TrainerOptions`、训练调用和 diffusion loss。`wg=2` 现在严格计算：

`wg*L_goal_fine + L_bridge_fine + 0.25*(wg*L_goal_coarse + L_bridge_coarse)`。

默认 `wg=1` 保留原浮点运算顺序、模型 state schema 和参数数量。桥权重仍为 1，粗头系数仍为 0.25；没有修改网络、map BCE、rollout 目标、投影、RMS scale、latent/noise/target 或采样协议。

上一轮诊断报告中“短程训练入口增加后即可执行”的表述不完整：固定配置校验、TrainerOptions、trainer 调用和 loss 计算当时没有形成闭环。本收据完成该勘误。上一轮冻结 forward 与 CPU bank 诊断没有训练，因此不受此接线缺口影响，结果继续有效。

## 修复范围

- `config.py`：只接受数值 `1.0/2.0`；拒绝 bool、NaN/Inf 和其他值。goal weight 保留在 semantic hash 内。
- `TrainerOptions` 与 `trainer_options()`：显式传递 `goal_v_weight`。
- `objectives.py`：终点 fine/coarse 独立加权，bridge 保持不变；返回未加权诊断分项。
- `trainer.py`：真实 loss 调用传入权重；记录四个未加权分项、weighted diffusion、active weight、总梯度范数、scene 顺序与已生成随机张量摘要。
- `train.py`：增加持久 `absolute_stop_epoch=50` 合同、epoch46–50 全 checkpoint、每轮资源摘要和恢复安全边界；short joint score 修正为 `JADE+0.5*JFDE`。
- materializer：两臂强制使用 `small_group_uncheckpoint_v1` 认证 profile，并分别记录 identity、execution 与唯一 semantic 差异。

这些是向后兼容的可选入口；不提供新参数时，原训练配置语义保持不变。

## CPU 验收

完整测试为 `60 passed`，覆盖：

- `load_config → trainer_options → TrainerOptions → actual diffusion_loss` 真实链路；
- `L(2)-L(1)=L_goal_fine+0.25*L_goal_coarse` 独立数学参考；
- fixed tensor 上 fine/coarse goal 输出梯度恰好加倍、bridge 梯度不变；
- mask/padding、不同人数、粗头和尾批归约；
- 非认证权重与额外 LR/objective/world/step 变化拒绝；
- semantic hash、认证执行 profile、epoch46/50 的原 100 轮 LR 时间轴和绝对停止边界；
- short selector 不混入 epoch45，模型 state schema 与 3,612,764 参数不变。

## 真实 CUDA 验收

固定 epoch45 checkpoint、相同真实 batch 和相同已生成 draws，分别比较 `wg=1/2`：

| 工作负载 | 路径 | 公式最大误差 | draws | rollout checkpoint | 结果 |
| --- | --- | ---: | --- | --- | --- |
| N=[1,2,5,3] | small-group | ≤5.56e-8 | 完全相同 | 关闭 | PASS |
| N=57 | dense fallback | ≤3.73e-8 | 完全相同 | rollout 时开启 | PASS |

两种 workload 的 rollout off/on 均通过。四个未加权 diffusion 分项、rollout、soft marginal ADE/FDE 在两臂完全一致；差异只来自 endpoint 权重。U-Net、denoiser、future geometry 和 map injection 总梯度范数均非零。AdamW proposal、clip、有限参数更新及 FP64 投影均通过，最小投影点积大于 `−1e-6` 容差。

模型、optimizer moments/steps、Python/NumPy/Torch CPU/CUDA/loader RNG 和 update=11970 在每个隔离副本中精确恢复；恢复 LR 为 `6.265153161346956e-5` 和 `6.265153161346956e-6`。诊断更新只存在内存，保存更新数为 0。源 checkpoint SHA256 验收前后均为 `425988…e1b1`。

结构化收据见 [GOAL_WEIGHT_REAL_CUDA_PREFLIGHT.json](validation/goal_weight_pilot_20261012/preflight/GOAL_WEIGHT_REAL_CUDA_PREFLIGHT.json)。

## 重尾勘查

五推理 seed 先在精确 source/window/agent occurrence 上平均，再按同一 track 连续减速标签聚合。最差窗口簇主要来自 `biwi_hotel` agent `288`：6 个连续减速窗口的有符号贡献为总体 minFDE 净差的约 74.36%；整条 track 贡献约 113.46%，超过 100% 是因为其他 occurrence 的改善发生抵消。

排除最强 1/3/5 个减速事件后的诊断均值分别为 `+0.00269/−0.00099/−0.00164 m`。这只是敏感性分析，官方指标仍保留全部 1,197 occurrences。top tracks 的重叠位置差为 0，没有 >5 m 的明显跳变。事件是连续标签启发式聚合，不称为统计独立或物理独立样本。

详情见 [重尾结构化结果](validation/goal_weight_pilot_20261012/preflight/tail_events/EPOCH45_HEAVY_TAIL_EVENT_SUMMARY.json)。

## 放行边界

接线工程验收已通过，可以 materialize 两臂并启动严格串行 pilot。该状态不代表 `wg=2` 已改善正式指标；效果结论只能由 epoch50 的冻结五 seed B1/P20 评价给出。
