# 正式 fresh goal 启动合同

配置：[FRESH_GOAL_CONFIG.yaml](FRESH_GOAL_CONFIG.yaml)。训练入口在启动前重新核验 `GOAL_TRAINING_READY`、manifest/data binding、配置 SHA、当前源码、CPU/CUDA/reload、完整生产 loader、完整 inner selection 和 fresh 初态。

```bash
../.conda/rsjg/bin/python -B tools/official_prior_launch.py \
  --archive docs/joint_dependency_v2/reviews/2026-10-05_2c25c1f_official_prior_fresh_goal \
  --output outputs/joint_dependency_v2/official_prior_fresh_goal_2c25c1f_20261005/formal_launch
```

启动器使用 `setsid + nohup` 派生独立 worker，日志写入 `formal_launch/fresh_goal.log`。worker 在查询 GPU 前再次执行全部只读身份检查，并拒绝已有输出目录、繁忙 GPU 或低于 2 GiB 的可用预算。训练最多占用 10 GiB GPU memory fraction，CPU 线程为4，关闭 TF32 和 cuDNN benchmark。

正式配置：FP32、seed3101、batch64、Adam lr0.001、clip1、ExponentialLR gamma0.99、最多150轮；每轮 train175 packs/11122 exposures 后执行 inner391 packs/24955 的 masked BCE 选模，严格保留更早 tie。24小时在 pack 边界检查；partial epoch/selection 不推进 scheduler，也不产出可作为父产物的完整 epoch checkpoint。

输出目录：

```text
output/hotel/runs/grouped_fresh_goal_official_prior_seed3101_20261005_114637/
```

本启动器只运行 fresh goal。完成并认证 selected goal 之前，fresh joint、cache 和 A0/A1 均保持等待，不自动启动。
