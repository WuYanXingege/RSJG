# Fresh joint 启动合同

冻结配置：[FRESH_JOINT_CONFIG.yaml](FRESH_JOINT_CONFIG.yaml)。启动前会重新核验 `JOINT_TRAINING_READY`、manifest/data binding、父权重 epoch 96 与 SHA256、配置 SHA、当前训练源码指纹、fresh 初态、CUDA/reload 和完整 inner receipts。

```bash
../.conda/rsjg/bin/python -B tools/official_prior_joint_launch.py \
  --archive docs/joint_dependency_v2/reviews/2026-10-06_6d04a87_official_prior_fresh_joint \
  --output outputs/joint_dependency_v2/official_prior_fresh_joint_6d04a87_20261006/formal_launch
```

启动器使用 `setsid + nohup` 派生独立 worker，日志写入 `formal_launch/fresh_joint.log`。worker 在访问 GPU 前再次做只读身份检查，并拒绝重复启动、已有正式 run 目录、繁忙 GPU 或低于 2 GiB 的可用预算；进程最多配置 10 GiB GPU memory fraction、4 个 CPU 线程，关闭 TF32 与 cuDNN benchmark。

正式配置：FP32、seed3101、batch64、Adam lr `1e-4`、clip1、ExponentialLR gamma0.99、最多250轮/43,750次成功更新/48小时。每个完整 train epoch（175 packs）后执行完整 inner（391 packs/24,955 samples）的 P20 原生像素 ADE 选模，严格保留更早 tie。

48 小时先到时，仅在至少一个 epoch 已完整训练、选择并原子保存后，才以 `wall_budget_partial_discarded` 正常收尾；当前未完成 epoch/selection 被丢弃。零完整 epoch 则失败，不生成可作为父产物的完成收据。

输出目录：

```text
output/hotel/runs/grouped_fresh_joint_official_prior_seed3101_20261006_023912/
```

本命令只运行 fresh joint，不自动启动 cache、A0 或 A1。
