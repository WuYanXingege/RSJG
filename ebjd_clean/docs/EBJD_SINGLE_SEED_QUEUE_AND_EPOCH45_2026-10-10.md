# EBJD 单训练种子队列与 HOTEL epoch 45 权重收据

时间：2026-10-10 14:14:46 +08:00

## 结论

按用户决定，实际训练队列由三个训练种子收缩为每个数据集一个训练种子，HOTEL 与 UNIV 均只保留 `seed 3101`。五个推理种子 `[2035, 2036, 2037, 2038, 2039]` 不变，它们用于固定 checkpoint 的推理波动评估，不会启动五次训练。

HOTEL `seed 3101` 继续当前运行；原三种子调度父进程已冻结，因此完成后不会进入 3102 或 3103。独立完成监视器将在确认 `last.pt` 到达 epoch 100 后，把 HOTEL 单种子队列标为 `COMPLETED`。UNIV 单种子等待器随后执行正式最大验证 batch 的 CUDA 预检，通过才启动 `seed 3101`。

## HOTEL 快照

- 训练进程：PID `1263016`，唯一 GPU 训练进程。
- 队列状态：`RUNNING`，仅有 `seed 3101`。
- 快照时训练已进入 epoch 49，连续更新编号至少到 `12847`。
- 最近完整 checkpoint 与验证边界：epoch 48，`last.pt` SHA256 `f2c5a8a552e12b48825bccce6768826991fd4b4e409b0017ad024cb48904330f`。
- epoch 45 仍是当时的最小约束违例候选，但没有满足相对 GDTS baseline 的零容差同时改善条件，因此其角色是 `least_violation`，不是 `best_eligible`。

## epoch 45 权重

epoch 45 权重没有丢失。训练器按选择角色保存为 `least_violation.pt`，而不是逐轮命名的 `epoch_045.pt`。现已为同一文件建立硬链接别名 `epoch_045.pt`，二者 inode、字节和 SHA256 完全相同：

- epoch：45
- update index：11970
- SHA256：`425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1`
- minADE：`0.13729323806597502`
- minFDE：`0.20226440469505041`
- JADE：`0.16520817553457082`
- JFDE：`0.2719628852020342`
- total marginal violation：`0.010956382495497019`

显式本地绑定收据为 `EPOCH_045_CHECKPOINT_BINDING.json`，SHA256 `246864e415cfa38c6b29f06815d43dfdf5a53c36fa777d7fb8e5afcc0678dc7b`。

## 单种子覆盖边界

既有 checkpoint 的 resolved config 保留 `minimum_training_seeds: 3`，因为该字段被提交 `9ecde66` 的配置校验器固定，并参与 checkpoint 身份。为了不破坏精确续训，未篡改已有 checkpoint 或模型配置。实际运行次数以新的队列 manifest 为准：`training_seeds: [3101]`、`minimum_training_seeds: 1`。

因此最终结果只构成单训练种子证据，不能再表述为三训练种子稳定性结论。五个推理种子仍可报告固定模型的采样均值和波动。

## 重复写入核验

进程命名空间差异曾导致一个新的恢复调度器短暂创建；发现宿主机原训练仍存活后已立即终止新增进程，保留原进程。对 epoch 48 后的正式日志检查得到：

- update `12769` 至 `12847` 共 79 条；
- 79 个 update index 全部唯一；
- 重复 update：0；
- 缺失 update：0。

因此没有把短暂进程重叠写入正式训练记录。

## UNIV 准备状态

UNIV 队列状态为 `WAITING`，仅包含 `seed 3101`。它不占用 GPU，等待 HOTEL 单种子队列完成后自动进行 CUDA 预检并启动。UNIV 的官方窗口导出、逐点轨迹/语义 parity 和匹配 GDTS baseline 绑定已完成；启动仍是 fail-closed，CUDA 预检不通过不会进入训练。
