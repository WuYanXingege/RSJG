# 新版 gated8 执行提速与固定终点的能力边界

**采用小场景去噪图保留策略。真实 SDD 短窗平均由 3.0374 降至 2.1535 秒/update，
提速 1.410×、时间减少 29.10%。本轮没有启动正式训练或改变终点模型。**

## 实际采用的修改

新增可选运行时字段 `training.denoiser_uncheckpointed_max_agent_slots: 16`。
仅 gated8、训练模式、固定终点 `eta=0`、单 world、物理组 `B×P×N≤16` 时，
保留八个去噪块的完整计算图，减少 backward 的 checkpoint 重算。
超过阈值仍使用原 checkpoint；显式 `checkpoint_blocks` 参数优先。
地图 U-Net checkpoint、CTF 可微地图采样、损失、增强、逻辑 batch、累积、
FP32/BF16 策略、优化器和终点候选均不变。没有新增可训练参数或 state_dict 键。

字段默认 0，历史配置继续走原路径。16 是本轮保守执行阈值，不自动扩大。
已测范围为 SDD、新版 192/4/768、4+4 层 gated8；没有外推其他折、旧六层或 rollout。

## 实测与数值验收

RTX 5070 Ti 16 GiB，PyTorch 2.8.0+cu128；正式 unified 入口的 BF16 前反向及 FP32 损失。
同折已验证 U-Net 初始化，其余网络共享一次新的初始化；复用训练折拟合尺度和正式 train bank。
逻辑 microbatch=4 场景，累积4次；保持逐场景物理计算，因此每次 update 覆盖16个场景。
原 CPU rigid 输入路径、4 workers。没有把 20-step 推理当作普通训练 update。

| 配对轮次 | 原路径 s/update | 优化 s/update | 提速 |
|---|---:|---:|---:|
| 1：原→优化 | 2.9945 | 2.1753 | 1.377× |
| 2：优化→原 | 3.0803 | 2.1317 | 1.445× |
| 两轮平均 | 3.0374 | 2.1535 | **1.410×** |

每臂预热3次，计时16次完整 optimizer update；两轮使用同一序列并交叉执行顺序。
计时包含读取等待、CPU增强、传输、全部前反向及更新；GPU仅在区间边界同步。
计时区间包含256个独立训练窗口，人数1–18、平均1.546875；地图6×256×256，
逻辑组最多72个padding槽位。不是只测单人，也没有截断行人。

- 8次配对更新：输入/增强及扩散随机数哈希一致；loss、梯度范数差为0。
- 参数相对L2差约 `4.28e-10`，与基线重复对照相同量级；未声称所有优化器状态逐位相同。
- encoder、coupling、denoiser均保留梯度；训练损失和梯度有限。
- 最终代码接入另做2次配对更新：loss及梯度范数仍一致，参数相对L2差 `2.36e-10`。
- CPU定向检查覆盖真实策略边界、显式覆盖优先、map checkpoint不变、strict-load，
  以及FP32 forward/backward一致性。完整测试收据见 `VALIDATION.json`。
- 原路径峰值 allocated 919.2–921.4 MiB，优化约919.0 MiB；样本中的大场景回退影响峰值。
  小场景保留图会增加局部活跃张量，不能据此宣称任何场景都零显存代价。
- 主进程RSS高水位约3.1 GiB；不是各worker总内存或精确增量。

本轮没有测完整epoch、验证或保存耗时，也没有验证新版8层精度优于旧6层。
短测使用两臂一致的固定LR（主体1e-4、U-Net 1e-5），不代表完整调度训练结果。

## 定位依据与未采用项

一次独立采样计时（带同步开销，不纳入上述提速数字）：

| 部分 | 秒/update |
|---|---:|
| 读取/拼接等待 | 0.010 |
| CPU增强 | 0.141 |
| MSCE forward | 0.183 |
| coupling forward | 0.021 |
| denoiser forward | 0.731 |
| backward（含重算） | 1.991 |
| optimizer apply | 0.005 |

这些分项没有穷尽传输、损失及Python开销，不能作为完整profile。
性能窗实际读取等待约0.008秒/update。当前主要瓶颈是反向/去噪及串行小场景执行，
不是此前GDTS的密集热图输入管线。

日志标量延迟回传＋紧凑mask虽数值一致，但两轮提速只有1.001×/0.998×，未接入正式代码。
整microbatch计算在数值门控中产生最大2.07% loss偏差；保留逐场景编码/配对、
只合并后段去噪后仍有1.96%偏差，均拒绝采用。差异几乎集中在pair loss；
离散指派对浮点扰动敏感是待核实解释，没有据此宣布网络实现错误。
失败原型只保留于基准脚本及收据，不进入模型/正式训练路径。

## minFDE 为何随桥训练不变

当前不是把指标硬编码为一个常数。对固定checkpoint提供的同一候选bank，
每个行人有20个终点 `G[k,i]`；H只改变排列：`H[k,i] = G[π_i(k),i]`。
`decode_conditioned(..., eta=0)` 将最后一帧严格拼接为条件终点。
因此对每个有效行人：

`min_k ||H[k,i] - GT[i]|| = min_k ||G[k,i] - GT[i]||`。

相同bank、成员、单位和评测seed下，G/H的逐人minFDE及其数据集均值不随桥参数变化。
不同bank/提供器权重/seed可以改变这个值。JADE/JFDE选一个共同world，仍能受配对影响；
minADE和轨迹碰撞也能受桥路径影响。训练中的MSCE地图头不等于冻结的外部终点提供器，
改善其BCE不能直接改变bank中的G。

这是有明确用途的能力边界：能保护强终点覆盖、把联合组织和路径改进归因清楚；
但不能让桥网络修复终点先验自身的遗漏或误差。**增加6层到8层也不会解除这个限制。**
若论文主张联合预测改善，固定终点合理；若主张通过桥网络学习进一步降低逐人minFDE，
当前设计不支持该主张。不能把它隐藏成“训练后还会自然下降”。

当前桥以GT终点条件训练，不能直接把推理eta改成1就期望得到可靠终点残差。
唯一优先结构方向是**单独改进可学习终点先验的候选覆盖**，保留现有固定终点桥作对照；
本轮未实施这种研究语义修改，也未启动新的结构实验。

## 使用与成本

干净目录 `network_release/ebjd_gated_bridge_v1/` 已同步该执行选项。
SDD gated8准备命令可增加：

```bash
python prepare.py --base-config "$BASE_CONFIG" --profile gated8 \
  --config-dir "$NEW_CONFIG_DIR" --run-output "$NEW_RUN_DIR" \
  --denoiser-uncheckpointed-max-agent-slots 16
```

训练命令仍保留 `--compact-scenes`。原正式配置/权重未修改，旧运行恢复仍须遵循既有源码和配置绑定。
本轮只准备新配置，没有启动该训练命令。

复现主要基准（在论文根目录，已验证环境，GPU空闲时）：

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python \
  scripts/benchmark_gated_training_runtime.py \
  --config "$BENCH_CONFIG" --train-bank "$TRAIN_BANK" --output "$NEW_BENCH_DIR" \
  --candidate small_denoiser_graph --updates 16 --max-gpu-seconds 350 --skip-profile
```

`BENCH_CONFIG` 使用本轮资产目录的 `CONFIG.json`（含已验证的训练折尺度）。
全部原始短测结果保存在 `RSJG_EBJD_assets/experiments/ebjd_gated_runtime_20261113_202549/`。
来源和SHA256见 `SUMMARY.json`，小型时间表见 `TIMING.csv`。
累计GPU占用墙钟 **858.74秒（14.31分钟）**，包括预热、重复对照、失败候选和最终接入检查。
共295次临时optimizer更新，正式更新=0，没有保存/覆盖正式checkpoint。
