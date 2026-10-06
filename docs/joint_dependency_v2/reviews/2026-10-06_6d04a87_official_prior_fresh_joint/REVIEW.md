# 官方语义先验 fresh joint 独立配置与验收

结论：fresh joint 已达到 **JOINT_TRAINING_READY**，并在发布 commit `9c593013eba22f5523f4e335544937e5a308495d` 上后台启动。随后因主实验恢复标准 HOTEL 协议而在完整 epoch 4 后停止；当前状态为 **FRESH_JOINT_STOPPED_AFTER_EPOCH4**。父权重严格绑定 fresh goal 的 selected epoch 96；仅加载 `goal_module`，history/registrar 与 diffusion 按 seed 3101 全新初始化。CUDA 两次真实更新、完整状态回读和 production full-inner 选模均通过。cache、A0、A1 未启动。

本轮沿用官方语义先验协议 `p2_grouped_univ_hotel_official_prior_v1`。它接受并冻结 GDTS/Goal-SAR 作者分发的语义栅格作为环境输入，但不把未知的分割器训练范围重新表述为 strict-clean 来源认证。data binding 为 `c17a6f031f0ffb1d94011f85bb8981aaef518b96863fbd80a51159a9d9361bff`。

## 绑定与初始化

| 项目 | 验收结果 |
|---|---|
| goal 父权重 | epoch 96；文件 SHA256 `f6a52cf228d733bd684aef043d843ba7d942e40495d5eec354ce1e4e3507ec13`；模型 state SHA256 `c8edfe48b142985515bc457606c4eef9e8ef056cf0f98a9b4305e1f83d93bf21` |
| 加载范围 | 仅 `goal_module.*`；strict load 且加载后 state identity 相同 |
| fresh 范围 | history/registrar 与 diffusion 从 seed 3101 新建；non-goal state SHA256 `395d5344c4d2c0368382af93d0018a947b12b861ecd1a1047a227285652bd3de` |
| 联合初态 | 未更新 combined state SHA256 `042479866b6b5cc5869a0958c8d6c212bbab963f59329462b7e7aa19c7e8193d` |
| 目标 | `1 × diffusion_loss + 20 × goal_BCE_loss` |
| goal 参数 | 不冻结；随联合目标继续训练 |

权威初态 receipt 为 [JOINT_FRESH_INITIAL.json](JOINT_FRESH_INITIAL.json)。manifest SHA256 为 `eb9831f475915ba9c2e3c863f655d6583e93e5bc962d4f763a63f1b61b8ef873`，冻结配置 SHA256 为 `125569a09db7da6be362f25d5625e78514f9d8efae65d34b3946f87553296146`。

## 实际执行验收

| 项目 | 结果 |
|---|---|
| 核心 CPU 回归 | 60 passed；覆盖 grouped production 路径、官方先验入口与本轮 joint 合同 |
| CUDA 更新 | FP32、batch64，真实 2 次 optimizer step；`goal_module.`、`registrar.`、`diffnet.` 每步均有非零梯度且参数均改变 |
| checkpoint reload | model、optimizer、scheduler、CPU/CUDA RNG exact；固定 replay 最大绝对差 0 |
| 完整 inner | 391/391 packs、24955/24955 样本、P20 原生像素 ADE；0 optimizer updates；target 在预测后读取；outer 未访问 |
| 生产入口 | `main.py --p2_launch_check` exit 0 |
| 聚合门禁 | [JOINT_READINESS.json](JOINT_READINESS.json) = `JOINT_TRAINING_READY`，9/9 checks true |

最终 receipt 绑定的 `src/p2_grouped_training.py` SHA256 为 `0abbdaf455ce949b6f7220c2c7bc567d967ffc6746e447c38cbd1c2a2f8bd2df`。完整 inner 实测 998.95 秒，工程 smoke ADE 为 104.323854 像素；该值来自仅更新两步的 `SMOKE_ONLY` checkpoint，只证明选模链路完整，**不是模型性能或收敛结果**。

## 正式训练合同

- 原 GDTS baseline，FP32，seed 3101，batch 64，Adam lr `1e-4`，clip 1，ExponentialLR gamma `0.99`。
- 上限 250 epochs / 43,750 successful updates / 48 小时。
- 每个完整 epoch 后运行完整 inner，指标为原生像素 ADE、P20；只有严格更小值替换 best，相等时保留更早 epoch。
- 48 小时边界在 pack 或 inner batch 前检查。若已存在完整 epoch，丢弃未完成的当前 epoch/selection，以最后一个完整 best 原子 checkpoint 收尾；若一个完整 epoch 都没有，则明确失败且不生成合格父产物。
- measured full-inner 单次约 16.6 分钟，因此 250 epoch 只是硬上限，48 小时可能先到；最终实际 epoch/update 数必须以 `run_complete.json` 为准。
- 本启动器只运行 fresh joint。合格 base 产生后，cache 与 A0/A1 仍需独立授权和验收。

## 结论边界

- 本轮没有访问 outer target 或计算 outer metric。
- goal 阶段 epoch 1 已包含 175 次更新；不把它称为随机初态，也不从 goal BCE 的小幅差异推断终点定位或轨迹收益。
- 本验收没有进入 A1 配对目标，没有修改 GDTS 网络结构，也没有把官方语义先验协议升级为 strict-clean 数据来源认证。
- 历史结果、旧 checkpoint 和既有审查均保留；运行中的正式 checkpoint、数据与日志不提交 GitHub。

机器汇总见 [RESULTS.json](RESULTS.json)，可复现命令与资源边界见 [COMMANDS_AND_BUDGET.md](COMMANDS_AND_BUDGET.md)。

启动快照见 [LAUNCH_SNAPSHOT.json](LAUNCH_SNAPSHOT.json)：GitHub 远端 17/17 个提交变更文件逐字节回读一致；worker PID/SID `1115621`，正式初态与验收 SHA 完全一致。快照时 epoch 1 的 175 个 train packs、175 次成功更新和 11,122 exposures 已完成，完整 inner selection 正在运行；尚未把 epoch 1 记为完整 checkpoint。
