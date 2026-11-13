# ETH gated8 启动合同与提速验收

任务目录：`RSJG_EBJD_assets/experiments/eth_gated8_unified_v1_20261113_214832/`。
实际后台状态以该目录的 `PIPELINE_STATE.json`、`LAUNCH_RECEIPT.json` 和
`training/train.jsonl` 为准；本文件不把配置完成等同于训练完成。

## 固定设置

- ETH留出折，training seed3101；主网络新初始化，使用已认证的ETH同折map U-Net初始化。
- 192/4/768，4个coarse＋4个refinement的endpoint-gated块；FULL/CTF/Raw。
- 冻结原生终点＋可学习联合配对＋固定终点条件桥；Brownian表示及原损失不变。
- 最多30轮；当前统一early-stop合同：最早10轮、8次验证patience、H_JADE、相对显著改善0.1%。
  原六层ETH记录为30轮固定运行，本次采用当前新网络入口的统一早停合同，已登记该差异。
- 每轮验证，20 worlds / 20 DDIM；rollout/projection全程OFF。
- microbatch4场景、梯度累积4次；逐场景物理计算，4 workers，legacy_cpu_rigid输入。
- AdamW；主体1e-4、U-Net 1e-5，warmup5轮、30轮余弦时间轴、最低比例0.1。
- 结束后原selector选best_eligible，否则best_jade；同一选中权重自动评测2035–2039。
  没有eligible时不宣称四指标合同通过。

训练4110窗口/36906行人出现；validation/test各253窗口/364行人出现，
平均1.4387人/留出窗口，成员镜像，不是独立测试集。没有截断行人或为了人数标签抽样。
复用旧统一ETH的同一train bank和五份validation bank，逐份SHA与旧RUN_IDENTITY一致。
场景、行人、frame和GT终点对齐已核对；不重新导出数据或bank。

## ETH提速实际结果

采用 `training.denoiser_uncheckpointed_max_agent_slots=16`。
小组不重算去噪块，大组保留checkpoint；地图U-Net checkpoint不变。

| 轮次 | gated8原路径 s/update | 优化 s/update | 提速 |
|---|---:|---:|---:|
| 1 | 4.2910 | 3.5080 | 1.223× |
| 2（反转顺序） | 4.3536 | 3.5442 | 1.228× |

平均1.226×，耗时减少约18.4%；仅相对同款8层原执行路径，不是相对旧6层或GDTS。
每臂预热2次、计时8次完整update，两轮使用同一真实序列。
128个计时窗口平均8.34375人、最多45人，22个窗口触发>16槽位回退；地图6×256×256。
峰值allocated约1.75 GiB；不是整个正式训练最大人数57的精确峰值上界。

连续8步的基线重复自身出现1.069% loss偏差，候选约0.289%；不能声称逐训练轨迹一致。
因此增加同父状态验收：每一步从共同参考模型、optimizer和RNG恢复，保留输入与增强顺序，
逐步loss/梯度范数差为0；候选末步参数相对L2差约3.10e-11。
仍使用loss 0.1% / 梯度范数1%的原门槛，没有按候选结果放宽。
该检查用于执行方式验收，长期指标需看本次真实训练，不用短测代替精度结论。
两次检查合计GPU墙钟479.21秒，所有短测状态均丢弃，未带入正式训练。

机器可读证据见 `RUNTIME_ACCEPTANCE.json`；完整中间结果留在任务资产目录。
其他ETH/UCY折没有在本轮重新训练，也没有据此宣称相同提速。

## 后台、恢复及结果边界

`scripts/run_gated_release_background.py` 使用独立run锁和GPU锁，启动前检查主机GPU。
由nohup＋setsid脱离会话，训练/评测子进程串行。不会因聊天结束而主动停止。
不自动重试失败；手动重启时只从同一release/config的完整last.pt恢复。
没有完整checkpoint的非空输出拒绝静默从头重训。手动入口保存在资产目录 `RESUME.sh`。
启动命令、源码HEAD、release摘要、配置和bank哈希均保存在RUN_MANIFEST及收据。

本次冻结终点bank的selection-seed minFDE约0.581859m，matched GDTS约0.580032m。
固定终点设计使8层桥训练不能消除这项差距；本次主要检验路径及联合指标，
不改baseline或eligible门槛。保留best_jade不等于声称best_eligible。
